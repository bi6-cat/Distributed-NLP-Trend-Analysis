"""
spark_jobs/cleaning_job.py — M2: Spark Cleaning, Schema Normalisation & Dedup Job

Luồng:
    HDFS /user/zett/raw_data/
        ├── voz/comments.csv + posts.csv      → VozAdapter
        ├── vatvo/articles.csv                → VatVoAdapter
        └── vnexpress/post_vnexpress.csv
                    /comment_vnexpress.csv    → VnExpressAdapter
                                ↓
              TextPreprocessor.clean() (lazy init trong mapPartitions)
                                ↓
              MinHashDeduplicator (Spark MinHashLSH dedup)
                                ↓
    Output Parquet: /user/zett/staged/stg_posts_core/

Cách chạy trên cluster (từ master node):
    spark-submit \\
        --master spark://192.168.56.11:7077 \\
        --num-executors 2 \\
        --executor-cores 2 \\
        --executor-memory 4g \\
        --driver-memory 2g \\
        --py-files dist/nlp_trend.zip \\
        --conf spark.executorEnv.NLP_SLANG_DICT=hdfs:///user/zett/ref/slang_dict.json \\
        --conf spark.executorEnv.NLP_STOPWORDS=hdfs:///user/zett/ref/stopwords_vi.txt \\
        spark_jobs/cleaning_job.py

    # Bật MinHash LSH dedup (mặc định tắt — khớp build_stg_core):
        spark_jobs/cleaning_job.py --with-dedup

Biến môi trường (tuỳ chỉnh qua --conf spark.executorEnv.*):
    HDFS_BASE       : hdfs://192.168.56.11:9000
    NLP_SLANG_DICT  : path slang_dict.json trên HDFS
    NLP_STOPWORDS   : path stopwords_vi.txt trên HDFS
    HDFS_OUTPUT     : đường dẫn HDFS ghi kết quả Parquet
"""

import argparse
import logging
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone, date

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Date filter — khớp với build_stg_core.py ─────────────────────────────────
_DATE_FROM = date(2026, 4, 28)
_DATE_TO   = date(2026, 4, 30)

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, LongType, TimestampType
)

# ── Config (override bằng env var hoặc spark-submit --conf) ──────────────────
HDFS_BASE  = os.environ.get("HDFS_BASE",    "hdfs://namenode:9000")
HDFS_USER  = os.environ.get("HDFS_USER", os.environ.get("HADOOP_USER_NAME", "root"))
HDFS_HOME  = os.environ.get("HDFS_HOME", f"/user/{HDFS_USER}")
HDFS_RAW   = os.environ.get("HDFS_INPUT",   f"{HDFS_BASE}{HDFS_HOME}/raw_data")
HDFS_OUT   = os.environ.get(
    "HDFS_OUTPUT",
    os.environ.get("HDFS_STG_POSTS_CORE", f"{HDFS_BASE}{HDFS_HOME}/staged/stg_posts_core"),
)

SLANG_PATH = os.environ.get("NLP_SLANG_DICT", f"{HDFS_BASE}{HDFS_HOME}/ref/slang_dict.json")
STOP_PATH  = os.environ.get("NLP_STOPWORDS",  f"{HDFS_BASE}{HDFS_HOME}/ref/stopwords_vi.txt")
CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT = os.environ.get("CLICKHOUSE_PORT", "8123")
CLICKHOUSE_DB = os.environ.get("CLICKHOUSE_DB", "tech_radar")
CLICKHOUSE_USER = os.environ.get("CLICKHOUSE_USER", "root")
CLICKHOUSE_PASS = os.environ.get("CLICKHOUSE_PASS", "root")

# ── Output Schema (khớp với ClickHouse stg_posts_core) ───────────────────────
OUTPUT_SCHEMA = StructType([
    StructField("post_id",        StringType(),  False),
    StructField("source",         StringType(),  False),
    StructField("author",         StringType(),  True),
    StructField("title",          StringType(),  True),
    StructField("body",           StringType(),  True),   # clean_html() output
    StructField("clean_text",     StringType(),  True),   # clean() output
    StructField("segmented_text", StringType(),  True),   # preprocess() output
    StructField("parent_id",      StringType(),  True),
    StructField("reaction_count", LongType(), True),
    StructField("view_count",     LongType(), True),
    StructField("comment_count",  LongType(), True),
    StructField("created_at",     TimestampType(), True), 
    StructField("crawled_at",     TimestampType(), False),
])


def _execute_clickhouse_sql(query: str) -> None:
    """Execute SQL against ClickHouse over HTTP without extra module dependencies."""
    url = (
        f"http://{CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}/"
        f"?database={urllib.parse.quote(CLICKHOUSE_DB)}"
        f"&user={urllib.parse.quote(CLICKHOUSE_USER)}"
        f"&password={urllib.parse.quote(CLICKHOUSE_PASS)}"
    )
    payload = query.encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")

    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status >= 400:
            body = response.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ClickHouse HTTP {response.status} while executing SQL: {body[:300]}"
            )


# ── Worker function (chạy trên mỗi Spark executor) ───────────────────────────
def process_voz_comments(iterator, slang_path: str, stop_path: str, crawled_ts: int):
    """
    Lazy init: import Adapter + TextPreprocessor bên trong worker.
    Nhận iterator các Spark Row (VOZ comments CSV).
    Yield tuple theo OUTPUT_SCHEMA.
    """
    import sys
    import os as _os

    # Đảm bảo worker tìm được package preprocessing/ và schemas/
    try:
        from pyspark import SparkFiles
        root = SparkFiles.getRootDirectory()
        if root not in sys.path:
            sys.path.insert(0, root)
    except Exception:
        pass

    from schemas.voz_adapter import VozAdapter
    from preprocessing.text_cleaner import TextPreprocessor

    adapter      = VozAdapter()
    preprocessor = TextPreprocessor(
        slang_dict_path=_resolve_hdfs(slang_path),
        stopwords_path =_resolve_hdfs(stop_path),
        use_vncorenlp=False,
    )

    rows = list(iterator)
    raw_dicts = [r.asDict() for r in rows]
    df = adapter.comments_to_df(raw_dicts)

    for _, row in df.iterrows():
        content = str(row.get("comment", "") or "")
        body    = preprocessor.clean_html(content)
        clean_t = preprocessor.clean(content)
        seg     = preprocessor.preprocess(content)
        yield (
            str(row.get("comment_id", "")),
            str(row.get("source", "voz")),
            str(row.get("user", "") or ""),
            None,                                        # title
            body,
            clean_t,
            seg,
            str(row.get("id_post", "") or ""),           # parent_id
            int(row.get("reaction_count", 0) or 0),
            None,                                        # view_count
            None,                                        # comment_count
            _to_datetime(row.get("created_at")),
            _to_datetime(crawled_ts),
        )


def process_voz_posts(iterator, slang_path: str, stop_path: str, crawled_ts: int):
    """Xử lý VOZ posts CSV."""
    import sys

    try:
        from pyspark import SparkFiles
        root = SparkFiles.getRootDirectory()
        if root not in sys.path:
            sys.path.insert(0, root)
    except Exception:
        pass

    from schemas.voz_adapter import VozAdapter
    from preprocessing.text_cleaner import TextPreprocessor

    adapter      = VozAdapter()
    preprocessor = TextPreprocessor(
        slang_dict_path=_resolve_hdfs(slang_path),
        stopwords_path =_resolve_hdfs(stop_path),
        use_vncorenlp=False,
    )

    rows = list(iterator)
    raw_dicts = [r.asDict() for r in rows]
    df = adapter.posts_to_df(raw_dicts)

    for _, row in df.iterrows():
        content = str(row.get("title", "") or "")
        body    = preprocessor.clean_html(content)
        clean_t = preprocessor.clean(content)
        seg     = preprocessor.preprocess(content)
        yield (
            str(row.get("id_post", "")),
            str(row.get("source", "voz")),
            str(row.get("author_name", "") or ""),
            content,                                     # title
            body,
            clean_t,
            seg,
            None,                                        # parent_id (post gốc)
            None,                                        # reaction_count
            _safe_int(row.get("view_count")),
            _safe_int(row.get("comment_count")),
            _to_datetime(row.get("created_at")),
            _to_datetime(crawled_ts),
        )


def process_vatvo(iterator, slang_path: str, stop_path: str, crawled_ts: int):
    """Xử lý VatVo articles CSV."""
    import sys

    try:
        from pyspark import SparkFiles
        root = SparkFiles.getRootDirectory()
        if root not in sys.path:
            sys.path.insert(0, root)
    except Exception:
        pass

    from schemas.vatvo_adapter import VatVoAdapter
    from preprocessing.text_cleaner import TextPreprocessor

    adapter      = VatVoAdapter()
    preprocessor = TextPreprocessor(
        slang_dict_path=_resolve_hdfs(slang_path),
        stopwords_path =_resolve_hdfs(stop_path),
        use_vncorenlp=False,
    )

    rows = list(iterator)
    raw_dicts = [r.asDict() for r in rows]
    df = adapter.articles_to_df(raw_dicts)

    for _, row in df.iterrows():
        content = str(row.get("content", "") or "")
        body    = preprocessor.clean_html(content)
        clean_t = preprocessor.clean(content)
        seg     = preprocessor.preprocess(content)
        yield (
            str(row.get("post_id", "")),
            str(row.get("source", "vatvo")),
            str(row.get("author", "") or ""),
            str(row.get("title", "") or ""),
            body,
            clean_t,
            seg,
            None,                                        # parent_id
            None,                                        # reaction_count
            None,                                        # view_count
            None,                                        # comment_count
            _to_datetime(row.get("created_at")),
            _to_datetime(crawled_ts),
        )


def process_vnexpress_posts(iterator, slang_path: str, stop_path: str, crawled_ts: int):
    """Xử lý VnExpress posts CSV."""
    import sys

    try:
        from pyspark import SparkFiles
        root = SparkFiles.getRootDirectory()
        if root not in sys.path:
            sys.path.insert(0, root)
    except Exception:
        pass

    from schemas.vnexpress_adapter import VnExpressAdapter
    from preprocessing.text_cleaner import TextPreprocessor

    adapter      = VnExpressAdapter()
    preprocessor = TextPreprocessor(
        slang_dict_path=_resolve_hdfs(slang_path),
        stopwords_path =_resolve_hdfs(stop_path),
        use_vncorenlp=False,
    )

    rows = list(iterator)
    raw_dicts = [r.asDict() for r in rows]
    df = adapter.posts_to_df(raw_dicts)

    for _, row in df.iterrows():
        content = str(row.get("body", "") or "")
        body    = preprocessor.clean_html(content)
        clean_t = preprocessor.clean(content)
        seg     = preprocessor.preprocess(content)
        yield (
            str(row.get("post_id", "")),
            str(row.get("source", "vnexpress")),
            str(row.get("author", "VnExpress")),
            str(row.get("title", "") or ""),
            body,
            clean_t,
            seg,
            None,                                        # parent_id
            _safe_int(row.get("reaction_count")),
            _safe_int(row.get("view_count")),
            _safe_int(row.get("comment_count")),
            _to_datetime(row.get("created_at")),
            _to_datetime(crawled_ts),
        )


def process_vnexpress_comments(iterator, slang_path: str, stop_path: str, crawled_ts: int):
    """Xử lý VnExpress comments CSV."""
    import sys

    try:
        from pyspark import SparkFiles
        root = SparkFiles.getRootDirectory()
        if root not in sys.path:
            sys.path.insert(0, root)
    except Exception:
        pass

    from schemas.vnexpress_adapter import VnExpressAdapter
    from preprocessing.text_cleaner import TextPreprocessor

    adapter      = VnExpressAdapter()
    preprocessor = TextPreprocessor(
        slang_dict_path=_resolve_hdfs(slang_path),
        stopwords_path =_resolve_hdfs(stop_path),
        use_vncorenlp=False,
    )

    rows = list(iterator)
    raw_dicts = [r.asDict() for r in rows]
    df = adapter.comments_to_df(raw_dicts)

    for _, row in df.iterrows():
        content = str(row.get("body", "") or "")
        body    = preprocessor.clean_html(content)
        clean_t = preprocessor.clean(content)
        seg     = preprocessor.preprocess(content)
        yield (
            str(row.get("post_id", "")),
            str(row.get("source", "vnexpress")),
            str(row.get("author", "") or ""),
            None,                                        # title
            body,
            clean_t,
            seg,
            str(row.get("parent_id", "") or ""),
            _safe_int(row.get("reaction_count")),
            None,                                        # view_count
            None,                                        # comment_count
            _to_datetime(row.get("created_at")),
            _to_datetime(crawled_ts),
        )


# ── Helper functions ──────────────────────────────────────────────────────────

def _safe_int(val) -> int | None:
    """
    Chuyển giá trị sang int, trả None nếu không hợp lệ.
    Xử lý được "Views\\n13,430" → 13430 (lấy số cuối cùng, khớp build_stg_core).
    """
    if val is None:
        return None
    s = str(val).replace(",", "").strip()
    nums = re.findall(r"\d+", s)
    if not nums:
        return None
    try:
        return int(nums[-1])
    except ValueError:
        return None


def _to_datetime(ts_val) -> datetime | None:
    """
    Unix timestamp int/float → datetime UTC. Trả None nếu không hợp lệ.
    Adapter luôn trả về int timestamp — không trả Epoch giả khi None
    để tránh ghi dữ liệu rác vào ClickHouse.
    """
    if ts_val is None:
        return None
    try:
        return datetime.fromtimestamp(float(str(ts_val)), tz=timezone.utc)
    except (ValueError, TypeError, OSError):
        return None




def _resolve_hdfs(path: str) -> str:
    """
    Nếu path là HDFS (hdfs://...), copy về /tmp trên worker trước khi dùng.
    Cần thiết vì TextPreprocessor đọc file bằng open() — không hiểu HDFS URI.
    """
    import os
    import subprocess

    if not path.startswith("hdfs://"):
        return path

    filename = path.split("/")[-1]
    local    = f"/tmp/{filename}"

    if not os.path.exists(local):
        try:
            subprocess.run(["hdfs", "dfs", "-get", path, local],
                           check=True, capture_output=True)
        except Exception as e:
            print(f"[WARN] Không copy được {path} về /tmp: {e}")
    return local


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments được truyền sau `spark-submit ... cleaning_job.py`."""
    parser = argparse.ArgumentParser(description="M2 Spark Cleaning & Dedup Job")
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        default=True,
        help="Bỏ qua bước MinHash LSH deduplication (mặc định: True — khớp build_stg_core)",
    )
    parser.add_argument(
        "--with-dedup",
        action="store_true",
        default=False,
        help="Bật MinHash LSH deduplication (tắt theo mặc định)",
    )
    # spark-submit truyền thêm các args không liên quan — bỏ qua
    args, _ = parser.parse_known_args()
    return args


def main():
    # Fix encoding tiếng Việt trên terminal Windows
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    cli_args = _parse_args()
    # --with-dedup override --no-dedup nếu cả 2 được truyền cùng lúc
    enable_dedup = cli_args.with_dedup and not cli_args.no_dedup

    spark = SparkSession.builder \
        .appName("M2_CleaningJob") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    crawled_ts = int(datetime.now(timezone.utc).timestamp())

    print(f"[INFO] HDFS RAW    : {HDFS_RAW}")
    print(f"[INFO] HDFS OUTPUT : {HDFS_OUT}")
    print(f"[INFO] Slang dict  : {SLANG_PATH}")
    print(f"[INFO] Stopwords   : {STOP_PATH}")

    # ── Đọc 5 file CSV từ HDFS ───────────────────────────────────────────────
    sc = spark.sparkContext
    read_csv = lambda path: (
        spark.read
             .option("header", "true")
             .option("multiLine", "true")
             .option("quote", '"')
             .option("escape", '"')
             .option("encoding", "UTF-8")
             .csv(path)
             .repartition(sc.defaultParallelism * 2)  # Ép Spark chia nhỏ data để chạy trên nhiều Worker
    )

    voz_comments_raw = read_csv(f"{HDFS_RAW}/voz/comments.csv")
    voz_posts_raw    = read_csv(f"{HDFS_RAW}/voz/posts.csv")
    vne_posts_raw    = read_csv(f"{HDFS_RAW}/vnexpress/post_vnexpress.csv")
    vne_cmts_raw     = read_csv(f"{HDFS_RAW}/vnexpress/comment_vnexpress.csv")

    # ── mapPartitions → process qua Adapter + TextPreprocessor ───────────────

    voz_c_rdd  = voz_comments_raw.rdd.mapPartitions(
        lambda it: process_voz_comments(it, SLANG_PATH, STOP_PATH, crawled_ts))
    voz_p_rdd  = voz_posts_raw.rdd.mapPartitions(
        lambda it: process_voz_posts(it, SLANG_PATH, STOP_PATH, crawled_ts))
    vne_p_rdd  = vne_posts_raw.rdd.mapPartitions(
        lambda it: process_vnexpress_posts(it, SLANG_PATH, STOP_PATH, crawled_ts))
    vne_c_rdd  = vne_cmts_raw.rdd.mapPartitions(
        lambda it: process_vnexpress_comments(it, SLANG_PATH, STOP_PATH, crawled_ts))

    # ── Union tất cả nguồn → DataFrame duy nhất ──────────────────────────────
    all_rdd = sc.union([voz_c_rdd, voz_p_rdd, vne_p_rdd, vne_c_rdd])
    result_df = spark.createDataFrame(all_rdd, OUTPUT_SCHEMA)

    from pyspark.sql.functions import col, expr, lit, to_date, when

    # ── Fix 1: Lọc body rỗng ─────────────────────────────────────────────────
    result_df = result_df.filter(
        result_df.body.isNotNull() & (result_df.body != "")
    )

    # ── Fix 2: Lọc date range [DATE_FROM, DATE_TO] — khớp build_stg_core ─────
    # created_at = None (adapter skip) bị loại, tránh ghi Epoch vào ClickHouse
    date_from_lit = lit(str(_DATE_FROM))
    date_to_lit   = lit(str(_DATE_TO))
    result_df = result_df.filter(
        col("created_at").isNotNull()
        & (to_date(col("created_at")) >= date_from_lit)
        & (to_date(col("created_at")) <= date_to_lit)
    )
    logger.info(f"[DateFilter] Giữ lại records từ {_DATE_FROM} đến {_DATE_TO}")

    # ── Fix 3: Xoá trùng lặp 100% ────────────────────────────────────────────
    result_df = result_df.dropDuplicates()

    # ── Fix 4: Cấp UUID cho post_id = None / "None" ───────────────────────────
    result_df = result_df.withColumn(
        "post_id",
        when(col("post_id").isNull() | (col("post_id") == "None"), expr("uuid()")).otherwise(col("post_id"))
    )

    # cache() trước khi làm include_post_ids join và orphan filter
    # uuid() là non-deterministic — cache tránh re-evaluate
    result_df = result_df.cache()

    # ── Fix 5: Include posts có comment trong range (OR logic) ────────────────
    # Một post có thể được tạo trước DATE_FROM nhưng vẫn nhận comment trong range
    # → giữ lại post đó để orphan filter không xoá comment hợp lệ
    comments_df = result_df.filter(col("parent_id").isNotNull())
    parent_ids  = comments_df.select(col("parent_id").alias("post_id")).distinct()

    # Posts nằm ngoài range nhưng có comment trong range → thêm lại từ raw
    # (các post này đã bị lọc ở Fix 2 nếu created_at ngoài range)
    # Giải pháp: nới lỏng filter cho posts (parent_id IS NULL) — chỉ yêu cầu
    # post_id xuất hiện trong parent_ids của comments đã lọc
    posts_in_range = result_df.filter(col("parent_id").isNull())

    # Đọc lại posts raw để lấy những post ngoài range nhưng có comment trong range
    extra_posts_rdd = sc.union([
        voz_posts_raw.rdd.mapPartitions(
            lambda it: process_voz_posts(it, SLANG_PATH, STOP_PATH, crawled_ts)),
        vne_posts_raw.rdd.mapPartitions(
            lambda it: process_vnexpress_posts(it, SLANG_PATH, STOP_PATH, crawled_ts)),
    ])
    all_posts_df = spark.createDataFrame(extra_posts_rdd, OUTPUT_SCHEMA) \
        .filter(col("body").isNotNull() & (col("body") != "")) \
        .filter(col("parent_id").isNull())

    # Lấy posts ngoài range nhưng post_id có trong parent_ids
    orphan_parent_posts = all_posts_df \
        .join(parent_ids, on="post_id", how="inner") \
        .filter(
            to_date(col("created_at")).isNull()
            | (to_date(col("created_at")) < date_from_lit)
            | (to_date(col("created_at")) > date_to_lit)
        )

    result_df = posts_in_range.union(orphan_parent_posts).union(comments_df)
    result_df = result_df.dropDuplicates(["post_id"])
    result_df = result_df.cache()

    # ── Fix 6: Xoá orphan comments (parent_id không có trong post_id) ─────────
    # Xảy ra sau dedup nếu post bị xoá nhưng comment giữ lại
    valid_post_ids = result_df.filter(col("parent_id").isNull()) \
                              .select(col("post_id").alias("_pid"))
    orphan_comments = result_df.filter(col("parent_id").isNotNull()) \
        .join(valid_post_ids, result_df["parent_id"] == col("_pid"), how="left_anti")
    n_orphan = orphan_comments.count()
    if n_orphan > 0:
        logger.warning(f"[Orphan] Xoá {n_orphan:,} orphan comments (post gốc không tồn tại)")
    result_df = result_df.filter(col("parent_id").isNull()).union(
        result_df.filter(col("parent_id").isNotNull())
                 .join(valid_post_ids, result_df["parent_id"] == col("_pid"), how="inner")
                 .drop("_pid")
    )
    result_df = result_df.cache()

    if enable_dedup:
        logger.info("[Dedup] Starting Spark MinHashLSH dedup...")

        from algorithms.minhash_dedup import MinHashDeduplicator
        from pyspark import StorageLevel

        # 1. Repartition BEFORE LSH (critical)
        target_partitions = spark.sparkContext.defaultParallelism * 3
        result_df = result_df.repartition(target_partitions)

        # 2. Persist to avoid recomputation / lineage explosion
        result_df = result_df.persist(StorageLevel.MEMORY_AND_DISK)

        # 3. Optional: checkpoint to cut lineage (VERY important for LSH)
        spark.sparkContext.setCheckpointDir(f"{HDFS_BASE}/tmp/spark-checkpoints")
        result_df = result_df.checkpoint()
        result_df.count() 

        # 4. Init deduplicator with safer params
        deduplicator = MinHashDeduplicator(
            num_perm=128,           # 128 bands
            num_rows=4,             # 4 rows per band → 512 total permutations
            threshold=0.85,
            ngram_size=5,           # character 5-grams
            num_features=1 << 18,
            text_col="clean_text",
            checkpoint_dir=f"{HDFS_BASE}/tmp/spark-checkpoints/dedup/",
            use_graphframes=False,
        )

        # 5. Run LSH
        result_df = deduplicator.fit_transform(result_df, spark)

        # 6. Repartition AFTER LSH to rebalance
        result_df = result_df.repartition(target_partitions)

        # 7. Persist final result
        result_df = result_df.persist(StorageLevel.MEMORY_AND_DISK)

        logger.info("[Dedup] Completed MinHashLSH dedup")


    # ── Ghi ra HDFS Parquet ───────────────────────────────────────────────────
    result_df.write \
        .mode("overwrite") \
        .partitionBy("source") \
        .parquet(HDFS_OUT)

    logger.info(f"[DONE] Đã ghi Parquet → {HDFS_OUT}")

    logger.info(
        "[INGEST] Skip direct ClickHouse ingest. "
        "Airflow must call scripts/hdfs_to_clickhouse.py stg_posts_core."
    )

    spark.stop()


if __name__ == "__main__":
    main()
