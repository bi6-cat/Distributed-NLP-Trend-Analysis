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

    # Tắt MinHash LSH dedup khi cần:
        spark_jobs/cleaning_job.py --no-dedup

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
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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
# ── Output Schema (khớp với ClickHouse stg_posts_core) ───────────────────────
OUTPUT_SCHEMA = StructType([
    StructField("post_id",        StringType(),  False),
    StructField("source",         StringType(),  False),
    StructField("author",         StringType(),  True),
    StructField("title",          StringType(),  True),
    StructField("body",           StringType(),  True),   # clean_html() output
    StructField("clean_text",     StringType(),  True),   # clean() output
    StructField("segmented_text", StringType(),  True),   # preprocess() output
    StructField("topic_text",     StringType(),  True),   # preprocess_for_topic() output
    StructField("parent_id",      StringType(),  True),
    StructField("reaction_count", LongType(), True),
    StructField("view_count",     LongType(), True),
    StructField("comment_count",  LongType(), True),
    StructField("created_at",     TimestampType(), True), 
    StructField("crawled_at",     TimestampType(), False),
])


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
        topic_t = preprocessor.preprocess_for_topic(content)
        yield (
            str(row.get("comment_id", "")),
            str(row.get("source", "voz")),
            str(row.get("user", "") or ""),
            None,                                        # title
            body,
            clean_t,
            seg,
            topic_t,
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
        topic_t = preprocessor.preprocess_for_topic(content)
        yield (
            str(row.get("id_post", "")),
            str(row.get("source", "voz")),
            str(row.get("author_name", "") or ""),
            content,                                     # title
            body,
            clean_t,
            seg,
            topic_t,
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
        topic_t = preprocessor.preprocess_for_topic(content)
        yield (
            str(row.get("post_id", "")),
            str(row.get("source", "vatvo")),
            str(row.get("author", "") or ""),
            str(row.get("title", "") or ""),
            body,
            clean_t,
            seg,
            topic_t,
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
        topic_t = preprocessor.preprocess_for_topic(content)
        yield (
            str(row.get("post_id", "")),
            str(row.get("source", "vnexpress")),
            str(row.get("author", "VnExpress")),
            str(row.get("title", "") or ""),
            body,
            clean_t,
            seg,
            topic_t,
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
        topic_t = preprocessor.preprocess_for_topic(content)
        yield (
            str(row.get("post_id", "")),
            str(row.get("source", "vnexpress")),
            str(row.get("author", "") or ""),
            None,                                        # title
            body,
            clean_t,
            seg,
            topic_t,
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
        default=False,
        help="Bỏ qua bước MinHash LSH deduplication (mặc định: dedup đang bật)",
    )
    parser.add_argument(
        "--with-dedup",
        action="store_true",
        default=True,
        help="Giữ tương thích cờ cũ; dedup hiện được bật theo mặc định",
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
    enable_dedup = not cli_args.no_dedup

    spark = SparkSession.builder \
        .appName("M2_CleaningJob") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    crawled_ts = int(datetime.now(timezone.utc).timestamp())

    print(f"[INFO] HDFS RAW    : {HDFS_RAW}")
    print(f"[INFO] HDFS OUTPUT : {HDFS_OUT}")
    print(f"[INFO] Slang dict  : {SLANG_PATH}")
    print(f"[INFO] Stopwords   : {STOP_PATH}")

    # ── Đọc raw CSV từ HDFS ──────────────────────────────────────────────────
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
    vatvo_raw        = read_csv(f"{HDFS_RAW}/vatvo/articles.csv")
    vne_posts_raw    = read_csv(f"{HDFS_RAW}/vnexpress/post_vnexpress.csv")
    vne_cmts_raw     = read_csv(f"{HDFS_RAW}/vnexpress/comment_vnexpress.csv")

    # ── mapPartitions → process qua Adapter + TextPreprocessor ───────────────

    voz_c_rdd  = voz_comments_raw.rdd.mapPartitions(
        lambda it: process_voz_comments(it, SLANG_PATH, STOP_PATH, crawled_ts))
    voz_p_rdd  = voz_posts_raw.rdd.mapPartitions(
        lambda it: process_voz_posts(it, SLANG_PATH, STOP_PATH, crawled_ts))
    vatvo_rdd  = vatvo_raw.rdd.mapPartitions(
        lambda it: process_vatvo(it, SLANG_PATH, STOP_PATH, crawled_ts))
    vne_p_rdd  = vne_posts_raw.rdd.mapPartitions(
        lambda it: process_vnexpress_posts(it, SLANG_PATH, STOP_PATH, crawled_ts))
    vne_c_rdd  = vne_cmts_raw.rdd.mapPartitions(
        lambda it: process_vnexpress_comments(it, SLANG_PATH, STOP_PATH, crawled_ts))

    # ── Union tất cả nguồn → DataFrame duy nhất ──────────────────────────────
    all_rdd = sc.union([voz_c_rdd, voz_p_rdd, vatvo_rdd, vne_p_rdd, vne_c_rdd])
    result_df = spark.createDataFrame(all_rdd, OUTPUT_SCHEMA)

    from pyspark.sql.functions import col, expr, trim, when

    def _log_stage_count(label: str, df) -> None:
        logger.info("[CleanAudit] %s: %s rows", label, f"{df.count():,}")

    # ── Fix 1: Lọc body rỗng ─────────────────────────────────────────────────
    result_df = result_df.filter(
        result_df.body.isNotNull() & (result_df.body != "")
    )
    _log_stage_count("after body filter", result_df)

    # ── Fix 1.5: Chỉ giữ rows còn usable text cho NLP/topic modeling ────────
    result_df = result_df.filter(
        col("segmented_text").isNotNull()
        & (col("segmented_text") != "")
        & col("topic_text").isNotNull()
        & (col("topic_text") != "")
    )
    _log_stage_count("after usable NLP text filter", result_df)

    # ── Fix 2: Xoá trùng lặp 100% ────────────────────────────────────────────
    result_df = result_df.dropDuplicates()
    _log_stage_count("after exact dedup", result_df)

    # ── Fix 3: Cấp UUID cho post_id = None / "None" ───────────────────────────
    result_df = result_df.withColumn(
        "post_id",
        when(col("post_id").isNull() | (col("post_id") == "None"), expr("uuid()")).otherwise(col("post_id"))
    )

    # cache() sau uuid() vì đây là biểu thức non-deterministic
    # uuid() là non-deterministic — cache tránh re-evaluate
    result_df = result_df.cache()
    _log_stage_count("after post_id normalization", result_df)

    clean_text_rows = result_df.filter(col("clean_text").isNotNull() & (trim(col("clean_text")) != "")).count()
    segmented_rows = result_df.filter(col("segmented_text").isNotNull() & (trim(col("segmented_text")) != "")).count()
    topic_rows = result_df.filter(col("topic_text").isNotNull() & (trim(col("topic_text")) != "")).count()
    logger.info(
        "[CleanAudit] non-empty text columns: clean_text=%s segmented_text=%s topic_text=%s",
        f"{clean_text_rows:,}",
        f"{segmented_rows:,}",
        f"{topic_rows:,}",
    )

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
            num_perm=128,
            threshold=0.85,
            k=5,
        )

        # 5. Run LSH
        result_df = deduplicator.fit_transform(result_df, spark)
        _log_stage_count("after MinHashLSH dedup", result_df)

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

    logger.info("[INGEST] Direct ClickHouse ingest is disabled in cleaning_job; DAG handles it.")

    spark.stop()


if __name__ == "__main__":
    main()
