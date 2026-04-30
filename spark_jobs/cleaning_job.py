"""
spark_jobs/cleaning_job.py — M2: Spark Cleaning & Schema Normalisation Job

Luồng:
    HDFS /user/zett/raw_data/
        ├── voz/comments.csv + posts.csv      → VozAdapter
        ├── vatvo/articles.csv                → VatVoAdapter
        └── vnexpress/post_vnexpress.csv
                    /comment_vnexpress.csv    → VnExpressAdapter
                                ↓
              TextPreprocessor.clean() (lazy init trong mapPartitions)
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

Biến môi trường (tuỳ chỉnh qua --conf spark.executorEnv.*):
    HDFS_BASE       : hdfs://192.168.56.11:9000
    NLP_SLANG_DICT  : path slang_dict.json trên HDFS
    NLP_STOPWORDS   : path stopwords_vi.txt trên HDFS
    HDFS_OUTPUT     : đường dẫn HDFS ghi kết quả Parquet
"""

import os
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, LongType,
)

# ── Config (override bằng env var hoặc spark-submit --conf) ──────────────────
HDFS_BASE  = os.environ.get("HDFS_BASE",    "hdfs://192.168.56.11:9000")
HDFS_RAW   = os.environ.get("HDFS_INPUT",   f"{HDFS_BASE}/user/zett/raw_data")
HDFS_OUT   = os.environ.get("HDFS_OUTPUT",  f"{HDFS_BASE}/user/zett/staged/stg_posts_core")

SLANG_PATH = os.environ.get("NLP_SLANG_DICT", f"{HDFS_BASE}/user/zett/ref/slang_dict.json")
STOP_PATH  = os.environ.get("NLP_STOPWORDS",  f"{HDFS_BASE}/user/zett/ref/stopwords_vi.txt")

# ── Output Schema (khớp với ClickHouse stg_posts_core) ───────────────────────
OUTPUT_SCHEMA = StructType([
    StructField("post_id",        StringType(),  False),
    StructField("source",         StringType(),  False),
    StructField("post_type",      StringType(),  True),
    StructField("author",         StringType(),  True),
    StructField("author_id",      StringType(),  True),
    StructField("title",          StringType(),  True),
    StructField("body",           StringType(),  True),   # clean() output
    StructField("segmented_text", StringType(),  True),   # preprocess() output
    StructField("parent_id",      StringType(),  True),
    StructField("reaction_count", IntegerType(), True),
    StructField("view_count",     IntegerType(), True),
    StructField("comment_count",  IntegerType(), True),
    StructField("created_at",     LongType(),    True),   # Unix timestamp (giây)
    StructField("crawled_at",     LongType(),    False),  # Unix timestamp khi job chạy
    StructField("tags",           StringType(),  True),
    StructField("url",            StringType(),  True),
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
        body    = preprocessor.clean(content)
        seg     = preprocessor.preprocess(content)
        yield (
            str(row.get("comment_id", "")),
            str(row.get("source", "voz")),
            str(row.get("post_type", "comment")),
            str(row.get("user", "") or ""),
            str(row.get("id_user", "") or ""),
            None,                                        # title
            body,
            seg,
            str(row.get("id_post", "") or ""),           # parent_id
            int(row.get("reaction_count", 0) or 0),
            None,                                        # view_count
            None,                                        # comment_count
            int(row.get("created_at", 0) or 0),
            crawled_ts,
            None,                                        # tags
            str(row.get("url", "") or ""),
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
        body    = preprocessor.clean(content)
        seg     = preprocessor.preprocess(content)
        yield (
            str(row.get("id_post", "")),
            str(row.get("source", "voz")),
            str(row.get("post_type", "post")),
            str(row.get("author_name", "") or ""),
            str(row.get("id_author", "") or ""),
            content,                                     # title
            body,
            seg,
            None,                                        # parent_id (post gốc)
            None,                                        # reaction_count
            _safe_int(row.get("view_count")),
            _safe_int(row.get("comment_count")),
            int(row.get("created_at", 0) or 0),
            crawled_ts,
            str(row.get("tags", "") or ""),
            None,                                        # url
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
        body    = preprocessor.clean(content)
        seg     = preprocessor.preprocess(content)
        yield (
            str(row.get("post_id", "")),
            str(row.get("source", "vatvo")),
            str(row.get("post_type", "article")),
            str(row.get("author", "") or ""),
            None,                                        # author_id
            str(row.get("title", "") or ""),
            body,
            seg,
            None,                                        # parent_id
            None,                                        # reaction_count
            None,                                        # view_count
            None,                                        # comment_count
            int(row.get("created_at", 0) or 0),
            crawled_ts,
            None,                                        # tags
            str(row.get("url", "") or ""),
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
        body    = preprocessor.clean(content)
        seg     = preprocessor.preprocess(content)
        yield (
            str(row.get("post_id", "")),
            str(row.get("source", "vnexpress")),
            "article",
            str(row.get("author", "VnExpress")),
            None,                                        # author_id
            str(row.get("title", "") or ""),
            body,
            seg,
            None,                                        # parent_id
            _safe_int(row.get("reaction_count")),
            _safe_int(row.get("view_count")),
            _safe_int(row.get("comment_count")),
            int(row.get("created_at", 0) or 0),
            crawled_ts,
            None,                                        # tags
            str(row.get("url", "") or ""),
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
        body    = preprocessor.clean(content)
        seg     = preprocessor.preprocess(content)
        yield (
            str(row.get("post_id", "")),
            str(row.get("source", "vnexpress")),
            "comment",
            str(row.get("author", "") or ""),
            None,                                        # author_id
            None,                                        # title
            body,
            seg,
            str(row.get("parent_id", "") or ""),
            _safe_int(row.get("reaction_count")),
            None,                                        # view_count
            None,                                        # comment_count
            int(row.get("created_at", 0) or 0),
            crawled_ts,
            None,                                        # tags
            str(row.get("url", "") or ""),
        )


# ── Helper functions ──────────────────────────────────────────────────────────

def _safe_int(val) -> int | None:
    """Chuyển giá trị sang int, trả None nếu không hợp lệ."""
    if val is None:
        return None
    try:
        return int(float(str(val)))
    except (ValueError, TypeError):
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

def main():
    import sys

    # Fix encoding tiếng Việt trên terminal Windows
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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
    read_csv = lambda path: (
        spark.read
             .option("header", "true")
             .option("multiLine", "true")
             .option("quote", '"')
             .option("escape", '"')
             .option("encoding", "UTF-8")
             .csv(path)
    )

    voz_comments_raw = read_csv(f"{HDFS_RAW}/voz/comments.csv")
    voz_posts_raw    = read_csv(f"{HDFS_RAW}/voz/posts.csv")
    vatvo_raw        = read_csv(f"{HDFS_RAW}/vatvo/articles.csv")
    vne_posts_raw    = read_csv(f"{HDFS_RAW}/vnexpress/post_vnexpress.csv")
    vne_cmts_raw     = read_csv(f"{HDFS_RAW}/vnexpress/comment_vnexpress.csv")

    print(f"[INFO] VOZ comments : {voz_comments_raw.count():,}")
    print(f"[INFO] VOZ posts    : {voz_posts_raw.count():,}")
    print(f"[INFO] VatVo        : {vatvo_raw.count():,}")
    print(f"[INFO] VnE posts    : {vne_posts_raw.count():,}")
    print(f"[INFO] VnE comments : {vne_cmts_raw.count():,}")

    # ── mapPartitions → process qua Adapter + TextPreprocessor ───────────────
    sc = spark.sparkContext

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

    # Lọc body rỗng (do adapter skip hoặc content quá ngắn)
    result_df = result_df.filter(
        result_df.body.isNotNull() & (result_df.body != "")
    )

    total = result_df.count()
    print(f"\n[INFO] Tổng bản ghi sau clean: {total:,}")

    # Thống kê theo nguồn
    result_df.groupBy("source", "post_type").count().orderBy("source").show()

    # ── Ghi ra HDFS Parquet ───────────────────────────────────────────────────
    result_df.write \
        .mode("overwrite") \
        .partitionBy("source") \
        .parquet(HDFS_OUT)

    print(f"[DONE] Đã ghi {total:,} bản ghi → {HDFS_OUT}")
    spark.stop()


if __name__ == "__main__":
    main()
