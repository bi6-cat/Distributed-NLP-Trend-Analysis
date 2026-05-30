"""Spark job that normalizes raw crawler CSVs into stg_posts_core Parquet."""

import argparse
import logging
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, trim, when
from pyspark.sql.types import (
    StructField,
    LongType,
    StringType,
    StructType,
    TimestampType,
)

# Config is intentionally env-driven because Airflow injects these values.
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
TOKENIZER_BACKEND = os.environ.get("NLP_TOKENIZER", "underthesea").strip().lower()
VNCORENLP_JAR = os.environ.get("NLP_VNCORENLP_JAR", "vncorenlp/VnCoreNLP-1.1.1.jar")
WEBHDFS_HOST = os.environ.get("WEBHDFS_HOST", "namenode:9870")

OUTPUT_SCHEMA = StructType([
    StructField("post_id",        StringType(),  False),
    StructField("source",         StringType(),  False),
    StructField("author",         StringType(),  True),   # ingested as author_name
    StructField("title",          StringType(),  True),
    StructField("body",           StringType(),  True),
    StructField("clean_text",     StringType(),  True),
    StructField("segmented_text", StringType(),  True),
    StructField("topic_text",     StringType(),  True),
    StructField("parent_id",      StringType(),  True),
    StructField("reaction_count", LongType(),    True),
    StructField("view_count",     LongType(),    True),
    StructField("comment_count",  LongType(),    True),
    StructField("created_at",     TimestampType(), True),
    StructField("crawled_at",     TimestampType(), False),
])


RAW_INPUTS = (
    ("voz_comments", "voz/comments.csv"),
    ("voz_posts", "voz/posts.csv"),
    ("vatvo_articles", "vatvo/articles.csv"),
    ("vnexpress_posts", "vnexpress/post_vnexpress.csv"),
    ("vnexpress_comments", "vnexpress/comment_vnexpress.csv"),
)


def _text_outputs(preprocessor, content: str) -> tuple[str, str, str, str]:
    return (
        preprocessor.clean_html(content),
        preprocessor.clean(content),
        preprocessor.preprocess(content),
        preprocessor.preprocess_for_topic(content),
    )


def _out(
    row,
    preprocessor,
    *,
    content: str,
    post_id,
    source: str,
    author=None,
    title=None,
    parent_id=None,
    reaction_count=None,
    view_count=None,
    comment_count=None,
    created_at=None,
    crawled_ts: int,
):
    body, clean_text, segmented_text, topic_text = _text_outputs(preprocessor, content)
    return (
        str(post_id or ""),
        str(row.get("source", source) or source),
        str(author or ""),
        title,
        body,
        clean_text,
        segmented_text,
        topic_text,
        None if parent_id is None else str(parent_id or ""),
        _safe_int(reaction_count),
        _safe_int(view_count),
        _safe_int(comment_count),
        _to_datetime(created_at),
        _to_datetime(crawled_ts),
    )


def _build_voz_comment(row, preprocessor, crawled_ts: int):
    return _out(
        row,
        preprocessor,
        content=str(row.get("comment", "") or ""),
        post_id=row.get("comment_id"),
        source="voz",
        author=row.get("user"),
        parent_id=row.get("id_post"),
        reaction_count=row.get("reaction_count", 0),
        created_at=row.get("created_at"),
        crawled_ts=crawled_ts,
    )


def _build_voz_post(row, preprocessor, crawled_ts: int):
    title = str(row.get("title", "") or "")
    return _out(
        row,
        preprocessor,
        content=title,
        post_id=row.get("id_post"),
        source="voz",
        author=row.get("author_name"),
        title=title,
        view_count=row.get("view_count"),
        comment_count=row.get("comment_count"),
        created_at=row.get("created_at"),
        crawled_ts=crawled_ts,
    )


def _build_vatvo_article(row, preprocessor, crawled_ts: int):
    return _out(
        row,
        preprocessor,
        content=str(row.get("content", "") or ""),
        post_id=row.get("post_id"),
        source="vatvo",
        author=row.get("author"),
        title=str(row.get("title", "") or ""),
        created_at=row.get("created_at"),
        crawled_ts=crawled_ts,
    )


def _build_vnexpress_post(row, preprocessor, crawled_ts: int):
    return _out(
        row,
        preprocessor,
        content=str(row.get("body", "") or ""),
        post_id=row.get("post_id"),
        source="vnexpress",
        author=row.get("author") or "VnExpress",
        title=str(row.get("title", "") or ""),
        reaction_count=row.get("reaction_count"),
        view_count=row.get("view_count"),
        comment_count=row.get("comment_count"),
        created_at=row.get("created_at"),
        crawled_ts=crawled_ts,
    )


def _build_vnexpress_comment(row, preprocessor, crawled_ts: int):
    return _out(
        row,
        preprocessor,
        content=str(row.get("body", "") or ""),
        post_id=row.get("post_id"),
        source="vnexpress",
        author=row.get("author"),
        parent_id=row.get("parent_id"),
        reaction_count=row.get("reaction_count"),
        created_at=row.get("created_at"),
        crawled_ts=crawled_ts,
    )


def _source_adapter(source_key: str):
    from schemas.vatvo_adapter import VatVoAdapter
    from schemas.vnexpress_adapter import VnExpressAdapter
    from schemas.voz_adapter import VozAdapter

    adapters = {
        "voz_comments": (VozAdapter(), "comments_to_df", _build_voz_comment),
        "voz_posts": (VozAdapter(), "posts_to_df", _build_voz_post),
        "vatvo_articles": (VatVoAdapter(), "articles_to_df", _build_vatvo_article),
        "vnexpress_posts": (VnExpressAdapter(), "posts_to_df", _build_vnexpress_post),
        "vnexpress_comments": (VnExpressAdapter(), "comments_to_df", _build_vnexpress_comment),
    }
    return adapters[source_key]


def process_partition(iterator, source_key: str, slang_path: str, stop_path: str, crawled_ts: int):
    rows = [r.asDict() for r in iterator]
    if not rows:
        return iter(())

    _ensure_worker_path()

    from preprocessing.text_cleaner import TextPreprocessor

    adapter, method_name, build_row = _source_adapter(source_key)
    preprocessor = TextPreprocessor(
        slang_dict_path=_resolve_hdfs(slang_path),
        stopwords_path=_resolve_hdfs(stop_path),
        use_vncorenlp=TOKENIZER_BACKEND == "vncorenlp",
        vncorenlp_jar=_resolve_hdfs(VNCORENLP_JAR),
    )
    df = getattr(adapter, method_name)(rows)
    return (build_row(row, preprocessor, crawled_ts) for _, row in df.iterrows())


def _ensure_worker_path() -> None:
    try:
        from pyspark import SparkFiles

        root = SparkFiles.getRootDirectory()
        if root not in sys.path:
            sys.path.insert(0, root)
    except Exception:
        pass


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
    import shutil
    import subprocess

    if not path.startswith("hdfs://"):
        return path

    filename = path.split("/")[-1]
    local    = f"/tmp/{filename}"

    if os.path.exists(local):
        return local

    hdfs_bin = shutil.which("hdfs")
    if hdfs_bin:
        try:
            subprocess.run([hdfs_bin, "dfs", "-get", path, local],
                           check=True, capture_output=True)
            return local
        except Exception as e:
            logger.warning("[HDFS] CLI copy failed for %s -> %s: %s", path, local, e)

    try:
        parsed = urllib.parse.urlparse(path)
        open_url = (
            f"http://{WEBHDFS_HOST}/webhdfs/v1{parsed.path}"
            f"?op=OPEN&user.name={urllib.parse.quote(HDFS_USER)}"
        )
        with urllib.request.urlopen(open_url, timeout=60) as response, open(local, "wb") as fh:
            fh.write(response.read())
        return local
    except Exception as e:
        logger.warning("[HDFS] WebHDFS copy failed for %s -> %s: %s", path, local, e)
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
    args, _ = parser.parse_known_args()
    return args


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    cli_args = _parse_args()
    enable_dedup = not cli_args.no_dedup

    spark = SparkSession.builder.appName("M2_CleaningJob").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    crawled_ts = int(datetime.now(timezone.utc).timestamp())
    parallelism = spark.sparkContext.defaultParallelism
    read_partitions = max(parallelism * 2, 8)

    logger.info("[Config] raw=%s output=%s", HDFS_RAW, HDFS_OUT)
    logger.info("[Config] slang=%s stopwords=%s", SLANG_PATH, STOP_PATH)
    logger.info("[Config] tokenizer=%s vncorenlp_jar=%s", TOKENIZER_BACKEND, VNCORENLP_JAR)
    logger.info("[Config] parallelism=%s read_partitions=%s dedup=%s", parallelism, read_partitions, enable_dedup)

    def read_csv(relative_path: str):
        return (
            spark.read
            .option("header", "true")
            .option("multiLine", "true")
            .option("quote", '"')
            .option("escape", '"')
            .option("encoding", "UTF-8")
            .csv(f"{HDFS_RAW}/{relative_path}")
            .repartition(read_partitions)
        )

    rdds = (
        read_csv(path).rdd.mapPartitions(
            lambda it, key=source_key: process_partition(it, key, SLANG_PATH, STOP_PATH, crawled_ts)
        )
        for source_key, path in RAW_INPUTS
    )
    all_rdd = spark.sparkContext.union(list(rdds))
    result_df = spark.createDataFrame(all_rdd, OUTPUT_SCHEMA)

    def _log_stage_count(label: str, df) -> None:
        logger.info("[CleanAudit] %s: %s rows", label, f"{df.count():,}")

    result_df = result_df.filter(
        col("body").isNotNull()
        & (trim(col("body")) != "")
        & col("segmented_text").isNotNull()
        & (trim(col("segmented_text")) != "")
        & col("topic_text").isNotNull()
        & (trim(col("topic_text")) != "")
    )
    _log_stage_count("after text filters", result_df)

    result_df = result_df.dropDuplicates()
    _log_stage_count("after exact dedup", result_df)

    result_df = result_df.withColumn(
        "post_id",
        when(
            col("post_id").isNull() | (trim(col("post_id")) == "") | (col("post_id") == "None"),
            expr("uuid()"),
        ).otherwise(col("post_id")),
    )
    result_df = result_df.persist(StorageLevel.MEMORY_AND_DISK)
    _log_stage_count("after post_id normalization", result_df)

    if enable_dedup:
        logger.info("[Dedup] Starting Spark MinHashLSH dedup...")

        from algorithms.minhash_dedup import MinHashDeduplicator

        target_partitions = max(parallelism * 3, 8)
        result_df = result_df.repartition(target_partitions)
        result_df = result_df.persist(StorageLevel.MEMORY_AND_DISK)

        spark.sparkContext.setCheckpointDir(f"{HDFS_BASE}/tmp/spark-checkpoints")
        result_df = result_df.checkpoint()
        result_df.count()

        deduplicator = MinHashDeduplicator(
            num_perm=128,
            threshold=0.85,
            k=5,
        )
        result_df = deduplicator.fit_transform(result_df, spark)
        _log_stage_count("after MinHashLSH dedup", result_df)
        logger.info("[Dedup] Completed MinHashLSH dedup")

    result_df.write.mode("overwrite").partitionBy("source").parquet(HDFS_OUT)

    logger.info(f"[DONE] Đã ghi Parquet → {HDFS_OUT}")
    logger.info("[INGEST] Direct ClickHouse ingest is disabled in cleaning_job; DAG handles it.")

    spark.stop()


if __name__ == "__main__":
    main()
