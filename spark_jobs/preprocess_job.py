"""
Spark job: chạy text preprocessing phân tán trên cluster (Task 1.2 — M4).

Đọc stg_posts_core từ HDFS Parquet (M2 output) → TextPreprocessor phân tán →
ghi Parquet với clean_text + segmented_text để dùng cho benchmark / verify
trước khi sentiment_job.py chạy inference.

KHÔNG chứa PhoBERT inference — đó là nhiệm vụ của sentiment_job.py (Task 3.1).

Cách chạy trên cluster:
    spark-submit \\
        --master spark://192.168.56.11:7077 \\
        --executor-memory 4g \\
        --total-executor-cores 8 \\
        --py-files dist/nlp_trend.zip \\
        --conf spark.executorEnv.NLP_SLANG_DICT=hdfs:///data/ref/slang_dict.json \\
        --conf spark.executorEnv.NLP_STOPWORDS=hdfs:///data/ref/stopwords_vi.txt \\
        spark_jobs/preprocess_job.py

Biến môi trường:
    HDFS_INPUT     : path Parquet stg_posts_core  (mặc định hdfs:///data/silver/posts_core/date=*/)
    HDFS_OUTPUT    : path Parquet đầu ra          (mặc định hdfs:///data/silver/posts_preprocessed/)
    NLP_SLANG_DICT : path slang_dict.json
    NLP_STOPWORDS  : path stopwords_vi.txt

Dev/debug local: dùng spark_jobs/preprocess_local.py thay thế.
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, TimestampType,
)

HDFS_INPUT      = os.environ.get("HDFS_INPUT",     "hdfs:///data/silver/posts_core/date=*/")
HDFS_OUTPUT     = os.environ.get("HDFS_OUTPUT",    "hdfs:///data/silver/posts_preprocessed/")
SLANG_DICT_PATH = os.environ.get("NLP_SLANG_DICT", "hdfs:///data/ref/slang_dict.json")
STOPWORDS_PATH  = os.environ.get("NLP_STOPWORDS",  "hdfs:///data/ref/stopwords_vi.txt")

# Schema đầu ra: stg_posts_core + clean_text + segmented_text
# sentiment_job.py đọc clean_text từ đây để chạy PhoBERT inference
OUTPUT_SCHEMA = StructType([
    StructField("post_id",        StringType(),    False),
    StructField("source",         StringType(),    True),
    StructField("author",         StringType(),    True),
    StructField("title",          StringType(),    True),
    StructField("body",           StringType(),    True),
    StructField("clean_text",     StringType(),    True),
    StructField("segmented_text", StringType(),    True),
    StructField("parent_id",      StringType(),    True),
    StructField("reaction_count", IntegerType(),   True),
    StructField("comment_count",  IntegerType(),   True),
    StructField("view_count",     IntegerType(),   True),
    StructField("created_at",     TimestampType(), True),
    StructField("crawled_at",     TimestampType(), True),
])


def process_partition(iterator):
    """
    Chạy TextPreprocessor trên mỗi Spark executor (lazy init).

    Không load PhoBERT — chỉ clean text + word segmentation.
    Yield tuple theo OUTPUT_SCHEMA.
    """
    import os, sys, traceback

    try:
        from pyspark import SparkFiles
        root = SparkFiles.getRootDirectory()
        if root not in sys.path:
            sys.path.insert(0, root)
    except Exception:
        pass

    slang_dict_path = os.environ.get("NLP_SLANG_DICT", "data/slang_dict.json")
    stopwords_path  = os.environ.get("NLP_STOPWORDS",  "data/stopwords_vi.txt")

    def _hdfs_to_local(hdfs_path, local_name):
        if not hdfs_path.startswith("hdfs://"):
            return hdfs_path
        local = f"/tmp/{local_name}"
        if not os.path.exists(local):
            import subprocess
            subprocess.run(["hdfs", "dfs", "-get", hdfs_path, local], check=True)
        return local

    slang_dict_path = _hdfs_to_local(slang_dict_path, "slang_dict.json")
    stopwords_path  = _hdfs_to_local(stopwords_path,  "stopwords_vi.txt")

    try:
        from preprocessing.text_cleaner import TextPreprocessor
    except Exception as e:
        print(f"[WORKER IMPORT ERROR] {e}")
        traceback.print_exc()
        return

    preprocessor = TextPreprocessor(
        slang_dict_path=slang_dict_path,
        stopwords_path=stopwords_path,
        use_vncorenlp=False,  # đổi True khi Java 11 đã sẵn trên tất cả nodes
    )

    for row in iterator:
        body = getattr(row, "body", None) or ""
        try:
            clean_text = preprocessor.preprocess(body)
            segmented_text = (
                preprocessor.segment(body)
                if hasattr(preprocessor, "segment")
                else clean_text
            )
        except Exception as e:
            print(f"[PREPROCESS ERROR] post_id={getattr(row, 'post_id', '?')} {e}")
            clean_text     = ""
            segmented_text = ""

        yield (
            str(getattr(row, "post_id",       "") or ""),
            getattr(row, "source",             None),
            getattr(row, "author",             None),
            getattr(row, "title",              None),
            body,
            clean_text,
            segmented_text,
            getattr(row, "parent_id",          None),
            getattr(row, "reaction_count",     None),
            getattr(row, "comment_count",      None),
            getattr(row, "view_count",         None),
            getattr(row, "created_at",         None),
            getattr(row, "crawled_at",         None),
        )


def main():
    spark = SparkSession.builder \
        .appName("NLP_PreprocessJob") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print(f"[INFO] Đọc stg_posts_core từ : {HDFS_INPUT}")
    raw_df = spark.read.parquet(HDFS_INPUT)

    for col_name in ("post_id", "body"):
        if col_name not in raw_df.columns:
            raw_df = raw_df.withColumn(col_name, lit(None))

    total = raw_df.count()
    print(f"[INFO] Tổng bản ghi đầu vào : {total:,}")

    processed_rdd = raw_df.rdd.mapPartitions(process_partition)
    processed_df  = spark.createDataFrame(processed_rdd, OUTPUT_SCHEMA)

    processed_df.write \
        .mode("overwrite") \
        .parquet(HDFS_OUTPUT)

    written = processed_df.count()
    print(f"[INFO] Đã ghi {written:,} bản ghi → {HDFS_OUTPUT}")

    processed_df.select("post_id", "body", "clean_text", "segmented_text") \
                .show(5, truncate=60)

    spark.stop()


if __name__ == "__main__":
    main()
