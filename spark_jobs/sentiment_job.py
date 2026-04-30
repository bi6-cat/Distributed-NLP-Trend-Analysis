"""
Task 3.1 — Tích hợp PhoBERT vào Spark qua mapPartitions.

Đọc raw JSON từ HDFS (do M1 crawl) → preprocessing → PhoBERT inference →
ghi kết quả vào ClickHouse table stg_sentiment_scores.

Cách chạy trên cluster:
    spark-submit \\
        --master spark://192.168.56.11:7077 \\
        --executor-memory 2g \\
        --total-executor-cores 4 \\
        --py-files dist/nlp_trend.zip \\
        --conf spark.jars=/opt/spark/jars/clickhouse-jdbc.jar \\
        --conf spark.executorEnv.NLP_MODEL_PATH=hdfs:///models/phobert_finetuned/final \\
        --conf spark.executorEnv.NLP_SLANG_DICT=hdfs:///data/ref/slang_dict.json \\
        --conf spark.executorEnv.NLP_STOPWORDS=hdfs:///data/ref/stopwords_vi.txt \\
        --conf spark.executorEnv.CLICKHOUSE_HOST=192.168.56.14 \\
        spark_jobs/sentiment_job.py

Biến môi trường (truyền qua --conf spark.executorEnv.* hoặc export trước):
    HDFS_INPUT       : glob path đọc raw JSON  (mặc định hdfs:///data/raw/**/date=*/*.json)
    NLP_MODEL_PATH   : path tới PhoBERT checkpoint (HDFS hoặc local shared fs)
    NLP_SLANG_DICT   : path slang_dict.json
    NLP_STOPWORDS    : path stopwords_vi.txt
    CLICKHOUSE_HOST  : IP storage node          (mặc định 192.168.56.14)
    CLICKHOUSE_PORT  : HTTP port ClickHouse     (mặc định 8123)
    CLICKHOUSE_DB    : database name            (mặc định default)
    CLICKHOUSE_USER  : user                     (mặc định default)
    CLICKHOUSE_PASS  : password                 (mặc định '')
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType,
)

# ── Cấu hình đọc từ env (driver đọc, truyền xuống executor qua broadcast) ──
HDFS_INPUT      = os.environ.get("HDFS_INPUT",      "hdfs:///data/raw/**/date=*/*.json")
MODEL_PATH      = os.environ.get("NLP_MODEL_PATH",  "hdfs:///models/phobert_finetuned/final")
SLANG_DICT_PATH = os.environ.get("NLP_SLANG_DICT",  "hdfs:///data/ref/slang_dict.json")
STOPWORDS_PATH  = os.environ.get("NLP_STOPWORDS",   "hdfs:///data/ref/stopwords_vi.txt")

CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST", "192.168.56.14")
CLICKHOUSE_PORT = os.environ.get("CLICKHOUSE_PORT", "8123")
CLICKHOUSE_DB   = os.environ.get("CLICKHOUSE_DB",   "default")
CLICKHOUSE_USER = os.environ.get("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASS = os.environ.get("CLICKHOUSE_PASS", "")

# Schema đầu ra — khớp với ClickHouse table stg_sentiment_scores
OUTPUT_SCHEMA = StructType([
    StructField("post_id",         StringType(),  True),
    StructField("source",          StringType(),  True),
    StructField("content",         StringType(),  True),
    StructField("clean_text",      StringType(),  True),
    StructField("sentiment_label", StringType(),  True),
    StructField("sentiment_id",    IntegerType(), True),
    StructField("confidence",      FloatType(),   True),
    StructField("post_type",       StringType(),  True),
    StructField("author",          StringType(),  True),
    StructField("author_id",       StringType(),  True),
    StructField("created_at",      IntegerType(), True),
    StructField("reaction_count",  IntegerType(), True),
    StructField("view_count",      IntegerType(), True),
    StructField("comment_count",   IntegerType(), True),
    StructField("parent_post_id",  StringType(),  True),
    StructField("title",           StringType(),  True),
    StructField("tags",            StringType(),  True),
])


def process_partition(iterator):
    """
    Chạy trên mỗi Spark executor (worker).

    Lazy init: PhoBERT được load 1 lần duy nhất mỗi partition,
    không serialize model qua mạng.

    Đầu vào : iterator các Spark Row từ raw JSON
    Đầu ra  : iterator các tuple theo thứ tự OUTPUT_SCHEMA
    """
    import os, sys, traceback

    # Đảm bảo import được preprocessing/ trên worker
    try:
        from pyspark import SparkFiles
        root = SparkFiles.getRootDirectory()
        if root not in sys.path:
            sys.path.insert(0, root)
    except Exception:
        pass

    # Resolve paths — ưu tiên env var (set bởi spark-submit)
    model_path      = os.environ.get("NLP_MODEL_PATH",  "models/phobert_finetuned/final")
    slang_dict_path = os.environ.get("NLP_SLANG_DICT",  "data/slang_dict.json")
    stopwords_path  = os.environ.get("NLP_STOPWORDS",   "data/stopwords_vi.txt")

    # Nếu path là HDFS (hdfs://...) thì đọc về /tmp trước
    def _hdfs_to_local(hdfs_path, local_name):
        """Copy file từ HDFS về /tmp nếu cần."""
        if not hdfs_path.startswith("hdfs://"):
            return hdfs_path
        local = f"/tmp/{local_name}"
        if not os.path.exists(local):
            import subprocess
            subprocess.run(["hdfs", "dfs", "-get", hdfs_path, local], check=True)
        return local

    slang_dict_path = _hdfs_to_local(slang_dict_path, "slang_dict.json")
    stopwords_path  = _hdfs_to_local(stopwords_path,  "stopwords_vi.txt")

    # Import sau khi sys.path đã cập nhật
    try:
        from preprocessing.text_cleaner import TextPreprocessor
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception as e:
        print(f"[WORKER IMPORT ERROR] {e}")
        traceback.print_exc()
        return

    # Init preprocessing + model (1 lần cho cả partition)
    preprocessor = TextPreprocessor(
        slang_dict_path=slang_dict_path,
        stopwords_path=stopwords_path,
        use_vncorenlp=False,   # cluster dùng underthesea; đổi True nếu Java 11 đã sẵn
    )

    LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    BATCH_SIZE = 32
    rows       = list(iterator)

    for start in range(0, len(rows), BATCH_SIZE):
        batch       = rows[start : start + BATCH_SIZE]
        clean_texts = []

        for row in batch:
            try:
                text  = row.content if isinstance(row.content, str) else (str(row.content) if row.content else "")
                clean = preprocessor.preprocess(text)
            except Exception as e:
                print(f"[PREPROCESS ERROR] {e}")
                clean = ""
            clean_texts.append(clean)

        # Tokenize + inference
        encoding = tokenizer(
            clean_texts,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            ).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)

        # Yield từng dòng theo OUTPUT_SCHEMA
        for i, row in enumerate(batch):
            yield (
                str(getattr(row, "post_id",        "") or ""),
                str(getattr(row, "source",         "") or ""),
                str(getattr(row, "content",        "") or ""),
                clean_texts[i],
                LABEL_MAP[int(preds[i])],
                int(preds[i]),
                float(probs[i][int(preds[i])]),
                str(getattr(row, "post_type",      "") or ""),
                str(getattr(row, "author",         "") or ""),
                str(getattr(row, "author_id",      "") or ""),
                int(getattr(row, "created_at",     0)  or 0),
                int(getattr(row, "reaction_count", 0)  or 0),
                int(getattr(row, "view_count",     0)  or 0),
                int(getattr(row, "comment_count",  0)  or 0),
                str(getattr(row, "parent_post_id", "") or ""),
                str(getattr(row, "title",          "") or ""),
                str(getattr(row, "tags",           "") or ""),
            )


def write_to_clickhouse(df, host, port, db, user, password, table="stg_sentiment_scores"):
    """
    Ghi Spark DataFrame vào ClickHouse qua JDBC.

    Yêu cầu: clickhouse-jdbc jar trong classpath (--jars khi spark-submit).
    """
    jdbc_url = f"jdbc:clickhouse://{host}:{port}/{db}"
    df.write \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", table) \
        .option("user", user) \
        .option("password", password) \
        .option("driver", "com.clickhouse.jdbc.ClickHouseDriver") \
        .mode("append") \
        .save()
    print(f"[INFO] Đã ghi {df.count()} dòng vào ClickHouse {table}")


def main():
    spark = SparkSession.builder \
        .appName("NLP_SentimentJob") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print(f"[INFO] Đọc raw data từ: {HDFS_INPUT}")
    raw_df = spark.read.json(HDFS_INPUT)

    # Chuẩn hoá tên cột — khớp với UniversalSocialPost (M1 schema)
    col_renames = {
        "thread_id": "post_id",
        "message":   "content",
        "id":        "post_id",
        "text":      "content",
    }
    for old, new in col_renames.items():
        if old in raw_df.columns and new not in raw_df.columns:
            raw_df = raw_df.withColumnRenamed(old, new)

    # Thêm cột null cho những field metadata không có trong source
    required_cols = [
        "post_id", "source", "content", "post_type", "author", "author_id",
        "created_at", "reaction_count", "view_count", "comment_count",
        "parent_post_id", "title", "tags",
    ]
    for col_name in required_cols:
        if col_name not in raw_df.columns:
            raw_df = raw_df.withColumn(col_name, lit(None))

    total = raw_df.count()
    print(f"[INFO] Tổng số bản ghi đầu vào: {total:,}")

    # Phân tán inference qua mapPartitions
    processed_rdd = raw_df.rdd.mapPartitions(process_partition)
    processed_df  = spark.createDataFrame(processed_rdd, OUTPUT_SCHEMA)

    # Ghi vào ClickHouse
    write_to_clickhouse(
        processed_df,
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        db=CLICKHOUSE_DB,
        user=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASS,
    )

    # Log phân bố sentiment
    print("\n── Phân bố sentiment ──")
    processed_df.groupBy("sentiment_label").count().show()

    spark.stop()


if __name__ == "__main__":
    main()
