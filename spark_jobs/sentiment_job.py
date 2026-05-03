"""
Task 3.1 — Tích hợp PhoBERT vào Spark qua mapPartitions.

Đọc stg_posts_core từ HDFS Parquet (M2 output) → PhoBERT inference →
ghi kết quả vào ClickHouse table stg_posts_nlp.

Input là stg_posts_core vì M2 đã làm preprocessing (HTML strip, slang,
word segment). Worker chỉ cần đọc segmented_text, không preprocess lại.

Cách chạy trên cluster:
    spark-submit \\
        --master spark://192.168.56.11:7077 \\
        --executor-memory 2g \\
        --total-executor-cores 4 \\
        --py-files dist/nlp_trend.zip \\
        --conf spark.jars=/opt/spark/jars/clickhouse-jdbc.jar \\
        --conf spark.executorEnv.NLP_MODEL_PATH=hdfs:///models/phobert_finetuned/final \\
        --conf spark.executorEnv.NLP_MODEL_VERSION=phobert_v1 \\
        --conf spark.executorEnv.CLICKHOUSE_HOST=192.168.56.14 \\
        spark_jobs/sentiment_job.py

Biến môi trường (truyền qua --conf spark.executorEnv.* hoặc export trước):
    HDFS_INPUT        : path Parquet stg_posts_core  (mặc định hdfs:///data/silver/posts_core/date=*/)
    NLP_MODEL_PATH    : path tới PhoBERT checkpoint (HDFS hoặc local shared fs)
    NLP_MODEL_VERSION : version string ghi vào model_version  (mặc định 'phobert_v1')
    CLICKHOUSE_HOST   : IP storage node          (mặc định 192.168.56.14)
    CLICKHOUSE_PORT   : HTTP port ClickHouse     (mặc định 8123)
    CLICKHOUSE_DB     : database name            (mặc định tech_radar)
    CLICKHOUSE_USER   : user                     (mặc định default)
    CLICKHOUSE_PASS   : password                 (mặc định '')
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import (
    StructType, StructField,
    StringType, FloatType, TimestampType,
)

HDFS_INPUT    = os.environ.get("HDFS_INPUT",        "hdfs:///data/silver/posts_core/date=*/")
MODEL_PATH    = os.environ.get("NLP_MODEL_PATH",    "hdfs:///models/phobert_finetuned/final")
MODEL_VERSION = os.environ.get("NLP_MODEL_VERSION", "phobert_v1")

CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST", "192.168.56.14")
CLICKHOUSE_PORT = os.environ.get("CLICKHOUSE_PORT", "8123")
CLICKHOUSE_DB   = os.environ.get("CLICKHOUSE_DB",   "tech_radar")
CLICKHOUSE_USER = os.environ.get("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASS = os.environ.get("CLICKHOUSE_PASS", "")

# Schema đầu ra — khớp với ClickHouse table stg_posts_nlp (schema M5)
# Single Responsibility: chỉ chứa NLP output, metadata post nằm ở stg_posts_core (M2)
OUTPUT_SCHEMA = StructType([
    StructField("post_id",         StringType(),   False),
    StructField("sentiment_label", StringType(),   True),
    StructField("sentiment_score", FloatType(),    True),
    StructField("model_version",   StringType(),   True),
    StructField("predicted_at",    TimestampType(), True),
])


def process_partition(iterator):
    """
    Chạy trên mỗi Spark executor (lazy init).

    Đầu vào : Row từ stg_posts_core — dùng segmented_text (M2 đã preprocess)
    Đầu ra  : tuple theo OUTPUT_SCHEMA (5 cột)

    Không chạy TextPreprocessor — M2 đã làm, tránh duplicate work.
    PhoBERT load 1 lần duy nhất mỗi partition, không serialize qua mạng.
    """
    import os, sys, traceback
    from datetime import datetime, timezone

    try:
        from pyspark import SparkFiles
        root = SparkFiles.getRootDirectory()
        if root not in sys.path:
            sys.path.insert(0, root)
    except Exception:
        pass

    model_path    = os.environ.get("NLP_MODEL_PATH",    "models/phobert_finetuned/final")
    model_version = os.environ.get("NLP_MODEL_VERSION", "phobert_v1")

    def _hdfs_to_local(hdfs_path, local_name):
        if not hdfs_path.startswith("hdfs://"):
            return hdfs_path
        local = f"/tmp/{local_name}"
        if not os.path.exists(local):
            import subprocess
            hdfs_bin = "/opt/hadoop/bin/hdfs"
            subprocess.run([hdfs_bin, "dfs", "-get", hdfs_path, local], check=True)
        return local

    model_path = _hdfs_to_local(model_path, "phobert_finetuned")

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception as e:
        print(f"[WORKER IMPORT ERROR] {e}", flush=True)
        traceback.print_exc()
        raise RuntimeError(f"Worker cannot import torch/transformers: {e}")

    LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model     = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[WORKER MODEL LOAD ERROR] model_path={model_path} error={e}", flush=True)
        traceback.print_exc()
        raise RuntimeError(f"Worker cannot load model from {model_path}: {e}")

    BATCH_SIZE = 32
    rows       = list(iterator)

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start : start + BATCH_SIZE]

        # Dùng segmented_text từ M2 (đã clean + word-segment)
        # Fallback về body nếu segmented_text chưa có
        texts = []
        for row in batch:
            text = getattr(row, "segmented_text", None) or getattr(row, "body", None) or ""
            texts.append(text if isinstance(text, str) else str(text))

        encoding = tokenizer(
            texts,
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

        predicted_at = datetime.now(timezone.utc).replace(tzinfo=None)

        for i, row in enumerate(batch):
            yield (
                str(getattr(row, "post_id", "") or ""),
                LABEL_MAP[int(preds[i])],
                float(probs[i][int(preds[i])]),
                model_version,
                predicted_at,
            )


def write_to_clickhouse(df, host, port, db, user, password, table="stg_posts_nlp"):
    """Ghi DataFrame vào ClickHouse qua HTTP API (TSV format)."""
    import urllib.request
    import urllib.parse

    rows = df.collect()
    if not rows:
        print("[INFO] Không có dòng nào để ghi.")
        return

    lines = []
    for r in rows:
        predicted_at = r.predicted_at.strftime("%Y-%m-%d %H:%M:%S") if r.predicted_at else "1970-01-01 00:00:00"
        line = "\t".join([
            str(r.post_id),
            str(r.sentiment_label),
            str(r.sentiment_score),
            str(r.model_version),
            predicted_at,
        ])
        lines.append(line)

    body = "\n".join(lines).encode("utf-8")
    query = f"INSERT INTO {db}.{table} (post_id, sentiment_label, sentiment_score, model_version, predicted_at) FORMAT TabSeparated"
    url = f"http://{host}:{port}/?query={urllib.parse.quote(query)}&user={user}&password={password}"

    req = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(req) as resp:
        resp.read()

    print(f"[INFO] Đã ghi {len(rows)} dòng vào ClickHouse {table}")


def main():
    spark = SparkSession.builder \
        .appName("NLP_SentimentJob") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Đọc stg_posts_core (M2 output) — đã có post_id, body, segmented_text
    print(f"[INFO] Đọc stg_posts_core từ: {HDFS_INPUT}")
    raw_df = spark.read.parquet(HDFS_INPUT)

    for col_name in ("post_id", "segmented_text", "body"):
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
