"""
crisis_detection.py  —  Daily Spark job (04:00 AM)

Pipeline:
  1. Đọc stg_core hôm nay (comments)   từ ClickHouse qua JDBC
  2. Đọc stg_nlp hôm nay (sentiments)  từ ClickHouse qua JDBC
  3. Đọc hourly_baseline               từ ClickHouse qua JDBC
  4. Aggregate → hourly features (count, neg_ratio, neg_score_avg)
  5. Tính z_score so với baseline → global_spike flag
  6. IsolationForest inference          trên driver (collect 24 rows)
  7. Kết hợp: is_spike  = global_spike AND if_spike
             is_crisis = CrisisClassifier output (nếu model tồn tại)
  8. Ghi stg_crisis_events              vào ClickHouse

Cách chạy:
    spark-submit \\
        --master spark://192.168.56.11:7077 \\
        --jars /opt/spark/jars/clickhouse-jdbc-0.6.0-all.jar \\
        --conf spark.executorEnv.CLICKHOUSE_HOST=192.168.56.14 \\
        --conf spark.executorEnv.IF_MODEL_PATH=/opt/models/isolation_forest_hourly.pkl \\
        spark_jobs/crisis_detection.py

Biến môi trường:
    CLICKHOUSE_HOST   : IP ClickHouse node       (mặc định 192.168.56.14)
    CLICKHOUSE_PORT   : JDBC port                (mặc định 8123)
    CLICKHOUSE_DB     : database                 (mặc định tech_radar)
    CLICKHOUSE_USER   : user                     (mặc định default)
    CLICKHOUSE_PASS   : password                 (mặc định '')
    JDBC_JAR          : đường dẫn tới JDBC jar      (mặc định /opt/spark/jars/clickhouse-jdbc-0.6.0-all.jar)
    IF_MODEL_PATH     : path IsolationForest pkl    (mặc định /opt/models/isolation_forest_hourly.pkl)
    CLF_MODEL_PATH    : path CrisisClassifier pkl   (mặc định /opt/models/crisis_classifier.pkl)
    Z_SCORE_THRESHOLD : ngưỡng z_score spike        (mặc định 2.0)
    TARGET_DATE       : ngày cần xử lý YYYY-MM-DD   (mặc định hôm nay)
"""

import os
import pickle
from datetime import date

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg, col, count, hour as spark_hour,
    lit, to_date, when, coalesce,
)
from pyspark.sql.types import (
    DateType, FloatType, IntegerType,
    StructField, StructType,
)

# ── Cấu hình ──────────────────────────────────────────────────────────────────

CLICKHOUSE_HOST   = os.environ.get("CLICKHOUSE_HOST",   "192.168.56.14")
CLICKHOUSE_PORT   = os.environ.get("CLICKHOUSE_PORT",   "8123")
CLICKHOUSE_DB     = os.environ.get("CLICKHOUSE_DB",     "tech_radar")
CLICKHOUSE_USER   = os.environ.get("CLICKHOUSE_USER",   "default")
CLICKHOUSE_PASS   = os.environ.get("CLICKHOUSE_PASS",   "")
JDBC_JAR          = os.environ.get("JDBC_JAR",          "/opt/spark/jars/clickhouse-jdbc-0.6.0-all.jar")
IF_MODEL_PATH     = os.environ.get("IF_MODEL_PATH",     "/opt/models/isolation_forest_hourly.pkl")
CLF_MODEL_PATH    = os.environ.get("CLF_MODEL_PATH",    "/opt/models/crisis_classifier.pkl")
Z_SCORE_THRESHOLD = float(os.environ.get("Z_SCORE_THRESHOLD", "2.0"))
TARGET_DATE       = os.environ.get("TARGET_DATE",       str(date.today()))

CLICKHOUSE_URL  = f"jdbc:clickhouse://{CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}/{CLICKHOUSE_DB}"
CLICKHOUSE_OPTS = {
    "driver":   "com.clickhouse.jdbc.ClickHouseDriver",
    "user":     CLICKHOUSE_USER,
    "password": CLICKHOUSE_PASS,
}

# Schema output — khớp CREATE TABLE stg_crisis_events
OUTPUT_SCHEMA = StructType([
    StructField("date",          DateType(),    False),
    StructField("hour",          IntegerType(), False),
    StructField("comment_count", IntegerType(), False),
    StructField("z_score",       FloatType(),   True),
    StructField("global_spike",  IntegerType(), False),
    StructField("if_spike",      IntegerType(), False),
    StructField("is_spike",      IntegerType(), False),
    StructField("is_crisis",     IntegerType(), False),
    StructField("neg_ratio",     FloatType(),   True),
    StructField("neg_score_avg", FloatType(),   True),
])


# ── Driver-side inference helpers ─────────────────────────────────────────────

def _load_pkl(path: str):
    """Load artifact pkl (dict {"scaler":..., "model":...}). Trả None nếu không tồn tại."""
    if not os.path.exists(path):
        print(f"[model] WARN: không tìm thấy {path} — bỏ qua inference.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _if_predict(pdf, artifact) -> np.ndarray:
    """
    IsolationForest inference trên pandas DataFrame (24 rows).
    Trả mảng 0/1 — 1 = anomaly (if_spike).
    """
    if artifact is None:
        return np.zeros(len(pdf), dtype=int)

    FEATURES = ["comment_count", "neg_ratio", "neg_score_avg"]
    X = pdf[FEATURES].fillna(0).values.astype(float)
    X_scaled = artifact["scaler"].transform(X)
    preds = artifact["model"].predict(X_scaled)   # -1 = anomaly, 1 = normal
    return (preds == -1).astype(int)


def _clf_predict(pdf, artifact) -> np.ndarray:
    """
    CrisisClassifier (LogisticRegression) inference trên pandas DataFrame.
    Trả mảng 0/1 — 1 = crisis.
    """
    if artifact is None:
        return np.zeros(len(pdf), dtype=int)

    FEATURES = artifact["features"]
    X = pdf[FEATURES].fillna(0).values.astype(float)
    X_scaled = artifact["scaler"].transform(X)
    return artifact["model"].predict(X_scaled).astype(int)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    spark = (
        SparkSession.builder
        .appName("crisis_detection_daily")
        .config("spark.jars", JDBC_JAR)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"[crisis] TARGET_DATE      : {TARGET_DATE}")
    print(f"[crisis] Z_SCORE_THRESHOLD: {Z_SCORE_THRESHOLD}")
    print(f"[crisis] IF_MODEL_PATH    : {IF_MODEL_PATH}")
    print(f"[crisis] CLF_MODEL_PATH   : {CLF_MODEL_PATH}")

    # ── 1. Đọc stg_core hôm nay (comments) ───────────────────────────────────
    print("[crisis] Bước 1/7: Đọc stg_core...")
    core = spark.read.jdbc(
        url=CLICKHOUSE_URL,
        table=f"""(
            SELECT post_id, created_at, parent_id
            FROM stg_core
            WHERE toDate(created_at) = '{TARGET_DATE}'
              AND parent_id IS NOT NULL
        ) t""",
        properties=CLICKHOUSE_OPTS,
    )
    n_core = core.count()
    print(f"[crisis] stg_core comments hôm nay: {n_core:,}")
    if n_core == 0:
        print("[crisis] Không có comment nào hôm nay. Dừng job.")
        spark.stop()
        return

    # ── 2. Đọc stg_nlp hôm nay (sentiments) ─────────────────────────────────
    print("[crisis] Bước 2/7: Đọc stg_nlp...")
    nlp = spark.read.jdbc(
        url=CLICKHOUSE_URL,
        table=f"""(
            SELECT n.post_id, n.sentiment_label, n.sentiment_score
            FROM stg_posts_nlp n
            JOIN stg_posts_core c ON c.post_id = n.post_id
            WHERE toDate(c.created_at) = '{TARGET_DATE}'
              AND c.parent_id IS NOT NULL
        ) t""",
        properties=CLICKHOUSE_OPTS,
    )
    print(f"[crisis] stg_nlp records: {nlp.count():,}")

    # ── 3. Đọc hourly_baseline ────────────────────────────────────────────────
    print("[crisis] Bước 3/7: Đọc hourly_baseline...")
    baseline = spark.read.jdbc(
        url=CLICKHOUSE_URL,
        table="hourly_baseline",
        properties=CLICKHOUSE_OPTS,
    )
    n_baseline = baseline.count()
    print(f"[crisis] hourly_baseline: {n_baseline} giờ")
    if n_baseline == 0:
        print("[crisis] WARN: hourly_baseline rỗng — z_score sẽ là NULL, global_spike = 0")

    # ── 4. Aggregate hourly features ──────────────────────────────────────────
    print("[crisis] Bước 4/7: Aggregate hourly features...")

    # 4a. Comment count theo giờ từ stg_core
    hourly_counts = (
        core
        .withColumn("date", to_date(col("created_at")))
        .withColumn("hour", spark_hour(col("created_at")))
        .groupBy("date", "hour")
        .agg(count("*").alias("comment_count"))
    )

    # 4b. Sentiment features từ stg_nlp (join vào stg_core để lấy created_at)
    core_with_time = (
        core
        .withColumn("date", to_date(col("created_at")))
        .withColumn("hour", spark_hour(col("created_at")))
        .select("post_id", "date", "hour")
    )
    nlp_with_time = nlp.join(core_with_time, on="post_id", how="left")

    nlp_lower = nlp_with_time.withColumn(
        "is_neg",
        when(col("sentiment_label") == "negative", 1).otherwise(0),
    )
    hourly_nlp = (
        nlp_lower
        .groupBy("date", "hour")
        .agg(
            avg("is_neg").alias("neg_ratio"),
            avg(
                when(col("sentiment_label") == "negative", col("sentiment_score"))
            ).alias("neg_score_avg"),
        )
    )

    # 4c. Join counts + sentiment features
    hourly = (
        hourly_counts
        .join(hourly_nlp, on=["date", "hour"], how="left")
        .withColumn("neg_ratio",     coalesce(col("neg_ratio"),     lit(0.0)).cast("float"))
        .withColumn("neg_score_avg", coalesce(col("neg_score_avg"), lit(0.0)).cast("float"))
    )

    # ── 5. Tính z_score và global_spike ──────────────────────────────────────
    print("[crisis] Bước 5/7: Tính z_score và global_spike...")

    hourly_with_baseline = hourly.join(
        baseline.select(
            col("hour"),
            col("baseline_median"),
            col("baseline_std"),
        ),
        on="hour",
        how="left",
    )

    hourly_with_zs = hourly_with_baseline.withColumn(
        "z_score",
        when(
            (col("baseline_std").isNotNull()) & (col("baseline_std") > 0),
            ((col("comment_count") - col("baseline_median")) / col("baseline_std")).cast("float"),
        ).otherwise(lit(None).cast("float")),
    ).withColumn(
        "global_spike",
        when(col("z_score") > Z_SCORE_THRESHOLD, 1).otherwise(0),
    )

    # ── 6. Collect về driver → IF inference + Classifier inference ───────────
    print("[crisis] Bước 6/7: Collect về driver và chạy inference...")

    # 24 rows/ngày — an toàn để collect về driver
    pdf = hourly_with_zs.select(
        "date", "hour", "comment_count", "z_score", "global_spike",
        "neg_ratio", "neg_score_avg",
    ).toPandas()

    # Load models trên driver
    if_artifact  = _load_pkl(IF_MODEL_PATH)
    clf_artifact = _load_pkl(CLF_MODEL_PATH)

    # IsolationForest → if_spike
    pdf["if_spike"] = _if_predict(pdf, if_artifact)

    # is_spike = global_spike AND if_spike
    pdf["is_spike"] = ((pdf["global_spike"] == 1) & (pdf["if_spike"] == 1)).astype(int)

    # CrisisClassifier features (phải khớp với features lúc train)
    CLF_FEATURES = ["comment_count", "z_score", "global_spike", "if_spike",
                    "neg_ratio", "neg_score_avg"]
    pdf["z_score"] = pdf["z_score"].fillna(0.0)

    # is_crisis = output CrisisClassifier (nếu không có model → fallback is_spike)
    if clf_artifact is not None:
        pdf["is_crisis"] = _clf_predict(pdf, clf_artifact)
    else:
        print("[crisis] WARN: không có CLF model — is_crisis = is_spike")
        pdf["is_crisis"] = pdf["is_spike"]

    # ── 7. Convert lại thành Spark DataFrame và ghi ───────────────────────────
    print("[crisis] Bước 7/7: Ghi stg_crisis_events...")

    # Cast đúng kiểu trước khi tạo DataFrame
    pdf["date"]          = pdf["date"].astype(str)   # Spark sẽ parse lại qua schema
    pdf["hour"]          = pdf["hour"].astype(int)
    pdf["comment_count"] = pdf["comment_count"].astype(int)
    pdf["z_score"]       = pdf["z_score"].astype(float)
    pdf["global_spike"]  = pdf["global_spike"].astype(int)
    pdf["if_spike"]      = pdf["if_spike"].astype(int)
    pdf["is_spike"]      = pdf["is_spike"].astype(int)
    pdf["is_crisis"]     = pdf["is_crisis"].astype(int)
    pdf["neg_ratio"]     = pdf["neg_ratio"].astype(float)
    pdf["neg_score_avg"] = pdf["neg_score_avg"].astype(float)

    final = spark.createDataFrame(
        pdf[["date", "hour", "comment_count", "z_score",
             "global_spike", "if_spike", "is_spike", "is_crisis",
             "neg_ratio", "neg_score_avg"]],
        schema=OUTPUT_SCHEMA,
    )

    # ── Ghi vào ClickHouse ────────────────────────────────────────────────────
    n_events = final.count()
    n_crisis  = final.filter(col("is_crisis") == 1).count()
    n_spike   = final.filter(col("is_spike")  == 1).count()
    print(f"[crisis] Kết quả: {n_events} giờ | {n_spike} spike | {n_crisis} crisis")

    final.write.jdbc(
        url=CLICKHOUSE_URL,
        table="stg_crisis_events",
        mode="append",
        properties=CLICKHOUSE_OPTS,
    )
    print(f"[crisis] Ghi {n_events} dòng vào stg_crisis_events — xong.")
    final.orderBy("hour").show(24, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
