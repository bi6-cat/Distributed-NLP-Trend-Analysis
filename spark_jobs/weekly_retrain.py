"""
weekly_retrain.py  —  Weekly Spark job (Chủ nhật 03:00 AM)

Thứ tự thực hiện:
  1. compute_baseline   — tính lại hourly_baseline từ 60 ngày lịch sử
  2. retrain_isolation_forest — retrain IF trên dữ liệu mới
  3. retrain_crisis_classifier — retrain classifier trên stg_crisis_events labeled
  4. upload_models      — push pkl lên HDFS

Cách chạy:
    spark-submit \\
        --master spark://192.168.56.11:7077 \\
        --jars /opt/spark/jars/clickhouse-jdbc-0.6.0-all.jar \\
        --conf spark.executorEnv.CLICKHOUSE_HOST=192.168.56.14 \\
        spark_jobs/weekly_retrain.py

Biến môi trường:
    CLICKHOUSE_HOST    : IP ClickHouse node        (mặc định 192.168.56.14)
    CLICKHOUSE_PORT    : JDBC port                 (mặc định 8123)
    CLICKHOUSE_DB      : database                  (mặc định tech_radar)
    CLICKHOUSE_USER    : user                      (mặc định default)
    CLICKHOUSE_PASS    : password                  (mặc định '')
    JDBC_JAR           : đường dẫn tới JDBC jar    (mặc định /opt/spark/jars/clickhouse-jdbc-0.6.0-all.jar)
    LOOKBACK_DAYS      : số ngày lịch sử IF        (mặc định 60)
    MODEL_LOCAL_DIR    : thư mục lưu pkl local     (mặc định /opt/models)
    HDFS_MODEL_DIR     : đường dẫn HDFS upload     (mặc định hdfs://192.168.56.11:9000/user/zett/models/crisis_detection)
    IF_CONTAMINATION   : tỉ lệ outlier IF          (mặc định 0.05)
    MIN_CRISIS_SAMPLES : số mẫu tối thiểu để retrain classifier (mặc định 50)
"""

import os
import pickle
import subprocess
from datetime import date

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg, col, count, hour as spark_hour,
    lit, percentile_approx, stddev,
    to_date, coalesce, when,
)

# ── Cấu hình ──────────────────────────────────────────────────────────────────

CLICKHOUSE_HOST     = os.environ.get("CLICKHOUSE_HOST",     "192.168.56.14")
CLICKHOUSE_PORT     = os.environ.get("CLICKHOUSE_PORT",     "8123")
CLICKHOUSE_DB       = os.environ.get("CLICKHOUSE_DB",       "tech_radar")
CLICKHOUSE_USER     = os.environ.get("CLICKHOUSE_USER",     "default")
CLICKHOUSE_PASS     = os.environ.get("CLICKHOUSE_PASS",     "")
JDBC_JAR            = os.environ.get("JDBC_JAR",            "/opt/spark/jars/clickhouse-jdbc-0.6.0-all.jar")
LOOKBACK_DAYS       = int(os.environ.get("LOOKBACK_DAYS",   "60"))
MODEL_LOCAL_DIR     = os.environ.get("MODEL_LOCAL_DIR",     "/opt/models")
HDFS_MODEL_DIR      = os.environ.get("HDFS_MODEL_DIR",      "hdfs://192.168.56.11:9000/user/zett/models/crisis_detection")
IF_CONTAMINATION    = float(os.environ.get("IF_CONTAMINATION",    "0.05"))
MIN_CRISIS_SAMPLES  = int(os.environ.get("MIN_CRISIS_SAMPLES",    "50"))

CLICKHOUSE_URL  = f"jdbc:clickhouse://{CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}/{CLICKHOUSE_DB}"
CLICKHOUSE_OPTS = {
    "driver":   "com.clickhouse.jdbc.ClickHouseDriver",
    "user":     CLICKHOUSE_USER,
    "password": CLICKHOUSE_PASS,
}

os.makedirs(MODEL_LOCAL_DIR, exist_ok=True)


# ── Step 1: compute_baseline ──────────────────────────────────────────────────

def step_compute_baseline(spark: SparkSession) -> None:
    """
    Tính lại hourly_baseline từ stg_core 60 ngày.
    Logic đúng: group by (date, hour) trước → median/std trên daily_count.
    """
    print(f"\n[retrain] ── Step 1: compute_baseline ({LOOKBACK_DAYS} ngày) ──")

    core = spark.read.jdbc(
        url=CLICKHOUSE_URL,
        table=f"""(
            SELECT post_id, created_at, parent_id
            FROM stg_core
            WHERE toDate(created_at) >= today() - {LOOKBACK_DAYS}
              AND parent_id IS NOT NULL
        ) t""",
        properties=CLICKHOUSE_OPTS,
    )
    n_total = core.count()
    print(f"[baseline] {n_total:,} comment records")

    if n_total == 0:
        print("[baseline] WARN: không có dữ liệu — bỏ qua bước này.")
        return

    # Bước 1: daily_count per (date, hour)
    daily_hourly = (
        core
        .withColumn("date", to_date(col("created_at")))
        .withColumn("hour", spark_hour(col("created_at")))
        .groupBy("date", "hour")
        .agg(count("*").alias("daily_count"))
    )

    # Bước 2: median + std theo hour
    baseline = (
        daily_hourly
        .groupBy("hour")
        .agg(
            percentile_approx("daily_count", 0.5).alias("baseline_median"),
            stddev("daily_count").alias("baseline_std"),
        )
        .withColumn("baseline_median", col("baseline_median").cast("float"))
        .withColumn("baseline_std",
                    coalesce(col("baseline_std").cast("float"), lit(0.0)))
        .fillna(0.0)
    )

    n_hours = baseline.count()
    baseline.write.jdbc(
        url=CLICKHOUSE_URL,
        table="hourly_baseline",
        mode="append",       # ReplacingMergeTree dedup theo hour
        properties=CLICKHOUSE_OPTS,
    )
    print(f"[baseline] Cập nhật {n_hours} giờ vào hourly_baseline ✓")


# ── Step 2: retrain IsolationForest ──────────────────────────────────────────

def step_retrain_isolation_forest(spark: SparkSession) -> str:
    """
    Đọc stg_core 60 ngày → tính hourly features → fit IsolationForest.
    Lưu pkl: {MODEL_LOCAL_DIR}/isolation_forest_hourly.pkl

    Returns: đường dẫn pkl đã lưu
    """
    print(f"\n[retrain] ── Step 2: retrain IsolationForest ──")
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Đọc stg_core + stg_nlp để lấy hourly features
    core = spark.read.jdbc(
        url=CLICKHOUSE_URL,
        table=f"""(
            SELECT post_id, created_at, parent_id
            FROM stg_core
            WHERE toDate(created_at) >= today() - {LOOKBACK_DAYS}
              AND parent_id IS NOT NULL
        ) t""",
        properties=CLICKHOUSE_OPTS,
    )

    nlp = spark.read.jdbc(
        url=CLICKHOUSE_URL,
        table=f"""(
            SELECT n.post_id, n.sentiment_label, n.sentiment_score
            FROM stg_posts_nlp n
            JOIN stg_posts_core c ON c.post_id = n.post_id
            WHERE toDate(c.created_at) >= today() - {LOOKBACK_DAYS}
              AND c.parent_id IS NOT NULL
        ) t""",
        properties=CLICKHOUSE_OPTS,
    )

    # Hourly comment count
    core_h = (
        core
        .withColumn("date", to_date(col("created_at")))
        .withColumn("hour", spark_hour(col("created_at")))
    )
    hourly_counts = (
        core_h
        .groupBy("date", "hour")
        .agg(count("*").alias("comment_count"))
    )

    # Hourly sentiment features
    core_ids = core_h.select("post_id", "date", "hour")
    nlp_h = (
        nlp
        .join(core_ids, on="post_id", how="left")
        .withColumn("is_neg",
                    when(col("sentiment_label") == "negative", 1).otherwise(0))
        .groupBy("date", "hour")
        .agg(
            avg("is_neg").alias("neg_ratio"),
            avg(when(col("sentiment_label") == "negative",
                     col("sentiment_score"))).alias("neg_score_avg"),
        )
    )

    hourly = (
        hourly_counts
        .join(nlp_h, on=["date", "hour"], how="left")
        .withColumn("neg_ratio",     coalesce(col("neg_ratio"),     lit(0.0)))
        .withColumn("neg_score_avg", coalesce(col("neg_score_avg"), lit(0.0)))
    )

    # Collect về driver để fit sklearn (dữ liệu hourly nhỏ — tối đa 60*24=1440 dòng)
    pdf = hourly.toPandas()
    n_rows = len(pdf)
    print(f"[IF] Tập train: {n_rows} (date, hour) records")

    if n_rows < 24:
        print(f"[IF] WARN: quá ít dữ liệu ({n_rows} dòng) — bỏ qua retrain IF.")
        return ""

    X = pdf[["comment_count", "neg_ratio", "neg_score_avg"]].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=IF_CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # Lưu cùng scaler để inference dùng đúng transform
    artifact = {"scaler": scaler, "model": model, "trained_at": str(date.today())}
    pkl_path = os.path.join(MODEL_LOCAL_DIR, "isolation_forest_hourly.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"[IF] Lưu model → {pkl_path} ✓")
    return pkl_path


# ── Step 3: retrain Crisis Classifier ────────────────────────────────────────

def step_retrain_crisis_classifier(spark: SparkSession) -> str:
    """
    Đọc stg_crisis_events có nhãn is_crisis → retrain LogisticRegression classifier.
    Lưu pkl: {MODEL_LOCAL_DIR}/crisis_classifier.pkl

    Yêu cầu tối thiểu MIN_CRISIS_SAMPLES dòng is_crisis=1 để tránh mô hình lệch.
    Returns: đường dẫn pkl đã lưu, hoặc '' nếu bỏ qua.
    """
    print(f"\n[retrain] ── Step 3: retrain CrisisClassifier (LogisticRegression) ──")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    events = spark.read.jdbc(
        url=CLICKHOUSE_URL,
        table=f"""(
            SELECT date, hour, comment_count, z_score,
                   global_spike, if_spike, neg_ratio, neg_score_avg, is_crisis
            FROM stg_crisis_events
            WHERE toDate(date) >= today() - {LOOKBACK_DAYS}
        ) t""",
        properties=CLICKHOUSE_OPTS,
    )

    pdf = events.toPandas()
    n_total  = len(pdf)
    n_crisis = (pdf["is_crisis"] == 1).sum()
    print(f"[clf] Dataset: {n_total} records, {n_crisis} crisis ({n_crisis/max(n_total,1)*100:.1f}%)")

    if n_crisis < MIN_CRISIS_SAMPLES:
        print(f"[clf] WARN: crisis samples ({n_crisis}) < MIN ({MIN_CRISIS_SAMPLES}) — bỏ qua retrain classifier.")
        return ""

    FEATURES = ["comment_count", "z_score", "global_spike", "if_spike", "neg_ratio", "neg_score_avg"]
    X = pdf[FEATURES].fillna(0).values
    y = pdf["is_crisis"].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        class_weight="balanced",   # xử lý imbalanced (crisis rất ít)
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X_scaled, y)

    artifact = {
        "scaler":   scaler,
        "model":    clf,
        "features": FEATURES,
        "trained_at": str(date.today()),
    }
    pkl_path = os.path.join(MODEL_LOCAL_DIR, "crisis_classifier.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"[clf] Lưu model → {pkl_path} ✓")
    return pkl_path


# ── Step 4: upload lên HDFS ───────────────────────────────────────────────────

def step_upload_to_hdfs(models: list[str]) -> None:
    """
    Push danh sách pkl lên HDFS_MODEL_DIR qua hdfs dfs -put.
    Dùng -f để overwrite nếu đã tồn tại.
    """
    print(f"\n[retrain] ── Step 4: upload models lên HDFS ──")
    hdfs_bin = os.environ.get("HDFS_BIN", "/opt/hadoop/bin/hdfs")

    # Tạo thư mục trên HDFS nếu chưa có
    subprocess.run(
        [hdfs_bin, "dfs", "-mkdir", "-p", HDFS_MODEL_DIR],
        check=False,
    )

    for local_path in models:
        if not local_path or not os.path.exists(local_path):
            print(f"[hdfs] SKIP (không tồn tại): {local_path}")
            continue

        filename = os.path.basename(local_path)
        hdfs_dest = f"{HDFS_MODEL_DIR}/{filename}"
        size_mb = os.path.getsize(local_path) / (1024 * 1024)

        print(f"[hdfs] {local_path} → {hdfs_dest}  ({size_mb:.1f} MB)")
        result = subprocess.run(
            [hdfs_bin, "dfs", "-put", "-f", local_path, hdfs_dest],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[hdfs] WARN: upload thất bại: {result.stderr.strip()}")
        else:
            print(f"[hdfs] OK: {hdfs_dest} ✓")

    # Verify: liệt kê thư mục HDFS
    result = subprocess.run(
        [hdfs_bin, "dfs", "-ls", HDFS_MODEL_DIR],
        capture_output=True, text=True,
    )
    print(f"[hdfs] Nội dung {HDFS_MODEL_DIR}:\n{result.stdout}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    spark = (
        SparkSession.builder
        .appName("weekly_retrain")
        .config("spark.jars", JDBC_JAR)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"[retrain] ══ weekly_retrain bắt đầu: {date.today()} ══")
    print(f"[retrain] LOOKBACK_DAYS : {LOOKBACK_DAYS}")
    print(f"[retrain] MODEL_DIR     : {MODEL_LOCAL_DIR}")
    print(f"[retrain] HDFS_DIR      : {HDFS_MODEL_DIR}")

    trained_models = []

    try:
        # Step 1 — baseline (không trả model)
        step_compute_baseline(spark)

        # Step 2 — IsolationForest
        if_path = step_retrain_isolation_forest(spark)
        if if_path:
            trained_models.append(if_path)

        # Step 3 — CrisisClassifier
        clf_path = step_retrain_crisis_classifier(spark)
        if clf_path:
            trained_models.append(clf_path)

    finally:
        spark.stop()   # stop trước khi subprocess HDFS để tránh giữ port

    # Step 4 — upload (chạy ngoài SparkSession)
    if trained_models:
        step_upload_to_hdfs(trained_models)
    else:
        print("[retrain] Không có model mới nào để upload.")

    print(f"\n[retrain] ══ Hoàn tất. Models đã upload: {len(trained_models)} ══")


if __name__ == "__main__":
    main()
