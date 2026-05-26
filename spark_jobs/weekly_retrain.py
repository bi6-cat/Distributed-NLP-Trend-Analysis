"""Weekly retrain job without Spark JDBC.

Flow:
  1. Read stg_posts_core from HDFS Parquet.
  2. Read small ClickHouse tables through HTTP SQL.
  3. Write Spark outputs to HDFS Parquet and ingest with ClickHouse hdfs().
  4. Upload model artifacts through WebHDFS.
"""

from __future__ import annotations

import os
import pickle
from datetime import date
from urllib.parse import urlparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    coalesce,
    col,
    count,
    countDistinct,
    current_date,
    date_sub,
    hour as spark_hour,
    lit,
    percentile_approx,
    stddev,
    sum as spark_sum,
    to_date,
    when,
)
from pyspark.sql.types import FloatType, StringType, StructField, StructType

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from spark_jobs.clickhouse_hdfs import query_df


LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "60"))
MODEL_LOCAL_DIR = os.environ.get("MODEL_LOCAL_DIR", "/tmp/airflow_models")
HDFS_MODEL_DIR = os.environ.get(
    "HDFS_MODEL_DIR",
    "hdfs://namenode:9000/user/zett/models/crisis_detection",
)
HDFS_STG_POSTS_CORE = os.environ.get(
    "HDFS_STG_POSTS_CORE",
    "hdfs://namenode:9000/user/zett/staged/stg_posts_core",
)
HDFS_STG_BASE = os.environ.get(
    "HDFS_STG_BASE",
    "hdfs://namenode:9000/user/zett/staged",
)
WEBHDFS_HOST = os.environ.get("WEBHDFS_HOST", "namenode")
WEBHDFS_PORT = int(os.environ.get("WEBHDFS_PORT", "9870"))
HDFS_USER = os.environ.get("HDFS_USER", "zett")
IF_CONTAMINATION = float(os.environ.get("IF_CONTAMINATION", "0.05"))
MIN_CRISIS_SAMPLES = int(os.environ.get("MIN_CRISIS_SAMPLES", "50"))

os.makedirs(MODEL_LOCAL_DIR, exist_ok=True)


def _hdfs_uri_path(uri: str) -> str:
    parsed = urlparse(uri)
    return parsed.path if parsed.scheme == "hdfs" else uri


def _webhdfs_mkdirs(path: str) -> None:
    import requests

    url = (
        f"http://{WEBHDFS_HOST}:{WEBHDFS_PORT}/webhdfs/v1{path}"
        f"?op=MKDIRS&user.name={HDFS_USER}"
    )
    response = requests.put(url, timeout=30)
    response.raise_for_status()


def _webhdfs_put(local_path: str, hdfs_path: str) -> None:
    import requests

    create_url = (
        f"http://{WEBHDFS_HOST}:{WEBHDFS_PORT}/webhdfs/v1{hdfs_path}"
        f"?op=CREATE&overwrite=true&user.name={HDFS_USER}"
    )
    response = requests.put(create_url, allow_redirects=False, timeout=30)
    response.raise_for_status()
    upload_url = response.headers["Location"]

    with open(local_path, "rb") as fh:
        upload = requests.put(upload_url, data=fh, timeout=120)
    upload.raise_for_status()


def load_core(spark: SparkSession):
    return (
        spark.read.parquet(HDFS_STG_POSTS_CORE)
        .select("post_id", "created_at", "parent_id", "source", "author")
        .where(col("parent_id").isNotNull())
        .where(to_date(col("created_at")) >= date_sub(current_date(), LOOKBACK_DAYS))
    )


def load_nlp(spark: SparkSession):
    pdf = query_df("SELECT post_id, sentiment_label, sentiment_score FROM stg_posts_nlp")
    schema = StructType(
        [
            StructField("post_id", StringType(), False),
            StructField("sentiment_label", StringType(), True),
            StructField("sentiment_score", FloatType(), True),
        ]
    )
    if pdf.empty:
        return spark.createDataFrame([], schema)
    return spark.createDataFrame(pdf, schema=schema)


def step_compute_baseline(spark: SparkSession, core) -> None:
    print(f"\n[retrain] Step 1: compute_baseline ({LOOKBACK_DAYS} days)")
    n_total = core.count()
    print(f"[baseline] {n_total:,} comment records")
    if n_total == 0:
        print("[baseline] WARN: no data, skip baseline.")
        return

    daily_hourly = (
        core.withColumn("date", to_date(col("created_at")))
        .withColumn("hour", spark_hour(col("created_at")))
        .groupBy("date", "hour")
        .agg(count("*").alias("daily_count"))
    )

    baseline = (
        daily_hourly.groupBy("hour")
        .agg(
            percentile_approx("daily_count", 0.5).alias("baseline_median"),
            stddev("daily_count").alias("baseline_std"),
        )
        .withColumn("baseline_median", col("baseline_median").cast("float"))
        .withColumn("baseline_std", coalesce(col("baseline_std").cast("float"), lit(0.0)))
        .fillna(0.0)
        .select("hour", "baseline_median", "baseline_std")
    )

    n_hours = baseline.count()
    baseline.write.mode("overwrite").parquet(f"{HDFS_STG_BASE}/hourly_baseline")
    print(f"[baseline] Wrote {n_hours} rows to HDFS hourly_baseline")


FEATURES_IF = [
    "comment_count", "velocity_ratio", "acceleration",
    "user_diversity", "cross_source", "z_score", "unique_posts_ratio",
]


def step_retrain_isolation_forest(spark: SparkSession, core, nlp) -> str:
    print("\n[retrain] Step 2: retrain IsolationForest")
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    core_h = (
        core
        .withColumn("date", to_date(col("created_at")))
        .withColumn("hour", spark_hour(col("created_at")))
    )
    hourly_struct = (
        core_h.groupBy("date", "hour")
        .agg(
            count("*").alias("comment_count"),
            countDistinct("parent_id").alias("unique_posts"),
            countDistinct("author").alias("unique_users"),
            spark_sum(when(col("source") == "voz",       1).otherwise(0)).alias("voz_count"),
            spark_sum(when(col("source") == "vnexpress", 1).otherwise(0)).alias("vne_count"),
        )
    )

    hourly = hourly_struct.orderBy("date", "hour")
    pdf = hourly.toPandas()
    n_rows = len(pdf)
    print(f"[IF] Training rows: {n_rows}")
    if n_rows < 24:
        print(f"[IF] WARN: too little data ({n_rows} rows), skip IF retrain.")
        return ""

    pdf["user_diversity"]     = (pdf["unique_users"] / (pdf["comment_count"] + 1)).round(4)
    pdf["cross_source"]       = (
        pdf[["vne_count", "voz_count"]].min(axis=1) /
        (pdf[["vne_count", "voz_count"]].max(axis=1) + 1)
    ).round(4)
    pdf["unique_posts_ratio"] = (pdf["unique_posts"] / (pdf["comment_count"] + 1)).round(4)

    roll3 = pdf["comment_count"].astype(float).rolling(window=3, min_periods=1).mean().shift(1)
    vel   = (pdf["comment_count"] / (roll3 + 1)).round(4)
    pdf["velocity_ratio"] = vel
    pdf["acceleration"]   = vel.diff().fillna(0).round(4)
    pdf["z_score"]        = 0.0

    x = pdf[FEATURES_IF].fillna(0).values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = IsolationForest(
        n_estimators=200,
        contamination=IF_CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_scaled)

    artifact = {"scaler": scaler, "model": model, "features": FEATURES_IF, "trained_at": str(date.today())}
    pkl_path = os.path.join(MODEL_LOCAL_DIR, "isolation_forest_hourly.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(artifact, fh)

    print(f"[IF] Saved model: {pkl_path}")
    return pkl_path


def step_retrain_crisis_classifier(spark: SparkSession) -> str:
    print("\n[retrain] Step 3: retrain CrisisClassifier")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    pdf = (
        spark.read.parquet(f"{HDFS_STG_BASE}/stg_crisis_hourly")
        .filter(col("date") >= date_sub(current_date(), LOOKBACK_DAYS))
        .select(
            "date", "hour", "comment_count", "z_score",
            "global_spike", "if_spike", "neg_ratio", "neg_score_avg", "is_crisis",
        )
        .toPandas()
    )
    n_total = len(pdf)
    if n_total == 0:
        print("[clf] WARN: no crisis hourly rows, skip classifier retrain.")
        return ""

    n_crisis = int((pdf["is_crisis"] == 1).sum())
    print(f"[clf] Dataset: {n_total} rows, {n_crisis} crisis")
    if n_crisis < MIN_CRISIS_SAMPLES:
        print(f"[clf] WARN: crisis samples ({n_crisis}) < MIN ({MIN_CRISIS_SAMPLES}), skip.")
        return ""

    features = ["comment_count", "z_score", "global_spike", "if_spike", "neg_ratio", "neg_score_avg"]
    x = pdf[features].fillna(0).values
    y = pdf["is_crisis"].astype(int).values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(x_scaled, y)

    artifact = {
        "scaler": scaler,
        "model": clf,
        "features": features,
        "trained_at": str(date.today()),
    }
    pkl_path = os.path.join(MODEL_LOCAL_DIR, "crisis_classifier.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(artifact, fh)

    print(f"[clf] Saved model: {pkl_path}")
    return pkl_path


def step_upload_to_hdfs(models: list[str]) -> None:
    print("\n[retrain] Step 4: upload models via WebHDFS")
    hdfs_dir = _hdfs_uri_path(HDFS_MODEL_DIR)
    _webhdfs_mkdirs(hdfs_dir)

    for local_path in models:
        if not local_path or not os.path.exists(local_path):
            print(f"[hdfs] SKIP missing: {local_path}")
            continue
        hdfs_dest = f"{hdfs_dir.rstrip('/')}/{os.path.basename(local_path)}"
        _webhdfs_put(local_path, hdfs_dest)
        print(f"[hdfs] Uploaded {local_path} -> {hdfs_dest}")


def main() -> None:
    spark = SparkSession.builder.appName("weekly_retrain").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print(f"[retrain] weekly_retrain started: {date.today()}")
    print(f"[retrain] LOOKBACK_DAYS       : {LOOKBACK_DAYS}")
    print(f"[retrain] HDFS_STG_POSTS_CORE: {HDFS_STG_POSTS_CORE}")
    print(f"[retrain] MODEL_DIR          : {MODEL_LOCAL_DIR}")
    print(f"[retrain] HDFS_MODEL_DIR     : {HDFS_MODEL_DIR}")

    trained_models: list[str] = []
    core = load_core(spark).cache()
    nlp = load_nlp(spark).cache()

    try:
        step_compute_baseline(spark, core)

        if_path = step_retrain_isolation_forest(spark, core, nlp)
        if if_path:
            trained_models.append(if_path)

        clf_path = step_retrain_crisis_classifier(spark)
        if clf_path:
            trained_models.append(clf_path)
    finally:
        spark.stop()

    if trained_models:
        step_upload_to_hdfs(trained_models)
    else:
        print("[retrain] No new models to upload.")

    print(f"[retrain] Done. Uploaded models: {len(trained_models)}")


if __name__ == "__main__":
    main()
