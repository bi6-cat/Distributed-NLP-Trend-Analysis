from __future__ import annotations

import hashlib
import json
import os
import pickle
from datetime import date
import tempfile

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    coalesce,
    col,
    count,
    countDistinct,
    hour as spark_hour,
    lit,
    sum as spark_sum,
    to_date,
    when,
)
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

HDFS_STG_POSTS_CORE = os.environ.get(
    "HDFS_STG_POSTS_CORE",
    "hdfs://namenode:9000/user/root/staged/stg_posts_core",
)
IF_MODEL_PATH   = os.environ.get("IF_MODEL_PATH",  "/tmp/airflow_models/isolation_forest_hourly.pkl")
CLF_MODEL_PATH  = os.environ.get("CLF_MODEL_PATH", "/tmp/airflow_models/crisis_classifier.pkl")
_models_dir     = os.path.dirname(IF_MODEL_PATH)
FEAT1_JSON_PATH = os.environ.get("FEAT1_JSON_PATH", os.path.join(_models_dir, "features_tier1.json"))
FEAT2_JSON_PATH = os.environ.get("FEAT2_JSON_PATH", os.path.join(_models_dir, "features_tier2.json"))
Z_SCORE_THRESHOLD = float(os.environ.get("Z_SCORE_THRESHOLD", "2.0"))
TARGET_DATE       = os.environ.get("TARGET_DATE", str(date.today()))
CLICKHOUSE_HOST   = os.environ.get("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT   = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
CLICKHOUSE_DB     = os.environ.get("CLICKHOUSE_DB", "tech_radar")
CLICKHOUSE_USER   = os.environ.get("CLICKHOUSE_USER", "root")
CLICKHOUSE_PASS   = os.environ.get("CLICKHOUSE_PASS", "root")

FEATURES_TIER1_DEFAULT = [
    "comment_count", "velocity_ratio", "acceleration",
    "user_diversity", "cross_source", "z_score", "unique_posts_ratio",
]
FEATURES_TIER2_DEFAULT = FEATURES_TIER1_DEFAULT + ["neg_ratio", "neg_score_avg", "voz_neg_ratio"]

NLP_SCHEMA = StructType([
    StructField("post_id",         StringType(), False),
    StructField("sentiment_label", StringType(), True),
    StructField("sentiment_score", FloatType(),  True),
    StructField("source",          StringType(), True),
])

BASELINE_SCHEMA = StructType([
    StructField("hour",            IntegerType(), False),
    StructField("baseline_median", FloatType(),   True),
    StructField("baseline_std",    FloatType(),   True),
])


def _get_clickhouse_client():
    import clickhouse_connect

    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        database=CLICKHOUSE_DB,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASS,
    )


def query_df(query: str) -> pd.DataFrame:
    client = _get_clickhouse_client()
    result = client.query_df(query)
    client.close()
    return result


def tmp_hdfs_dir(name: str) -> str:
    base = os.environ.get("HDFS_TMP_DIR", tempfile.gettempdir())
    return os.path.join(base, name.replace("/", "_"))


def write_parquet_and_ingest(df, table_name: str, tmp_dir: str, truncate: bool = False) -> None:
    client = _get_clickhouse_client()
    os.makedirs(tmp_dir, exist_ok=True)
    pdf = df.toPandas()

    if truncate:
        client.command(f"TRUNCATE TABLE IF EXISTS {CLICKHOUSE_DB}.{table_name}")

    if not pdf.empty:
        client.insert_df(table_name, pdf)

    client.close()


def _load_pkl(path: str):
    if not os.path.exists(path):
        print(f"[model] WARN: missing {path}, skip inference artifact.")
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _load_features(path: str, fallback: list) -> list:
    if not os.path.exists(path):
        print(f"[features] WARN: missing {path}, using default: {fallback}")
        return fallback
    with open(path) as fh:
        return json.load(fh)


def _if_predict(pdf: pd.DataFrame, pipeline, features: list) -> np.ndarray:
    if pipeline is None:
        return np.zeros(len(pdf), dtype=int)
    X = pdf[features].fillna(0).values.astype(float)
    return (pipeline.predict(X) == -1).astype(int)


def _if_score(pdf: pd.DataFrame, pipeline, features: list) -> np.ndarray:
    if pipeline is None:
        return np.zeros(len(pdf), dtype=float)
    X = pdf[features].fillna(0).values.astype(float)
    return pipeline.decision_function(X).astype(float)


def _clf_predict(pdf: pd.DataFrame, pipeline, features: list) -> np.ndarray:
    if pipeline is None:
        return np.zeros(len(pdf), dtype=int)
    X = pdf[features].fillna(0).values.astype(float)
    return pipeline.predict(X).astype(int)


def build_crisis_events(pdf: pd.DataFrame, target_date: str) -> pd.DataFrame:
    crisis_hours = pdf[pdf["is_crisis"] == 1].sort_values("hour").reset_index(drop=True)
    if crisis_hours.empty:
        return pd.DataFrame()

    crisis_hours["gap"] = crisis_hours["hour"].diff().fillna(0) > 1
    crisis_hours["event_group"] = crisis_hours["gap"].cumsum()

    def _severity(z_max: float) -> str:
        if z_max > 4:
            return "HIGH"
        if z_max > 2:
            return "MEDIUM"
        return "LOW"

    rows = []
    for _, grp in crisis_hours.groupby("event_group"):
        z_max = grp["z_score"].max()
        rows.append({
            "event_id":           hashlib.md5(
                                      f"{target_date} {grp['hour'].min():02d}:00:00".encode()
                                  ).hexdigest()[:16],
            "detected_at":        f"{target_date} {grp['hour'].min():02d}:00:00",
            "duration_hours":     int(len(grp)),
            "severity":           _severity(z_max),
            "anomaly_score":      float(grp["if_score"].min()),
            "neg_ratio":          float(grp["neg_ratio"].mean()),
            "mention_velocity":   float(grp["velocity_ratio"].max()),
            "trigger_conditions": (
                f"global_spike=1,if_spike=1,"
                f"z_score={z_max:.2f},"
                f"neg_ratio={grp['neg_ratio'].mean():.2f}"
            ),
            "affected_topics":    None,
            "evidence_post_ids":  None,
        })
    return pd.DataFrame(rows)


def load_nlp(spark: SparkSession):
    pdf = query_df(f"""
        SELECT n.post_id, n.sentiment_label, n.sentiment_score, c.source
        FROM stg_posts_nlp AS n
        INNER JOIN stg_posts_core AS c ON c.post_id = n.post_id
        WHERE toDate(c.created_at) = '{TARGET_DATE}'
          AND c.parent_id IS NOT NULL
    """)
    if pdf.empty:
        return spark.createDataFrame([], NLP_SCHEMA)
    return spark.createDataFrame(pdf, schema=NLP_SCHEMA)


def load_baseline(spark: SparkSession):
    pdf = query_df("SELECT hour, baseline_median, baseline_std FROM hourly_baseline")
    if pdf.empty:
        return spark.createDataFrame([], BASELINE_SCHEMA)
    return spark.createDataFrame(pdf, schema=BASELINE_SCHEMA)


def load_recent_hourly() -> pd.DataFrame:
    return query_df(f"""
        SELECT hour, comment_count
        FROM stg_crisis_hourly
        WHERE date < '{TARGET_DATE}'
        ORDER BY date ASC, hour ASC
        LIMIT 72
    """)


def _compute_velocity(hourly_pdf: pd.DataFrame, history_pdf: pd.DataFrame) -> pd.DataFrame:
    hourly_sorted = hourly_pdf.sort_values("hour").reset_index(drop=True)

    if history_pdf is not None and not history_pdf.empty:
        hist_counts  = history_pdf["comment_count"].astype(float).reset_index(drop=True)
        today_counts = hourly_sorted["comment_count"].astype(float)
        combined = pd.concat([hist_counts, today_counts], ignore_index=True)
    else:
        combined = hourly_sorted["comment_count"].astype(float).copy()

    roll3 = combined.rolling(window=3, min_periods=1).mean().shift(1)
    vel   = (combined / (roll3 + 1)).round(4)
    acc   = vel.diff().fillna(0).round(4)

    n = len(hourly_sorted)
    hourly_sorted["velocity_ratio"] = vel.values[-n:]
    hourly_sorted["acceleration"]   = acc.values[-n:]
    return hourly_sorted


def main() -> None:
    spark = SparkSession.builder.appName("crisis_detection_daily").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print(f"[crisis] TARGET_DATE        : {TARGET_DATE}")
    print(f"[crisis] HDFS_STG_POSTS_CORE: {HDFS_STG_POSTS_CORE}")
    print(f"[crisis] Z_SCORE_THRESHOLD  : {Z_SCORE_THRESHOLD}")
    print(f"[crisis] IF_MODEL_PATH      : {IF_MODEL_PATH}")
    print(f"[crisis] CLF_MODEL_PATH     : {CLF_MODEL_PATH}")

    print("[crisis] Step 1/9: read stg_posts_core from HDFS Parquet")
    core = (
        spark.read.parquet(HDFS_STG_POSTS_CORE)
        .select("post_id", "source", "author", "created_at", "parent_id")
        .where(to_date(col("created_at")) == lit(TARGET_DATE))
        .where(col("parent_id").isNotNull())
    )
    n_core = core.count()
    print(f"[crisis] comments on target date: {n_core:,}")
    if n_core == 0:
        print("[crisis] No comments for target date. Stop job.")
        spark.stop()
        return

    print("[crisis] Step 2/9: read stg_posts_nlp from ClickHouse")
    nlp = load_nlp(spark)
    print(f"[crisis] stg_posts_nlp records: {nlp.count():,}")

    print("[crisis] Step 3/9: read hourly_baseline from ClickHouse")
    baseline = load_baseline(spark)
    print(f"[crisis] hourly_baseline rows: {baseline.count()}")

    print("[crisis] Step 4/9: aggregate hourly structural features")
    core_with_time = (
        core
        .withColumn("date", to_date(col("created_at")))
        .withColumn("hour", spark_hour(col("created_at")))
    )
    hourly_base = (
        core_with_time
        .groupBy("date", "hour")
        .agg(
            count("*").alias("comment_count"),
            countDistinct("parent_id").alias("unique_posts"),
            countDistinct("author").alias("unique_users"),
            spark_sum(when(col("source") == "voz",       1).otherwise(0)).alias("voz_count"),
            spark_sum(when(col("source") == "vnexpress", 1).otherwise(0)).alias("vne_count"),
        )
    )

    print("[crisis] Step 5/9: aggregate hourly sentiment features")
    core_keys    = core_with_time.select("post_id", "date", "hour")
    nlp_with_time = (
        nlp.join(core_keys, on="post_id", how="inner")
        .withColumn("is_neg", when(col("sentiment_label") == "negative", 1.0).otherwise(0.0))
    )
    hourly_nlp = (
        nlp_with_time
        .groupBy("date", "hour")
        .agg(
            avg("is_neg").alias("neg_ratio"),
            avg(when(col("sentiment_label") == "negative", col("sentiment_score"))).alias("neg_score_avg"),
            avg(when(col("source") == "voz", col("is_neg"))).alias("voz_neg_ratio"),
        )
    )

    print("[crisis] Step 6/9: compute z_score and global_spike")
    hourly = (
        hourly_base
        .join(hourly_nlp, on=["date", "hour"], how="left")
        .join(baseline.select("hour", "baseline_median", "baseline_std"), on="hour", how="left")
        .withColumn("neg_ratio",     coalesce(col("neg_ratio"),     lit(0.0)).cast("float"))
        .withColumn("neg_score_avg", coalesce(col("neg_score_avg"), lit(0.0)).cast("float"))
        .withColumn("voz_neg_ratio", coalesce(col("voz_neg_ratio"), lit(0.0)).cast("float"))
        .withColumn(
            "z_score",
            when(
                col("baseline_std").isNotNull() & (col("baseline_std") > 0),
                ((col("comment_count") - col("baseline_median")) / col("baseline_std")).cast("float"),
            ).otherwise(lit(None).cast("float")),
        )
        .withColumn("global_spike", when(col("z_score") > Z_SCORE_THRESHOLD, 1).otherwise(0))
    )

    print("[crisis] Step 7/9: driver-side feature engineering and model inference")
    pdf = hourly.select(
        "date", "hour", "comment_count",
        "unique_posts", "unique_users", "vne_count", "voz_count",
        "z_score", "global_spike",
        "neg_ratio", "neg_score_avg", "voz_neg_ratio",
    ).toPandas()

    pdf["user_diversity"]     = (pdf["unique_users"] / (pdf["comment_count"] + 1)).round(4)
    pdf["cross_source"]       = (
        pdf[["vne_count", "voz_count"]].min(axis=1) /
        (pdf[["vne_count", "voz_count"]].max(axis=1) + 1)
    ).round(4)
    pdf["unique_posts_ratio"] = (pdf["unique_posts"] / (pdf["comment_count"] + 1)).round(4)

    history_pdf = load_recent_hourly()
    if not history_pdf.empty:
        print(f"[crisis] Loaded {len(history_pdf)} historical hours for velocity rolling window")
    else:
        print("[crisis] WARN: no historical data — velocity_ratio and acceleration default to 0")
    pdf = _compute_velocity(pdf, history_pdf)
    pdf["z_score"] = pdf["z_score"].fillna(0.0)

    if_pipeline  = _load_pkl(IF_MODEL_PATH)
    clf_pipeline = _load_pkl(CLF_MODEL_PATH)
    features_if  = _load_features(FEAT1_JSON_PATH, FEATURES_TIER1_DEFAULT)
    features_clf = _load_features(FEAT2_JSON_PATH, FEATURES_TIER2_DEFAULT)
    print(f"[crisis] IF  features: {features_if}")
    print(f"[crisis] CLF features: {features_clf}")

    pdf["if_spike"] = _if_predict(pdf, if_pipeline, features_if)
    pdf["if_score"] = _if_score(pdf, if_pipeline, features_if)
    pdf["is_spike"] = ((pdf["global_spike"] == 1) & (pdf["if_spike"] == 1)).astype(int)

    pdf["is_crisis"] = 0
    spike_mask = pdf["is_spike"] == 1
    if clf_pipeline is not None and spike_mask.sum() > 0:
        pdf.loc[spike_mask, "is_crisis"] = _clf_predict(pdf[spike_mask], clf_pipeline, features_clf)
    elif spike_mask.sum() > 0:
        pdf.loc[spike_mask, "is_crisis"] = 1

    pdf["date"] = pdf["date"].astype(str)
    for c in ("hour", "comment_count", "global_spike", "if_spike", "is_spike", "is_crisis"):
        pdf[c] = pdf[c].astype(int)
    for c in ("z_score", "neg_ratio", "neg_score_avg"):
        pdf[c] = pdf[c].astype(float)

    print("[crisis] Step 8/9: write Parquet and ingest stg_crisis_hourly")
    final_raw = spark.createDataFrame(
        pdf[["date", "hour", "comment_count", "z_score",
             "global_spike", "if_spike", "is_spike", "is_crisis",
             "neg_ratio", "neg_score_avg"]]
    )
    final = final_raw.select(
        to_date(col("date")).alias("date"),
        col("hour").cast("int"),
        col("comment_count").cast("int"),
        col("z_score").cast("float"),
        col("global_spike").cast("int"),
        col("if_spike").cast("int"),
        col("is_spike").cast("int"),
        col("is_crisis").cast("int"),
        col("neg_ratio").cast("float"),
        col("neg_score_avg").cast("float"),
    )
    n_rows   = final.count()
    n_spike  = final.filter(col("is_spike") == 1).count()
    n_crisis = final.filter(col("is_crisis") == 1).count()
    print(f"[crisis] Result: {n_rows} hours | {n_spike} spike | {n_crisis} crisis")
    write_parquet_and_ingest(
        final,
        "stg_crisis_hourly",
        tmp_hdfs_dir(f"stg_crisis_hourly/date={TARGET_DATE}"),
        truncate=False,
    )
    print(f"[crisis] Ingested {n_rows} rows into stg_crisis_hourly")
    final.orderBy("hour").show(24, truncate=False)

    print("[crisis] Step 9/9: build and ingest stg_crisis_events")
    events_pdf = build_crisis_events(pdf, TARGET_DATE)
    if not events_pdf.empty:
        print(f"[crisis] Events detected: {len(events_pdf)}")
        print(events_pdf[["detected_at", "severity", "duration_hours", "neg_ratio"]].to_string())
        events_spark = spark.createDataFrame(events_pdf[[
            "event_id", "detected_at", "severity",
            "anomaly_score", "neg_ratio", "mention_velocity",
            "trigger_conditions", "duration_hours",
        ]])
        write_parquet_and_ingest(
            events_spark,
            "stg_crisis_events",
            tmp_hdfs_dir(f"stg_crisis_events/date={TARGET_DATE}"),
            truncate=False,
        )
    else:
        print("[crisis] No crisis events today.")
    spark.stop()


if __name__ == "__main__":
    main()
