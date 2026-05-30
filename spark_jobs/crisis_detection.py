from __future__ import annotations

import hashlib
import os
from datetime import date

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
    ArrayType,
    DoubleType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
import pickle

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IF_MODEL_PATH = os.environ.get("IF_MODEL_PATH", "/opt/airflow/models/isolation_forest_hourly.pkl")

from spark_jobs.clickhouse_hdfs import query_df, tmp_hdfs_dir, write_parquet_and_ingest

HDFS_STG_BASE = os.environ.get(
    "HDFS_STG_BASE",
    "hdfs://namenode:9000/user/zett/staged",
)
HDFS_STG_POSTS_CORE = os.environ.get(
    "HDFS_STG_POSTS_CORE",
    "hdfs://namenode:9000/user/zett/staged/stg_posts_core",
)
TARGET_DATE = os.environ.get("TARGET_DATE", str(date.today()))

# ── Signal 1: Isolation Forest ────────────────────────────────────────────────
IF_FEATURES      = ["vol_z", "neg_z", "mv_z"]
IF_CONTAMINATION = 0.05
IF_N_ESTIMATORS  = 200
IF_MIN_CMTS      = 5

# ── Signal 2: Global Spike ────────────────────────────────────────────────────
GLOBAL_Z_THRESH = float(os.environ.get("Z_SCORE_THRESHOLD", "2.0"))

# ── Signal 3: Per-Post Spike ──────────────────────────────────────────────────
PPS_SIGMA    = 2.0
PPS_MIN_CMTS = 5

# ── Voting ────────────────────────────────────────────────────────────────────
VOTING_THRESHOLD = 2
NEG_RATIO_MIN    = 0.55

# ── Severity (calibrated từ p97/p90 non-crisis hours) ────────────────────────
SEVERITY_HIGH_THRESH = 0.2920
SEVERITY_MED_THRESH  = 0.2469

NLP_SCHEMA = StructType([
    StructField("post_id",         StringType(), False),
    StructField("sentiment_label", StringType(), True),
    StructField("sentiment_score", FloatType(),  True),
])


def load_nlp(spark: SparkSession):
    # Lấy toàn bộ stg_posts_nlp từ ClickHouse — join với HDFS core được thực hiện ở Step 5
    pdf = query_df("SELECT post_id, sentiment_label, sentiment_score FROM stg_posts_nlp")
    if pdf.empty:
        print("[crisis] WARN: stg_posts_nlp is empty in ClickHouse — neg_ratio will be 0")
        return spark.createDataFrame([], NLP_SCHEMA)
    pdf["post_id"] = pdf["post_id"].astype(str)
    print(f"[crisis] stg_posts_nlp loaded: {len(pdf):,} rows from ClickHouse")
    return spark.createDataFrame(pdf, schema=NLP_SCHEMA)


def load_baseline(spark: SparkSession) -> pd.DataFrame:
    """Đọc hourly_baseline từ HDFS (tính bởi compute_baseline.py)."""
    try:
        pdf = spark.read.parquet(f"{HDFS_STG_BASE}/hourly_baseline").toPandas()
        # Fallback nếu baseline cũ chưa có các cột neg/unique
        for col_name, default in [
            ("neg_ratio_mean",    0.0),
            ("neg_ratio_std",     0.01),
            ("unique_posts_mean", 1.0),
            ("unique_posts_std",  1.0),
        ]:
            if col_name not in pdf.columns:
                pdf[col_name] = default
        print(f"[crisis] hourly_baseline loaded: {len(pdf)} hours")
        return pdf.set_index("hour")
    except Exception as e:
        print(f"[crisis] WARN: cannot load hourly_baseline ({e}) — z-scores default to 0")
        return pd.DataFrame()


def get_evidence_posts(core_with_time, nlp, crisis_hour_set, min_cmts=5, neg_threshold=0.5):
    return (
        core_with_time
        .filter(col("hour").isin(list(crisis_hour_set)))
        .filter(col("parent_id").isNotNull())
        .join(nlp.select("post_id", "sentiment_label"), on="post_id", how="inner")
        .withColumn("is_neg", when(col("sentiment_label") == "negative", 1.0).otherwise(0.0))
        .groupBy("hour", "parent_id")
        .agg(
            count("*").alias("cmt_count"),
            avg("is_neg").alias("post_neg_ratio"),
        )
        .filter(col("cmt_count") >= min_cmts)
        .filter(col("post_neg_ratio") >= neg_threshold)
        .orderBy("hour", col("cmt_count").desc())
        .collect()
    )


# ── 3-Signal Crisis Logic (driver-side pandas) ───────────────────────────────

def _add_zscores(pdf: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """
    Tính vol_z, neg_z, mv_z từ hourly_baseline (μ, σ từ lịch sử toàn bộ).
    z = (x - μ) / σ  — stable, không phụ thuộc số rows trong ngày.
    """
    pdf = pdf.copy()
    if baseline.empty:
        pdf["vol_z"] = 0.0
        pdf["neg_z"] = 0.0
        pdf["mv_z"]  = 0.0
        return pdf

    def _z(val_series, mean_col, std_col):
        mu  = pdf["hour"].map(baseline[mean_col]).fillna(0.0)
        std = pdf["hour"].map(baseline[std_col]).fillna(0.01).clip(lower=0.01)
        return ((val_series - mu) / std).fillna(0.0)

    pdf["vol_z"] = _z(pdf["comment_count"], "baseline_median",   "baseline_std")
    pdf["neg_z"] = _z(pdf["neg_ratio"],     "neg_ratio_mean",    "neg_ratio_std")
    pdf["mv_z"]  = _z(pdf["unique_posts"],  "unique_posts_mean", "unique_posts_std")
    return pdf


def _load_or_fit_isolation_forest(pdf: pd.DataFrame) -> Pipeline:
    if os.path.exists(IF_MODEL_PATH):
        with open(IF_MODEL_PATH, "rb") as fh:
            artifact = pickle.load(fh)
        pipe = artifact["pipeline"]
        print(f"[crisis] IF model loaded from {IF_MODEL_PATH} (trained {artifact.get('trained_at','?')})")
        return pipe

    print(f"[crisis] IF model not found at {IF_MODEL_PATH} — fitting on today's data")
    normal = pdf[pdf["comment_count"] >= IF_MIN_CMTS]
    if len(normal) < 5:
        normal = pdf
    X = normal[IF_FEATURES].fillna(0).values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("if",     IsolationForest(
            n_estimators  = IF_N_ESTIMATORS,
            contamination = IF_CONTAMINATION,
            random_state  = 42,
        )),
    ])
    pipe.fit(X)
    return pipe


def _apply_signals(pdf: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    pdf = _add_zscores(pdf, baseline)

    ifo = _load_or_fit_isolation_forest(pdf)
    X_all = pdf[IF_FEATURES].fillna(0).values
    pdf["if_score"]  = -ifo.decision_function(X_all)
    pdf["signal_if"] = (ifo.predict(X_all) == -1).astype(int)

    pdf["signal_global"] = (
        (pdf["vol_z"] > GLOBAL_Z_THRESH) &
        (pdf["comment_count"] >= IF_MIN_CMTS)
    ).astype(int)

    if "signal_post" not in pdf.columns:
        pdf["signal_post"] = 0

    pdf["votes"] = pdf["signal_if"] + pdf["signal_global"] + pdf["signal_post"]
    pdf["is_crisis"] = (
        (pdf["votes"] >= VOTING_THRESHOLD) &
        (pdf["neg_ratio"] >= NEG_RATIO_MIN)
    ).astype(int)
    pdf["is_trending"] = (
        (pdf["signal_global"] == 1) & (pdf["is_crisis"] == 0)
    ).astype(int)

    n_crisis   = pdf["is_crisis"].sum()
    n_trending = pdf["is_trending"].sum()
    print(f"[crisis] Voting ≥{VOTING_THRESHOLD}/3: {n_crisis} crisis hours | {n_trending} trending hours")
    print(f"[crisis] vol_z range: [{pdf['vol_z'].min():.2f}, {pdf['vol_z'].max():.2f}] "
          f"| neg_z: [{pdf['neg_z'].min():.2f}, {pdf['neg_z'].max():.2f}]")
    return pdf


def _compute_pps(core_pdf: pd.DataFrame) -> pd.DataFrame:
    if core_pdf.empty:
        return pd.DataFrame(columns=["hour", "signal_post"])

    daily = (
        core_pdf.groupby(["parent_id", "date"])
        .size().rename("daily_cmts").reset_index()
    )
    daily_global = (
        daily.groupby("date")["daily_cmts"]
        .agg(["mean", "std"]).reset_index()
    )
    daily_global.columns = ["date", "global_mean", "global_std"]
    daily_global["global_std"] = daily_global["global_std"].fillna(1.0)

    daily = daily.merge(daily_global, on="date", how="left")
    daily["post_thresh"] = daily["global_mean"] + PPS_SIGMA * daily["global_std"]
    daily["post_spike"]  = (
        (daily["daily_cmts"] > daily["post_thresh"]) &
        (daily["daily_cmts"] >= PPS_MIN_CMTS)
    )

    spike_set = set(zip(
        daily.loc[daily["post_spike"], "parent_id"].astype(str),
        daily.loc[daily["post_spike"], "date"].astype(str),
    ))
    core_pdf = core_pdf.copy()
    core_pdf["has_pps"] = core_pdf.apply(
        lambda r: 1 if (str(r["parent_id"]), str(r["date"])) in spike_set else 0,
        axis=1,
    )
    pps = (
        core_pdf.groupby("hour")["has_pps"]
        .max().reset_index()
        .rename(columns={"has_pps": "signal_post"})
    )
    pps["signal_post"] = pps["signal_post"].astype(int)
    print(f"[crisis] Per-Post Spike: {daily['post_spike'].sum()} posts spiked "
          f"→ {pps['signal_post'].sum()} hours flagged")
    return pps


def build_events(pdf: pd.DataFrame, target_date: str, post_ids_by_hour: dict) -> pd.DataFrame:
    rows = []
    for _, row in pdf[pdf["is_crisis"] == 1].iterrows():
        h = int(row["hour"])

        vol_z_n = float(np.clip(row.get("vol_z", 0), 0, 5)) / 5
        neg_z_n = float(np.clip(row.get("neg_z", 0), 0, 5)) / 5
        neg_abs = float(np.clip((row.get("neg_ratio", 0) - 0.55) / 0.45, 0, 1))
        mv_z_n  = float(np.clip(row.get("mv_z",  0), 0, 5)) / 5
        anomaly_score = float(np.clip(
            0.40 * vol_z_n + 0.35 * neg_z_n + 0.15 * neg_abs + 0.10 * mv_z_n,
            0, 1,
        ))

        if anomaly_score >= SEVERITY_HIGH_THRESH:
            severity = "HIGH"
        elif anomaly_score >= SEVERITY_MED_THRESH:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        triggers = []
        if row.get("signal_if")     == 1: triggers.append("isolation_forest")
        if row.get("signal_global") == 1: triggers.append(f"global_spike_z{row.get('vol_z', 0):.2f}")
        if row.get("signal_post")   == 1: triggers.append("per_post_spike")

        evidence = post_ids_by_hour.get(h, [])
        raw_id   = f"{target_date}-{h:02d}_{evidence[0] if evidence else 'none'}"
        event_id = "ce_" + hashlib.md5(raw_id.encode()).hexdigest()[:12]

        rows.append({
            "event_id":           event_id,
            "detected_at":        f"{target_date} {h:02d}:00:00",
            "severity":           severity,
            "anomaly_score":      round(anomaly_score, 6),
            "trigger_conditions": triggers,
            "affected_topics":    [],
            "neg_ratio":          round(float(row["neg_ratio"]), 6),
            "mention_velocity":   round(float(row.get("unique_posts", 0)), 6),
            "evidence_post_ids":  evidence,
        })

    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.sort_values("detected_at").reset_index(drop=True)
        print(f"[crisis] Events: {len(events)} | HIGH={(events['severity']=='HIGH').sum()} "
              f"MEDIUM={(events['severity']=='MEDIUM').sum()} LOW={(events['severity']=='LOW').sum()}")
    return events


def main() -> None:
    spark = SparkSession.builder.appName("crisis_detection_daily").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print(f"[crisis] TARGET_DATE        : {TARGET_DATE}")
    print(f"[crisis] HDFS_STG_POSTS_CORE: {HDFS_STG_POSTS_CORE}")
    print(f"[crisis] GLOBAL_Z_THRESH    : {GLOBAL_Z_THRESH}")
    print(f"[crisis] NEG_RATIO_MIN      : {NEG_RATIO_MIN}")
    print(f"[crisis] VOTING_THRESHOLD   : {VOTING_THRESHOLD}/3")

    # ── Step 1: stg_posts_core từ HDFS ───────────────────────────────────────
    print("[crisis] Step 1/8: read stg_posts_core from HDFS Parquet")
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

    # ── Step 2: stg_posts_nlp từ ClickHouse ──────────────────────────────────
    print("[crisis] Step 2/8: read stg_posts_nlp from ClickHouse")
    nlp = load_nlp(spark)
    print(f"[crisis] stg_posts_nlp records: {nlp.count():,}")

    # ── Step 3: hourly_baseline từ HDFS ──────────────────────────────────────
    print("[crisis] Step 3/8: read hourly_baseline from HDFS")
    baseline = load_baseline(spark)

    # ── Step 4: aggregate hourly structural features ──────────────────────────
    print("[crisis] Step 4/8: aggregate hourly structural features")
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

    # ── Step 5: aggregate hourly sentiment features ───────────────────────────
    print("[crisis] Step 5/8: aggregate hourly sentiment features")
    core_keys     = core_with_time.select("post_id", "source", "date", "hour")
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

    # ── Step 6: join và toPandas ──────────────────────────────────────────────
    print("[crisis] Step 6/8: join features and collect to driver")
    hourly = (
        hourly_base
        .join(hourly_nlp, on=["date", "hour"], how="left")
        .withColumn("neg_ratio",     coalesce(col("neg_ratio"),     lit(0.0)).cast("float"))
        .withColumn("neg_score_avg", coalesce(col("neg_score_avg"), lit(0.0)).cast("float"))
        .withColumn("voz_neg_ratio", coalesce(col("voz_neg_ratio"), lit(0.0)).cast("float"))
    )
    pdf = hourly.select(
        "date", "hour", "comment_count",
        "unique_posts", "unique_users", "vne_count", "voz_count",
        "neg_ratio", "neg_score_avg", "voz_neg_ratio",
    ).toPandas()
    pdf = pdf.sort_values(["date", "hour"]).reset_index(drop=True)

    core_pdf = core_with_time.select("post_id", "parent_id", "date", "hour").toPandas()
    core_pdf["date"] = core_pdf["date"].astype(str)

    pps = _compute_pps(core_pdf)
    pdf = pdf.merge(pps, on="hour", how="left")
    pdf["signal_post"] = pdf["signal_post"].fillna(0).astype(int)

    # ── Step 7: 3-signal voting ───────────────────────────────────────────────
    print("[crisis] Step 7/8: 3-signal voting (IF + GlobalSpike + PerPostSpike)")
    pdf = _apply_signals(pdf, baseline)

    pdf["date"] = pdf["date"].astype(str)
    for c in ("hour", "comment_count", "signal_if", "signal_global", "signal_post",
              "votes", "is_crisis", "is_trending"):
        pdf[c] = pdf[c].astype(int)
    for c in ("neg_ratio", "neg_score_avg", "vol_z", "neg_z", "mv_z", "if_score"):
        pdf[c] = pdf[c].astype(float)

    # ── Step 8: write stg_crisis_hourly to HDFS ───────────────────────────────
    print("[crisis] Step 8/8: write results to HDFS and ClickHouse")
    final_spark = spark.createDataFrame(
        pdf[["date", "hour", "comment_count",
             "vol_z", "neg_z", "mv_z",
             "signal_if", "signal_global", "signal_post",
             "votes", "is_crisis", "is_trending",
             "neg_ratio", "neg_score_avg"]]
    ).select(
        to_date(col("date")).alias("date"),
        col("hour").cast("int"),
        col("comment_count").cast("int"),
        col("vol_z").cast("float"),
        col("neg_z").cast("float"),
        col("mv_z").cast("float"),
        col("signal_if").cast("int"),
        col("signal_global").cast("int"),
        col("signal_post").cast("int"),
        col("votes").cast("int"),
        col("is_crisis").cast("int"),
        col("is_trending").cast("int"),
        col("neg_ratio").cast("float"),
        col("neg_score_avg").cast("float"),
    )
    n_rows     = final_spark.count()
    n_crisis   = final_spark.filter(col("is_crisis")   == 1).count()
    n_trending = final_spark.filter(col("is_trending")  == 1).count()
    print(f"[crisis] Result: {n_rows} hours | {n_crisis} crisis | {n_trending} trending")
    final_spark.write.mode("overwrite").parquet(
        f"{HDFS_STG_BASE}/stg_crisis_hourly/date={TARGET_DATE}"
    )
    print(f"[crisis] Wrote {n_rows} rows → HDFS stg_crisis_hourly/date={TARGET_DATE}")
    final_spark.orderBy("hour").show(24, truncate=False)

    # ── Step 9: build events và ingest ────────────────────────────────────────
    crisis_hour_set = set(int(h) for h in pdf.loc[pdf["is_crisis"] == 1, "hour"].tolist())
    post_ids_by_hour: dict[int, list[str]] = {}
    if crisis_hour_set:
        for row in get_evidence_posts(core_with_time, nlp, crisis_hour_set):
            post_ids_by_hour.setdefault(int(row["hour"]), []).append(str(row["parent_id"]))
        print(f"[crisis] Crisis hours: {len(crisis_hour_set)} | evidence posts: "
              f"{sum(len(v) for v in post_ids_by_hour.values())}")

    events_pdf = build_events(pdf, TARGET_DATE, post_ids_by_hour)

    if not events_pdf.empty:
        print(events_pdf[["detected_at", "severity", "anomaly_score",
                          "neg_ratio", "trigger_conditions"]].to_string())

        events_schema = StructType([
            StructField("event_id",           StringType(),             False),
            StructField("detected_at",        TimestampType(),          False),
            StructField("severity",           StringType(),             True),
            StructField("anomaly_score",      DoubleType(),             True),
            StructField("neg_ratio",          FloatType(),              True),
            StructField("mention_velocity",   FloatType(),              True),
            StructField("trigger_conditions", ArrayType(StringType()),  True),
            StructField("affected_topics",    ArrayType(IntegerType()), True),
            StructField("evidence_post_ids",  ArrayType(StringType()),  True),
        ])
        events_pdf["detected_at"] = pd.to_datetime(events_pdf["detected_at"])
        events_spark = spark.createDataFrame(
            events_pdf[[
                "event_id", "detected_at", "severity",
                "anomaly_score", "neg_ratio", "mention_velocity",
                "trigger_conditions", "affected_topics", "evidence_post_ids",
            ]],
            schema=events_schema,
        )
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
