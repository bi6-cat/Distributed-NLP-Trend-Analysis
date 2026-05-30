"""Compute hourly baseline without JDBC.

Flow:
  1. Read stg_posts_core Parquet from HDFS.
  2. Compute hourly_baseline with Spark (comment_count, neg_ratio, unique_posts).
  3. Write result as Parquet to HDFS.
"""

from __future__ import annotations

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    coalesce,
    col,
    count,
    countDistinct,
    hour as spark_hour,
    lit,
    percentile_approx,
    stddev,
    to_date,
    when,
)

HDFS_STG_POSTS_CORE = os.environ.get(
    "HDFS_STG_POSTS_CORE",
    "hdfs://namenode:9000/user/zett/staged/stg_posts_core",
)
HDFS_STG_BASE = os.environ.get(
    "HDFS_STG_BASE",
    "hdfs://namenode:9000/user/zett/staged",
)
CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST", "clickhouse")


def main() -> None:
    spark = SparkSession.builder.appName("compute_hourly_baseline").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print(f"[baseline] Reading stg_posts_core Parquet: {HDFS_STG_POSTS_CORE}")
    core = (
        spark.read.parquet(HDFS_STG_POSTS_CORE)
        .select("post_id", "created_at", "parent_id", "source")
        .where(col("parent_id").isNotNull())
    )

    n_total = core.count()
    print(f"[baseline] Read {n_total:,} comment records")
    if n_total == 0:
        print("[baseline] No data. Stop job.")
        spark.stop()
        return

    core_with_time = (
        core
        .withColumn("date", to_date(col("created_at")))
        .withColumn("hour", spark_hour(col("created_at")))
    )

    # Lấy sentiment từ ClickHouse qua HTTP để tính neg_ratio baseline
    try:
        import urllib.request
        url = f"http://{CLICKHOUSE_HOST}:8123/?user=app&password=&database=tech_radar"
        sql = "SELECT post_id, sentiment_label FROM stg_posts_nlp"
        req = urllib.request.Request(url, data=sql.encode())
        with urllib.request.urlopen(req, timeout=30) as r:
            lines = r.read().decode().strip().split("\n")
        rows = [ln.split("\t") for ln in lines if ln]
        import pandas as pd
        nlp_pdf = pd.DataFrame(rows, columns=["post_id", "sentiment_label"])
        nlp_spark = spark.createDataFrame(nlp_pdf)
        print(f"[baseline] NLP rows from ClickHouse: {len(nlp_pdf):,}")

        core_nlp = (
            core_with_time
            .join(nlp_spark.select("post_id", "sentiment_label"), on="post_id", how="left")
            .withColumn("is_neg", when(col("sentiment_label") == "negative", 1.0).otherwise(0.0))
        )
    except Exception as e:
        print(f"[baseline] WARN: cannot load NLP from ClickHouse ({e}) — neg_ratio baseline = 0")
        core_nlp = core_with_time.withColumn("is_neg", lit(0.0))

    # Daily aggregation per (date, hour)
    daily_hourly = (
        core_nlp
        .groupBy("date", "hour")
        .agg(
            count("*").alias("daily_count"),
            avg("is_neg").alias("daily_neg_ratio"),
            countDistinct("parent_id").alias("daily_unique_posts"),
        )
    )

    # Per-hour baseline: median + stddev across all days
    baseline = (
        daily_hourly.groupBy("hour")
        .agg(
            percentile_approx("daily_count",        0.5).alias("baseline_median"),
            stddev("daily_count")                      .alias("baseline_std"),
            avg("daily_neg_ratio")                     .alias("neg_ratio_mean"),
            stddev("daily_neg_ratio")                  .alias("neg_ratio_std"),
            avg("daily_unique_posts")                  .alias("unique_posts_mean"),
            stddev("daily_unique_posts")               .alias("unique_posts_std"),
        )
        .withColumn("baseline_median",   col("baseline_median").cast("float"))
        .withColumn("baseline_std",      coalesce(col("baseline_std").cast("float"),      lit(1.0)))
        .withColumn("neg_ratio_mean",    coalesce(col("neg_ratio_mean").cast("float"),    lit(0.0)))
        .withColumn("neg_ratio_std",     coalesce(col("neg_ratio_std").cast("float"),     lit(0.01)))
        .withColumn("unique_posts_mean", coalesce(col("unique_posts_mean").cast("float"), lit(1.0)))
        .withColumn("unique_posts_std",  coalesce(col("unique_posts_std").cast("float"),  lit(1.0)))
        .select("hour", "baseline_median", "baseline_std",
                "neg_ratio_mean", "neg_ratio_std",
                "unique_posts_mean", "unique_posts_std")
    )

    n_hours = baseline.count()
    print(f"[baseline] Computed baseline for {n_hours} hours")

    hdfs_out = f"{HDFS_STG_BASE}/hourly_baseline"
    baseline.coalesce(1).write.mode("overwrite").parquet(hdfs_out)
    print(f"[baseline] Wrote {n_hours} rows to HDFS {hdfs_out}")
    baseline.orderBy("hour").show(24, truncate=False)
    spark.stop()


if __name__ == "__main__":
    main()
