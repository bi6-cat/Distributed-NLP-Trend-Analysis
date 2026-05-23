"""Compute hourly baseline without JDBC.

Flow:
  1. Read stg_posts_core Parquet from HDFS.
  2. Compute hourly_baseline with Spark.
  3. Write result as Parquet to HDFS.
  4. Load it into ClickHouse with INSERT SELECT FROM hdfs(...).
"""

from __future__ import annotations

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    coalesce,
    col,
    count,
    current_date,
    date_sub,
    hour as spark_hour,
    lit,
    percentile_approx,
    stddev,
    to_date,
)

from spark_jobs.clickhouse_hdfs import tmp_hdfs_dir, write_parquet_and_ingest


LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "60"))
HDFS_STG_POSTS_CORE = os.environ.get(
    "HDFS_STG_POSTS_CORE",
    "hdfs://namenode:9000/user/zett/staged/stg_posts_core",
)


def main() -> None:
    spark = SparkSession.builder.appName("compute_hourly_baseline").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print(f"[baseline] Reading stg_posts_core Parquet: {HDFS_STG_POSTS_CORE}")
    core = (
        spark.read.parquet(HDFS_STG_POSTS_CORE)
        .select("post_id", "created_at", "parent_id")
        .where(col("parent_id").isNotNull())
        .where(to_date(col("created_at")) >= date_sub(current_date(), LOOKBACK_DAYS))
    )

    n_total = core.count()
    print(f"[baseline] Read {n_total:,} comment records")
    if n_total == 0:
        print("[baseline] No data. Stop job.")
        spark.stop()
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
    print(f"[baseline] Computed baseline for {n_hours} hours")

    write_parquet_and_ingest(
        baseline,
        "hourly_baseline",
        tmp_hdfs_dir("hourly_baseline"),
        truncate=True,
    )
    print(f"[baseline] Ingested {n_hours} rows into hourly_baseline")

    baseline.show(24, truncate=False)
    spark.stop()


if __name__ == "__main__":
    main()
