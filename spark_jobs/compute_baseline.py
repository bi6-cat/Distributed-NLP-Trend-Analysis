"""
compute_baseline.py  —  Weekly Spark job (Chủ nhật 03:00 AM)

Đọc stg_core 60 ngày → tính hourly_baseline → ghi ClickHouse.
Phải chạy TRƯỚC crisis_detection daily job.

Cách chạy:
    spark-submit \\
        --master spark://192.168.56.11:7077 \\
        --jars /opt/spark/jars/clickhouse-jdbc-0.6.0-all.jar \\
        spark_jobs/compute_baseline.py

Biến môi trường:
    CLICKHOUSE_HOST : IP ClickHouse node   (mặc định 192.168.56.14)
    CLICKHOUSE_PORT : JDBC port            (mặc định 8123)
    CLICKHOUSE_DB   : database             (mặc định tech_radar)
    CLICKHOUSE_USER : user                 (mặc định default)
    CLICKHOUSE_PASS : password             (mặc định '')
    JDBC_JAR        : đường dẫn tới jar   (mặc định /opt/spark/jars/clickhouse-jdbc-0.6.0-all.jar)
    LOOKBACK_DAYS   : số ngày lịch sử     (mặc định 60)
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    hour as spark_hour,
    percentile_approx,
    stddev,
    lit,
    coalesce,
)

CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST", "192.168.56.14")
CLICKHOUSE_PORT = os.environ.get("CLICKHOUSE_PORT", "8123")
CLICKHOUSE_DB   = os.environ.get("CLICKHOUSE_DB",   "tech_radar")
CLICKHOUSE_USER = os.environ.get("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASS = os.environ.get("CLICKHOUSE_PASS", "")
JDBC_JAR        = os.environ.get("JDBC_JAR", "/opt/spark/jars/clickhouse-jdbc-0.6.0-all.jar")
LOOKBACK_DAYS   = int(os.environ.get("LOOKBACK_DAYS", "60"))

CLICKHOUSE_URL  = f"jdbc:clickhouse://{CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}/{CLICKHOUSE_DB}"
CLICKHOUSE_OPTS = {
    "driver":   "com.clickhouse.jdbc.ClickHouseDriver",
    "user":     CLICKHOUSE_USER,
    "password": CLICKHOUSE_PASS,
}


def main():
    spark = (
        SparkSession.builder
        .appName("compute_hourly_baseline")
        .config("spark.jars", JDBC_JAR)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # 1. Đọc stg_core — chỉ lấy comments (parent_id IS NOT NULL)
    print(f"[baseline] Đọc stg_core ({LOOKBACK_DAYS} ngày gần nhất)...")
    query = f"""(
        SELECT post_id, created_at, parent_id
        FROM stg_core
        WHERE toDate(created_at) >= today() - {LOOKBACK_DAYS}
          AND parent_id IS NOT NULL
    ) t"""

    core = spark.read.jdbc(
        url=CLICKHOUSE_URL,
        table=query,
        properties=CLICKHOUSE_OPTS,
    )
    n_total = core.count()
    print(f"[baseline] Đọc được {n_total:,} comment records")

    if n_total == 0:
        print("[baseline] Không có dữ liệu. Dừng job.")
        spark.stop()
        return

    # 2. Bước 1: đếm comment theo (date, hour) — mỗi dòng = 1 ô (ngày × giờ)
    #    Đây là "daily_count" cho từng giờ của từng ngày cụ thể
    from pyspark.sql.functions import to_date
    daily_hourly = (
        core
        .withColumn("date", to_date(col("created_at")))
        .withColumn("hour", spark_hour(col("created_at")))
        .groupBy("date", "hour")
        .agg(count("*").alias("daily_count"))
    )

    # 3. Bước 2: tính median + std TRÊN các daily_count cùng giờ
    #    percentile_approx(0.5) = median — ít bị kéo bởi spike ngày đột biến
    baseline = (
        daily_hourly
        .groupBy("hour")
        .agg(
            percentile_approx("daily_count", 0.5).alias("baseline_median"),
            stddev("daily_count").alias("baseline_std"),
        )
        .withColumn("baseline_median", col("baseline_median").cast("float"))
        .withColumn(
            "baseline_std",
            coalesce(col("baseline_std").cast("float"), lit(0.0)),
        )
        .fillna(0.0)
    )

    n_hours = baseline.count()
    print(f"[baseline] Tính xong baseline cho {n_hours} giờ")

    # 4. Ghi vào ClickHouse — ReplacingMergeTree tự deduplicate theo hour
    print("[baseline] Ghi hourly_baseline vào ClickHouse...")
    baseline.write.jdbc(
        url=CLICKHOUSE_URL,
        table="hourly_baseline",
        mode="append",
        properties=CLICKHOUSE_OPTS,
    )
    print(f"[baseline] Hoàn tất — đã cập nhật {n_hours} giờ (hour 0-23)")

    baseline.show(24, truncate=False)
    spark.stop()


if __name__ == "__main__":
    main()
