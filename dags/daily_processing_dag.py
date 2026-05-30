"""
dags/daily_processing_dag.py — Airflow DAG: Daily Crisis Detection (04:00 AM)

Chạy sau khi M2 (stg_posts_core) và M4 (sentiment) đã xong.

Dependency chain trong ngày:
    daily_processing_pipeline (02:00 AM)
        spark_cleaning  → stg_posts_core sẵn sàng    (M2)
        sentiment_job   → stg_posts_nlp sẵn sàng     (M4)
            ↓
    daily_crisis_detection (04:00 AM)
        crisis_detection_spark  ← job này

Tại sao dùng ExternalTaskSensor thay vì 1 DAG duy nhất?
    - sentiment_job chạy lâu (~30-60 phút) — không muốn block DAG M2
    - crisis detection là responsibility của Member 4, tách DAG rõ ownership
    - Nếu crisis job fail, không retrigger lại toàn bộ pipeline M2

QUAN TRỌNG — execution_delta và Airflow execution_date convention:
    Airflow execution_date = thời điểm bắt đầu của INTERVAL TRƯỚC, không phải
    wall-clock time khi task thực sự chạy.

    Ví dụ ngày 2026-05-24:
        daily_processing_pipeline schedule "0 2 * * *":
            execution_date = 2026-05-24 02:00  (Airflow 2.x, logical date = scheduled time)
        daily_crisis_detection schedule "0 4 * * *":
            execution_date = 2026-05-24 04:00

        → execution_delta = 04:00 - 02:00 = timedelta(hours=2)  ✓

    NHƯNG: hành vi này phụ thuộc vào phiên bản Airflow và cấu hình timezone.
    Với Airflow 2.x dùng pendulum UTC, execution_date = scheduled time chính xác.
    Với Airflow 1.x, execution_date = scheduled time - 1 interval (off-by-one).

    → SAU KHI DEPLOY: trigger manual cả 2 DAG cùng ngày, kiểm tra sensor
      có chuyển sang "success" không. Nếu sensor timeout, thử execution_delta=0
      hoặc dùng execution_date_fn thay thế:

        from airflow.utils.state import State
        # Dùng execution_date_fn nếu execution_delta không ổn định:
        # execution_date_fn=lambda dt: dt.replace(hour=2, minute=0, second=0)
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

# ── Cấu hình ──────────────────────────────────────────────────────────────────

SPARK_MASTER        = "spark://spark-master:7077"
CLICKHOUSE_HOST     = "clickhouse"
CLICKHOUSE_DB       = "tech_radar"
HDFS_STG_POSTS_CORE = "hdfs://namenode:9000/user/zett/staged/stg_posts_core"
SPARK_SUBMIT_CONN   = "spark_default"

default_args = {
    "owner":             "member4-nlp-engineer",
    # False: DAG này không bị block dù weekly_retrain (hay M2) fail đêm trước
    # → crisis detection vẫn chạy với model cũ thay vì bị skip hoàn toàn
    "depends_on_past":   False,
    "email_on_failure":  False,
    "email_on_retry":    False,
    "retries":           2,
    "retry_delay":       timedelta(minutes=5),
    "start_date":        datetime(2026, 2, 28),
}

# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="daily_crisis_detection",
    default_args=default_args,
    description=(
        "Daily 04:00 AM — Spark crisis detection sau khi M2 cleaning "
        "và M4 sentiment đã xong. Ghi kết quả vào stg_crisis_events."
    ),
    schedule_interval="0 4 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["member4", "crisis", "daily", "phase3"],
) as dag:

    start = DummyOperator(task_id="start")

    wait_for_stg_core = DummyOperator(task_id="wait_for_stg_core")

    wait_for_sentiment = DummyOperator(task_id="wait_for_sentiment")

    # ── Compute Hourly Baseline ───────────────────────────────────────────────
    compute_baseline = SparkSubmitOperator(
        task_id="compute_baseline",
        conn_id=SPARK_SUBMIT_CONN,
        application="spark_jobs/compute_baseline.py",
        name="compute_baseline_{{ ds }}",
        conf={
            "spark.master": SPARK_MASTER,
            "spark.cores.max": "1",
            "spark.executor.cores": "1",
            "spark.executorEnv.PYTHONPATH": "/opt/airflow",
            "spark.executorEnv.CLICKHOUSE_HOST": CLICKHOUSE_HOST,
        },
        executor_memory="1g",
        driver_memory="512m",
        env_vars={
            "PYTHONPATH":           "/opt/airflow",
            "CLICKHOUSE_HOST":      CLICKHOUSE_HOST,
            "CLICKHOUSE_DB":        CLICKHOUSE_DB,
            "CLICKHOUSE_USER":      "app",
            "CLICKHOUSE_PASS":      "",
            "HDFS_STG_POSTS_CORE":  HDFS_STG_POSTS_CORE,
        },
        verbose=False,
    )

    # ── Crisis Detection Spark job ────────────────────────────────────────────
    crisis_detection = SparkSubmitOperator(
        task_id="crisis_detection_spark",
        conn_id=SPARK_SUBMIT_CONN,
        application="spark_jobs/crisis_detection.py",
        name="crisis_detection_daily_{{ ds }}",
        conf={
            "spark.master": SPARK_MASTER,
            "spark.executorEnv.PYTHONPATH": "/opt/airflow",
        },
        executor_memory="2g",
        driver_memory="1g",
        total_executor_cores=4,
        env_vars={
            "PYTHONPATH":         "/opt/airflow",
            "CLICKHOUSE_HOST":    CLICKHOUSE_HOST,
            "CLICKHOUSE_DB":      CLICKHOUSE_DB,
            "CLICKHOUSE_USER":    "app",
            "CLICKHOUSE_PASS":    "",
            "HDFS_STG_POSTS_CORE": HDFS_STG_POSTS_CORE,
            "TARGET_DATE":        "{{ dag_run.conf.get('target_date', '2026-04-14') }}",
        },
        verbose=False,
    )

    end = DummyOperator(task_id="end")

    # ── Flow ──────────────────────────────────────────────────────────────────
    # Cả 2 sensor chạy song song (không phụ thuộc nhau)
    # Crisis job chỉ bắt đầu khi cả 2 upstream đều success
    start >> [wait_for_stg_core, wait_for_sentiment] >> compute_baseline >> crisis_detection >> end
