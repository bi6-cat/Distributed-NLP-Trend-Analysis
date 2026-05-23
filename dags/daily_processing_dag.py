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
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago

# ── Cấu hình ──────────────────────────────────────────────────────────────────

SPARK_MASTER      = "spark://192.168.56.11:7077"
CLICKHOUSE_HOST   = "192.168.56.14"
CLICKHOUSE_DB     = "tech_radar"
JDBC_JAR          = "/opt/spark/jars/clickhouse-jdbc-0.6.0-all.jar"
IF_MODEL_PATH     = "/opt/models/isolation_forest_hourly.pkl"
CLF_MODEL_PATH    = "/opt/models/crisis_classifier.pkl"
SPARK_SUBMIT_CONN = "spark_default"   # Airflow Connection ID cho Spark

default_args = {
    "owner":             "member4-nlp-engineer",
    # False: DAG này không bị block dù weekly_retrain (hay M2) fail đêm trước
    # → crisis detection vẫn chạy với model cũ thay vì bị skip hoàn toàn
    "depends_on_past":   False,
    "email_on_failure":  False,
    "email_on_retry":    False,
    "retries":           2,
    "retry_delay":       timedelta(minutes=5),
    "start_date":        days_ago(1),
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

    # ── Sensor 1: chờ M2 spark_cleaning xong ─────────────────────────────────
    # execution_delta=2h: DAG này execution_date=04:00, upstream execution_date=02:00
    # → sensor tìm run của daily_processing_pipeline có execution_date = 04:00 - 2h = 02:00
    #
    # VERIFY SAU DEPLOY: trigger manual cả 2 DAG cùng ngày, kiểm tra sensor pass.
    # Nếu timeout → thử execution_delta=timedelta(0) hoặc dùng execution_date_fn:
    #   execution_date_fn=lambda dt: dt.replace(hour=2, minute=0, second=0, microsecond=0)
    wait_for_stg_core = ExternalTaskSensor(
        task_id="wait_for_stg_core",
        external_dag_id="daily_processing_pipeline",
        external_task_id="spark_cleaning",
        execution_delta=timedelta(hours=2),
        timeout=3600,          # tối đa 60 phút chờ
        poke_interval=60,      # kiểm tra mỗi 60 giây
        mode="reschedule",     # nhả slot khi chờ, không block worker
        soft_fail=False,
    )

    # ── Sensor 2: chờ M4 sentiment_analysis xong ─────────────────────────────
    # Cùng execution_delta=2h như sensor 1 — sentiment_analysis nằm trong
    # cùng DAG daily_processing_pipeline (schedule 02:00)
    wait_for_sentiment = ExternalTaskSensor(
        task_id="wait_for_sentiment",
        external_dag_id="daily_processing_pipeline",
        external_task_id="sentiment_analysis",
        execution_delta=timedelta(hours=2),
        timeout=3600,
        poke_interval=60,
        mode="reschedule",
        soft_fail=False,
    )

    # ── Crisis Detection Spark job ────────────────────────────────────────────
    crisis_detection = SparkSubmitOperator(
        task_id="crisis_detection_spark",
        conn_id=SPARK_SUBMIT_CONN,
        application="spark_jobs/crisis_detection.py",
        name="crisis_detection_daily_{{ ds }}",
        master=SPARK_MASTER,
        executor_memory="2g",
        driver_memory="1g",
        total_executor_cores=4,
        jars=JDBC_JAR,
        env_vars={
            "CLICKHOUSE_HOST":    CLICKHOUSE_HOST,
            "CLICKHOUSE_DB":      CLICKHOUSE_DB,
            "CLICKHOUSE_USER":    "default",
            "CLICKHOUSE_PASS":    "",
            "IF_MODEL_PATH":      IF_MODEL_PATH,
            "CLF_MODEL_PATH":     CLF_MODEL_PATH,
            "Z_SCORE_THRESHOLD":  "2.0",
            "TARGET_DATE":        "{{ ds }}",   # YYYY-MM-DD của execution_date
        },
        verbose=False,
    )

    end = DummyOperator(task_id="end")

    # ── Flow ──────────────────────────────────────────────────────────────────
    # Cả 2 sensor chạy song song (không phụ thuộc nhau)
    # Crisis job chỉ bắt đầu khi cả 2 upstream đều success
    start >> [wait_for_stg_core, wait_for_sentiment] >> crisis_detection >> end
