"""
dags/weekly_retrain_dag.py — Airflow DAG: Weekly Model Retrain (Chủ nhật 03:00 AM)

Thứ tự tuần tự bắt buộc:
    compute_baseline → retrain_models → verify_hdfs

Tại sao phải tuần tự (không song song)?
    - retrain_models đọc stg_core cùng window 60 ngày với compute_baseline
      → chạy trước để giảm tải JDBC (chỉ 1 query lớn tại 1 thời điểm)
    - verify_hdfs phải chờ retrain_models upload xong mới có gì để kiểm tra
    - weekly_retrain.py gộp cả 3 step trong 1 SparkSession → 1 task duy nhất

Lưu ý schedule:
    Chủ nhật 03:00 AM — chạy TRƯỚC daily_crisis_detection (04:00 AM)
    → đảm bảo model mới được dùng ngay trong ngày thứ Hai
"""

from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.dates import days_ago

# ── Cấu hình ──────────────────────────────────────────────────────────────────

SPARK_MASTER      = "spark://192.168.56.11:7077"
CLICKHOUSE_HOST   = "192.168.56.14"
CLICKHOUSE_DB     = "tech_radar"
JDBC_JAR          = "/opt/spark/jars/clickhouse-jdbc-0.6.0-all.jar"
MODEL_LOCAL_DIR   = "/opt/models"
HDFS_MODEL_DIR    = "hdfs://192.168.56.11:9000/user/zett/models/crisis_detection"
HDFS_BIN          = "/opt/hadoop/bin/hdfs"
SPARK_SUBMIT_CONN = "spark_default"

# Danh sách pkl cần verify sau upload
EXPECTED_MODELS = [
    "isolation_forest_hourly.pkl",
    "crisis_classifier.pkl",
]

default_args = {
    "owner":             "member4-nlp-engineer",
    "depends_on_past":   False,
    "email_on_failure":  False,
    "email_on_retry":    False,
    "retries":           1,
    "retry_delay":       timedelta(minutes=10),
    "start_date":        days_ago(1),
}

# ── Task functions ────────────────────────────────────────────────────────────

def task_verify_hdfs(**context):
    """
    Kiểm tra từng model file đã tồn tại trên HDFS sau khi retrain_models xong.
    Raise nếu thiếu bất kỳ file nào — Airflow sẽ retry hoặc alert.
    """
    import subprocess

    missing = []
    for model_file in EXPECTED_MODELS:
        hdfs_path = f"{HDFS_MODEL_DIR}/{model_file}"
        result = subprocess.run(
            [HDFS_BIN, "dfs", "-test", "-e", hdfs_path],
            capture_output=True,
        )
        if result.returncode != 0:
            missing.append(hdfs_path)

    if missing:
        raise FileNotFoundError(
            f"Thiếu {len(missing)} model(s) trên HDFS:\n" + "\n".join(missing)
        )

    # Log danh sách files đã verify
    result = subprocess.run(
        [HDFS_BIN, "dfs", "-ls", "-h", HDFS_MODEL_DIR],
        capture_output=True, text=True,
    )
    print(f"[verify] HDFS {HDFS_MODEL_DIR}:\n{result.stdout}")
    print(f"[verify] {len(EXPECTED_MODELS)} model(s) verified ✓")


# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="weekly_retrain",
    default_args=default_args,
    description=(
        "Chủ nhật 03:00 AM — compute_baseline → retrain IF + LogReg classifier "
        "→ upload HDFS. Tuần tự bắt buộc. Chạy trước daily_crisis_detection."
    ),
    schedule_interval="0 3 * * 0",   # Chủ nhật 03:00 AM
    catchup=False,
    max_active_runs=1,
    tags=["member4", "retrain", "weekly", "phase3"],
) as dag:

    start = DummyOperator(task_id="start")

    # ── Task 1+2+3 gộp: compute_baseline + retrain + upload HDFS ─────────────
    # weekly_retrain.py chạy cả 3 step trong 1 SparkSession
    # → tránh khởi động/tắt Spark 3 lần, tiết kiệm ~2 phút overhead mỗi lần
    retrain_models = SparkSubmitOperator(
        task_id="retrain_models",
        conn_id=SPARK_SUBMIT_CONN,
        application="spark_jobs/weekly_retrain.py",
        name="weekly_retrain_{{ ds }}",
        master=SPARK_MASTER,
        executor_memory="4g",
        driver_memory="2g",
        total_executor_cores=4,
        jars=JDBC_JAR,
        env_vars={
            "CLICKHOUSE_HOST":    CLICKHOUSE_HOST,
            "CLICKHOUSE_DB":      CLICKHOUSE_DB,
            "CLICKHOUSE_USER":    "default",
            "CLICKHOUSE_PASS":    "",
            "LOOKBACK_DAYS":      "60",
            "MODEL_LOCAL_DIR":    MODEL_LOCAL_DIR,
            "HDFS_MODEL_DIR":     HDFS_MODEL_DIR,
            "IF_CONTAMINATION":   "0.05",
            "MIN_CRISIS_SAMPLES": "50",
        },
        verbose=False,
        execution_timeout=timedelta(hours=2),
    )

    # ── Task verify: kiểm tra HDFS sau khi upload ─────────────────────────────
    verify_hdfs = PythonOperator(
        task_id="verify_hdfs",
        python_callable=task_verify_hdfs,
        provide_context=True,
    )

    end = DummyOperator(task_id="end")

    # ── Flow: tuần tự bắt buộc ───────────────────────────────────────────────
    start >> retrain_models >> verify_hdfs >> end
