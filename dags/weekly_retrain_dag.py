"""Airflow DAG for weekly model retraining."""

from datetime import timedelta
from urllib.parse import urlparse

import requests
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.dates import days_ago


SPARK_MASTER = "spark://spark-master:7077"
CLICKHOUSE_HOST = "clickhouse"
CLICKHOUSE_DB = "tech_radar"
CLICKHOUSE_USER = "app"
CLICKHOUSE_PASS = ""
MODEL_LOCAL_DIR = "/tmp/airflow_models"
HDFS_MODEL_DIR = "hdfs://namenode:9000/user/zett/models/crisis_detection"
HDFS_STG_POSTS_CORE = "hdfs://namenode:9000/user/zett/staged/stg_posts_core"
WEBHDFS_HOST = "namenode"
WEBHDFS_PORT = 9870
HDFS_USER = "zett"
SPARK_SUBMIT_CONN = "spark_default"

EXPECTED_MODELS = [
    "isolation_forest_hourly.pkl",
    "crisis_classifier.pkl",
]

default_args = {
    "owner": "member4-nlp-engineer",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "start_date": days_ago(1),
}


def task_verify_hdfs(**context):
    hdfs_dir = urlparse(HDFS_MODEL_DIR).path
    missing = []
    for model_file in EXPECTED_MODELS:
        hdfs_path = f"{hdfs_dir.rstrip('/')}/{model_file}"
        url = (
            f"http://{WEBHDFS_HOST}:{WEBHDFS_PORT}/webhdfs/v1{hdfs_path}"
            f"?op=GETFILESTATUS&user.name={HDFS_USER}"
        )
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            missing.append(hdfs_path)

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} model(s) on HDFS:\n" + "\n".join(missing)
        )

    list_url = (
        f"http://{WEBHDFS_HOST}:{WEBHDFS_PORT}/webhdfs/v1{hdfs_dir}"
        f"?op=LISTSTATUS&user.name={HDFS_USER}"
    )
    print(f"[verify] HDFS {HDFS_MODEL_DIR}: {requests.get(list_url, timeout=30).text}")
    print(f"[verify] {len(EXPECTED_MODELS)} model(s) verified")


with DAG(
    dag_id="weekly_retrain",
    default_args=default_args,
    description="Weekly retrain using HDFS Parquet and ClickHouse HTTP ingest.",
    schedule_interval="0 3 * * 0",
    catchup=False,
    max_active_runs=1,
    tags=["member4", "retrain", "weekly", "phase3"],
) as dag:
    start = DummyOperator(task_id="start")

    retrain_models = SparkSubmitOperator(
        task_id="retrain_models",
        conn_id=SPARK_SUBMIT_CONN,
        application="spark_jobs/weekly_retrain.py",
        name="weekly_retrain_{{ ds }}",
        conf={"spark.master": SPARK_MASTER},
        executor_memory="4g",
        driver_memory="2g",
        total_executor_cores=4,
        env_vars={
            "CLICKHOUSE_HOST": CLICKHOUSE_HOST,
            "CLICKHOUSE_DB": CLICKHOUSE_DB,
            "CLICKHOUSE_USER": CLICKHOUSE_USER,
            "CLICKHOUSE_PASS": CLICKHOUSE_PASS,
            "HDFS_STG_POSTS_CORE": HDFS_STG_POSTS_CORE,
            "WEBHDFS_HOST": WEBHDFS_HOST,
            "WEBHDFS_PORT": str(WEBHDFS_PORT),
            "HDFS_USER": HDFS_USER,
            "LOOKBACK_DAYS": "60",
            "MODEL_LOCAL_DIR": MODEL_LOCAL_DIR,
            "HDFS_MODEL_DIR": HDFS_MODEL_DIR,
            "IF_CONTAMINATION": "0.05",
            "MIN_CRISIS_SAMPLES": "50",
        },
        verbose=False,
        execution_timeout=timedelta(hours=2),
    )

    verify_hdfs = PythonOperator(
        task_id="verify_hdfs",
        python_callable=task_verify_hdfs,
        provide_context=True,
    )

    end = DummyOperator(task_id="end")

    start >> retrain_models >> verify_hdfs >> end
