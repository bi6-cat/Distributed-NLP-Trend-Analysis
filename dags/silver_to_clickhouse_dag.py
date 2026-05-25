"""
Daily orchestration for loading HDFS Silver Parquet into ClickHouse staging.

The Spark layer owns producing partitioned Parquet and _SUCCESS markers.
ClickHouse pulls the data via hdfs(), then dbt builds the semantic models.
"""

from __future__ import annotations

import os
import shlex
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Iterable

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.bash import BashSensor
from airflow.utils.trigger_rule import TriggerRule

try:
    from airflow.operators.empty import EmptyOperator
except ImportError:  # Airflow < 2.3
    from airflow.operators.dummy import DummyOperator as EmptyOperator


CLICKHOUSE_HTTP_URL = os.getenv("CLICKHOUSE_HTTP_URL", "http://localhost:8123/")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
HDFS_BASE = os.getenv("HDFS_BASE", "hdfs://namenode:9000")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DBT_PROJECT_DIR = os.getenv(
    "DBT_PROJECT_DIR",
    os.path.join(REPO_ROOT, "warehouse", "dbt_project"),
)
DBT_BIN = os.getenv("DBT_BIN", "dbt")


DELETE_CORE_SQL = """
ALTER TABLE tech_radar.stg_posts_core
    DELETE WHERE created_at >= toDateTime('{{ ds }} 00:00:00')
      AND created_at <  toDateTime('{{ macros.ds_add(ds, 1) }} 00:00:00')
SETTINGS mutations_sync = 1
"""

INSERT_CORE_SQL = """
INSERT INTO tech_radar.stg_posts_core
(
    post_id,
    source,
    author_id,
    author_name,
    title,
    body,
    segmented_text,
    parent_id,
    reaction_count,
    comment_count,
    view_count,
    created_at,
    crawled_at,
    loaded_at
)
SELECT
    post_id,
    source,
    CAST(NULL AS Nullable(String)) AS author_id,
    ifNull(author, 'unknown') AS author_name,
    title,
    ifNull(body, '') AS body,
    ifNull(segmented_text, '') AS segmented_text,
    parent_id,
    reaction_count,
    comment_count,
    view_count,
    created_at,
    crawled_at,
    now() AS loaded_at
FROM hdfs(
    '{{ params.hdfs_base }}/data/silver/posts_core/date={{ ds }}/*.parquet',
    'Parquet',
    'post_id String,
     source String,
     author Nullable(String),
     title Nullable(String),
     body Nullable(String),
     clean_text Nullable(String),
     segmented_text Nullable(String),
     parent_id Nullable(String),
     reaction_count Nullable(Int32),
     view_count Nullable(Int32),
     comment_count Nullable(Int32),
     created_at DateTime,
     crawled_at DateTime'
)
"""

INSERT_NLP_SQL = """
INSERT INTO tech_radar.stg_posts_nlp
(
    post_id,
    sentiment_label,
    sentiment_score,
    model_version,
    predicted_at,
    loaded_at
)
SELECT
    post_id,
    sentiment_label,
    sentiment_score,
    model_version,
    predicted_at,
    now() AS loaded_at
FROM hdfs(
    '{{ params.hdfs_base }}/data/silver/posts_nlp/date={{ ds }}/*.parquet',
    'Parquet',
    'post_id String,
     sentiment_label String,
     sentiment_score Float32,
     model_version String,
     predicted_at DateTime'
)
"""

INSERT_POST_TOPICS_SQL = """
INSERT INTO tech_radar.stg_post_topics
(
    post_id,
    topic_id,
    topic_probability,
    model_type,
    predicted_at,
    loaded_at
)
SELECT
    post_id,
    topic_id,
    topic_probability,
    model_type,
    predicted_at,
    now() AS loaded_at
FROM hdfs(
    '{{ params.hdfs_base }}/data/silver/post_topics/date={{ ds }}/*.parquet',
    'Parquet',
    'post_id String,
     topic_id Int32,
     topic_probability Float32,
     model_type String,
     predicted_at DateTime'
)
"""

INSERT_TOPICS_SQL = """
INSERT INTO tech_radar.stg_topics
(
    topic_id,
    label,
    top_keywords,
    coherence_score,
    model_version,
    created_at
)
SELECT
    topic_id,
    label,
    top_keywords,
    coherence_score,
    model_version,
    created_at
FROM hdfs(
    '{{ params.hdfs_base }}/data/silver/topics/date={{ ds }}/*.parquet',
    'Parquet',
    'topic_id Int32,
     label String,
     top_keywords Array(String),
     coherence_score Nullable(Float32),
     model_version String,
     created_at DateTime'
)
"""

DELETE_KEYWORD_FREQ_SQL = """
ALTER TABLE tech_radar.stg_keyword_freq
    DELETE WHERE window_start >= toDateTime('{{ ds }} 00:00:00')
      AND window_start <  toDateTime('{{ macros.ds_add(ds, 1) }} 00:00:00')
SETTINGS mutations_sync = 1
"""

INSERT_KEYWORD_FREQ_SQL = """
INSERT INTO tech_radar.stg_keyword_freq
(
    keyword,
    window_start,
    window_end,
    estimated_count,
    source
)
SELECT
    keyword,
    window_start,
    window_end,
    estimated_count,
    source
FROM hdfs(
    '{{ params.hdfs_base }}/data/silver/keyword_freq/date={{ ds }}/*.parquet',
    'Parquet',
    'keyword String,
     window_start DateTime,
     window_end DateTime,
     estimated_count Int64,
     source String'
)
"""

DELETE_CRISIS_EVENTS_SQL = """
ALTER TABLE tech_radar.stg_crisis_events
    DELETE WHERE detected_at >= toDateTime('{{ ds }} 00:00:00')
      AND detected_at <  toDateTime('{{ macros.ds_add(ds, 1) }} 00:00:00')
SETTINGS mutations_sync = 1
"""

INSERT_CRISIS_EVENTS_SQL = """
INSERT INTO tech_radar.stg_crisis_events
(
    event_id,
    detected_at,
    severity,
    anomaly_score,
    trigger_conditions,
    affected_topics,
    neg_ratio,
    mention_velocity,
    evidence_post_ids
)
SELECT
    event_id,
    detected_at,
    severity,
    anomaly_score,
    trigger_conditions,
    affected_topics,
    neg_ratio,
    mention_velocity,
    evidence_post_ids
FROM hdfs(
    '{{ params.hdfs_base }}/data/silver/crisis_events/date={{ ds }}/*.parquet',
    'Parquet',
    'event_id String,
     detected_at DateTime,
     severity String,
     anomaly_score Float64,
     trigger_conditions Array(String),
     affected_topics Array(Int32),
     neg_ratio Float32,
     mention_velocity Float32,
     evidence_post_ids Array(String)'
)
"""

VALIDATE_CORE_SQL = """
SELECT count()
FROM tech_radar.stg_posts_core
WHERE created_at >= toDateTime('{{ ds }} 00:00:00')
  AND created_at <  toDateTime('{{ macros.ds_add(ds, 1) }} 00:00:00')
FORMAT TSV
"""


def _clickhouse_post(sql: str) -> str:
    params = {"user": CLICKHOUSE_USER}
    if CLICKHOUSE_PASSWORD:
        params["password"] = CLICKHOUSE_PASSWORD

    url = CLICKHOUSE_HTTP_URL
    separator = "&" if "?" in url else "?"
    url = f"{url}{separator}{urllib.parse.urlencode(params)}"

    request = urllib.request.Request(
        url=url,
        data=sql.strip().encode("utf-8"),
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.read().decode("utf-8").strip()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ClickHouse HTTP {exc.code}: {body}") from exc


def execute_clickhouse_sql(sql_statements: Iterable[str], **_) -> None:
    for sql in sql_statements:
        _clickhouse_post(sql)


def validate_core_partition(sql: str, **_) -> None:
    output = _clickhouse_post(sql)
    count = int(output.splitlines()[0]) if output else 0
    if count <= 0:
        raise ValueError("No rows loaded for the logical date in stg_posts_core")


default_args = {
    "owner": "data-platform",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def hdfs_success_sensor(task_id: str, relative_path: str, required: bool) -> BashSensor:
    return BashSensor(
        task_id=task_id,
        bash_command=(
            "hdfs dfs -test -e "
            f"{{{{ params.hdfs_base }}}}{relative_path}/date={{{{ ds }}}}/_SUCCESS"
        ),
        params={"hdfs_base": HDFS_BASE},
        poke_interval=300,
        timeout=60 * 60 * 3 if required else 60 * 30,
        mode="reschedule",
        soft_fail=not required,
    )


with DAG(
    dag_id="silver_to_clickhouse_daily",
    description="Pull Silver Parquet from HDFS into ClickHouse staging, then run dbt.",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="0 3 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["hdfs", "clickhouse", "dbt", "silver", "staging"],
) as dag:
    start = EmptyOperator(task_id="start")

    wait_posts_core = hdfs_success_sensor(
        task_id="wait_posts_core_success",
        relative_path="/data/silver/posts_core",
        required=True,
    )
    wait_posts_nlp = hdfs_success_sensor(
        task_id="wait_posts_nlp_success",
        relative_path="/data/silver/posts_nlp",
        required=False,
    )
    wait_post_topics = hdfs_success_sensor(
        task_id="wait_post_topics_success",
        relative_path="/data/silver/post_topics",
        required=False,
    )
    wait_topics = hdfs_success_sensor(
        task_id="wait_topics_success",
        relative_path="/data/silver/topics",
        required=False,
    )
    wait_keyword_freq = hdfs_success_sensor(
        task_id="wait_keyword_freq_success",
        relative_path="/data/silver/keyword_freq",
        required=False,
    )
    wait_crisis_events = hdfs_success_sensor(
        task_id="wait_crisis_events_success",
        relative_path="/data/silver/crisis_events",
        required=False,
    )

    load_posts_core = PythonOperator(
        task_id="load_posts_core_to_clickhouse",
        python_callable=execute_clickhouse_sql,
        op_kwargs={
            "sql_statements": [DELETE_CORE_SQL, INSERT_CORE_SQL],
        },
        params={"hdfs_base": HDFS_BASE},
    )

    load_posts_nlp = PythonOperator(
        task_id="load_posts_nlp_to_clickhouse",
        python_callable=execute_clickhouse_sql,
        op_kwargs={"sql_statements": [INSERT_NLP_SQL]},
        params={"hdfs_base": HDFS_BASE},
    )

    load_post_topics = PythonOperator(
        task_id="load_post_topics_to_clickhouse",
        python_callable=execute_clickhouse_sql,
        op_kwargs={"sql_statements": [INSERT_POST_TOPICS_SQL]},
        params={"hdfs_base": HDFS_BASE},
    )

    load_topics = PythonOperator(
        task_id="load_topics_to_clickhouse",
        python_callable=execute_clickhouse_sql,
        op_kwargs={"sql_statements": [INSERT_TOPICS_SQL]},
        params={"hdfs_base": HDFS_BASE},
    )

    load_keyword_freq = PythonOperator(
        task_id="load_keyword_freq_to_clickhouse",
        python_callable=execute_clickhouse_sql,
        op_kwargs={
            "sql_statements": [DELETE_KEYWORD_FREQ_SQL, INSERT_KEYWORD_FREQ_SQL],
        },
        params={"hdfs_base": HDFS_BASE},
    )

    load_crisis_events = PythonOperator(
        task_id="load_crisis_events_to_clickhouse",
        python_callable=execute_clickhouse_sql,
        op_kwargs={
            "sql_statements": [DELETE_CRISIS_EVENTS_SQL, INSERT_CRISIS_EVENTS_SQL],
        },
        params={"hdfs_base": HDFS_BASE},
    )

    optional_ingestion_done = EmptyOperator(
        task_id="optional_ingestion_done",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    assert_core_loaded = PythonOperator(
        task_id="assert_core_loaded",
        python_callable=validate_core_partition,
        op_kwargs={"sql": VALIDATE_CORE_SQL},
    )

    dbt_run = BashOperator(
        task_id="dbt_run_models",
        bash_command=(
            f"cd {shlex.quote(DBT_PROJECT_DIR)} && "
            f"{shlex.quote(DBT_BIN)} run --select staging intermediate marts && "
            f"{shlex.quote(DBT_BIN)} test --select staging intermediate marts"
        ),
        trigger_rule=TriggerRule.ALL_SUCCESS,
        retries=1,
        retry_delay=timedelta(minutes=10),
    )

    end = EmptyOperator(task_id="end")

    start >> [
        wait_posts_core,
        wait_posts_nlp,
        wait_post_topics,
        wait_topics,
        wait_keyword_freq,
        wait_crisis_events,
    ]

    wait_posts_core >> load_posts_core
    wait_posts_nlp >> load_posts_nlp
    wait_post_topics >> load_post_topics
    wait_topics >> load_topics
    wait_keyword_freq >> load_keyword_freq
    wait_crisis_events >> load_crisis_events

    [
        load_posts_nlp,
        load_post_topics,
        load_topics,
        load_keyword_freq,
        load_crisis_events,
    ] >> optional_ingestion_done

    [load_posts_core, optional_ingestion_done] >> assert_core_loaded
    assert_core_loaded >> dbt_run >> end
