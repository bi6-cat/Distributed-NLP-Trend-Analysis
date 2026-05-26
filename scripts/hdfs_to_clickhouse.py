"""Load the six staged Parquet datasets from HDFS into ClickHouse.

This module is intentionally narrow: it only loads tables declared in
warehouse/clickhouse/init_schema.sql and never uses SELECT *.
"""

from __future__ import annotations

import argparse
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


HDFS_USER = _env("HDFS_USER", _env("HADOOP_USER_NAME", "root"))
HDFS_NAMENODE = _env("HDFS_NAMENODE", "namenode:9000")
HDFS_STAGED_ROOT = _env(
    "HDFS_STAGED_ROOT",
    f"hdfs://{HDFS_NAMENODE}/user/{HDFS_USER}/staged",
).rstrip("/")
WEBHDFS_HOST = _env("WEBHDFS_HOST", "namenode:9870")

CLICKHOUSE_HOST = _env("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT = _env("CLICKHOUSE_PORT", "8123")
CLICKHOUSE_DB = _env("CLICKHOUSE_DB", "tech_radar")
CLICKHOUSE_USER = _env("CLICKHOUSE_USER", "root")
CLICKHOUSE_PASSWORD = _env("CLICKHOUSE_PASSWORD", _env("CLICKHOUSE_PASS", "root"))


@dataclass(frozen=True)
class TableSpec:
    dataset: str
    table: str
    mode: str
    columns: tuple[str, ...]
    select_exprs: tuple[str, ...]
    default_env: str
    allow_empty: bool = False
    parquet_glob: str = "*.parquet"


TABLE_SPECS: dict[str, TableSpec] = {
    "stg_posts_core": TableSpec(
        dataset="stg_posts_core",
        table="stg_posts_core",
        mode="full_refresh",
        default_env="HDFS_STG_POSTS_CORE",
        parquet_glob="*/*.parquet",
        columns=(
            "post_id",
            "source",
            "author_id",
            "author_name",
            "title",
            "body",
            "segmented_text",
            "parent_id",
            "reaction_count",
            "comment_count",
            "view_count",
            "created_at",
            "crawled_at",
        ),
        select_exprs=(
            "post_id",
            "extract(_path, 'source=([^/]+)/') AS source",
            "CAST(NULL, 'Nullable(String)') AS author_id",
            "ifNull(author, '') AS author_name",
            "title",
            "ifNull(body, '') AS body",
            "ifNull(segmented_text, '') AS segmented_text",
            "parent_id",
            "toNullable(toInt32(reaction_count)) AS reaction_count",
            "toNullable(toInt32(comment_count)) AS comment_count",
            "toNullable(toInt32(view_count)) AS view_count",
            "created_at",
            "crawled_at",
        ),
    ),
    "stg_posts_nlp": TableSpec(
        dataset="stg_posts_nlp",
        table="stg_posts_nlp",
        mode="replace_latest",
        default_env="HDFS_STG_POSTS_NLP",
        columns=("post_id", "sentiment_label", "sentiment_score", "model_version", "predicted_at"),
        select_exprs=(
            "post_id",
            "sentiment_label",
            "toFloat32(sentiment_score) AS sentiment_score",
            "if(model_version != '', model_version, extract(_path, 'model_version=([^/]+)/')) AS model_version",
            "predicted_at",
        ),
    ),
    "stg_post_topics": TableSpec(
        dataset="stg_post_topics",
        table="stg_post_topics",
        mode="replace_latest",
        default_env="HDFS_STG_POST_TOPICS",
        columns=("post_id", "topic_id", "topic_probability", "model_type", "predicted_at"),
        select_exprs=(
            "post_id",
            "toInt32(topic_id) AS topic_id",
            "toFloat32(topic_probability) AS topic_probability",
            "if(model_type != '', model_type, extract(_path, 'model_type=([^/]+)/')) AS model_type",
            "predicted_at",
        ),
    ),
    "stg_topics": TableSpec(
        dataset="stg_topics",
        table="stg_topics",
        mode="replace_latest",
        default_env="HDFS_STG_TOPICS",
        columns=("topic_id", "label", "top_keywords", "coherence_score", "model_version", "created_at"),
        select_exprs=(
            "toInt32(topic_id) AS topic_id",
            "label",
            "top_keywords",
            "toNullable(toFloat32(coherence_score)) AS coherence_score",
            "model_version",
            "created_at",
        ),
    ),
    "stg_keyword_freq": TableSpec(
        dataset="stg_keyword_freq",
        table="stg_keyword_freq",
        mode="append",
        default_env="HDFS_STG_KEYWORD_FREQ",
        parquet_glob="*/*.parquet",
        columns=("keyword", "window_start", "window_end", "estimated_count", "source"),
        select_exprs=(
            "keyword",
            "window_start",
            "window_end",
            "toInt64(estimated_count) AS estimated_count",
            "source",
        ),
    ),
    "stg_crisis_events": TableSpec(
        dataset="stg_crisis_events",
        table="stg_crisis_events",
        mode="append",
        default_env="HDFS_STG_CRISIS_EVENTS",
        allow_empty=True,
        parquet_glob="*/*.parquet",
        columns=(
            "event_id",
            "detected_at",
            "severity",
            "anomaly_score",
            "trigger_conditions",
            "affected_topics",
            "neg_ratio",
            "mention_velocity",
            "evidence_post_ids",
        ),
        select_exprs=(
            "event_id",
            "detected_at",
            "severity",
            "toFloat64(anomaly_score) AS anomaly_score",
            "trigger_conditions",
            "affected_topics",
            "toFloat32(neg_ratio) AS neg_ratio",
            "toFloat32(mention_velocity) AS mention_velocity",
            "evidence_post_ids",
        ),
    ),
}


def _clickhouse_url() -> str:
    return (
        f"http://{CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}/"
        f"?database={urllib.parse.quote(CLICKHOUSE_DB)}"
        f"&user={urllib.parse.quote(CLICKHOUSE_USER)}"
        f"&password={urllib.parse.quote(CLICKHOUSE_PASSWORD)}"
    )


def execute_sql(sql: str) -> str:
    request = urllib.request.Request(_clickhouse_url(), data=sql.encode("utf-8"), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = response.read().decode("utf-8", errors="replace")
            if response.status >= 400:
                raise RuntimeError(f"ClickHouse HTTP {response.status}: {body[:500]}")
            return body
    except Exception as exc:
        raise RuntimeError(f"ClickHouse query failed: {exc}\nSQL:\n{sql}") from exc


def dataset_path(spec: TableSpec) -> str:
    return _env(spec.default_env, f"{HDFS_STAGED_ROOT}/{spec.dataset}").rstrip("/")


def hdfs_glob(path: str, spec: TableSpec) -> str:
    return f"{path}/{spec.parquet_glob}"


def _hdfs_path(path: str) -> str:
    parsed = urllib.parse.urlparse(path)
    return parsed.path or path


def _webhdfs_liststatus(hdfs_path: str) -> list[dict]:
    url = (
        f"http://{WEBHDFS_HOST}/webhdfs/v1{hdfs_path}"
        f"?op=LISTSTATUS&user.name={urllib.parse.quote(HDFS_USER)}"
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        import json

        body = response.read().decode("utf-8")
        return json.loads(body).get("FileStatuses", {}).get("FileStatus", [])


def hdfs_has_parquet(path: str) -> bool:
    root = _hdfs_path(path).rstrip("/")

    def _walk(hdfs_dir: str) -> bool:
        statuses = _webhdfs_liststatus(hdfs_dir)
        for status in statuses:
            suffix = status.get("pathSuffix", "")
            child = f"{hdfs_dir.rstrip('/')}/{suffix}"
            if status.get("type") == "FILE" and suffix.endswith(".parquet"):
                return True
            if status.get("type") == "DIRECTORY" and _walk(child):
                return True
        return False

    try:
        return _walk(root)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def smoke_test(path: str, spec: TableSpec) -> int:
    sql = (
        "SELECT count()\n"
        f"FROM hdfs('{hdfs_glob(path, spec)}', 'Parquet')"
    )
    body = execute_sql(sql).strip()
    return int(body or "0")


def ingest_dataset_to_clickhouse(dataset_name: str, mode_override: str | None = None) -> None:
    if dataset_name not in TABLE_SPECS:
        allowed = ", ".join(sorted(TABLE_SPECS))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Allowed: {allowed}")

    spec = TABLE_SPECS[dataset_name]
    mode = mode_override or spec.mode
    path = dataset_path(spec)

    print(f"[ingest] dataset={spec.dataset}")
    print(f"[ingest] table={CLICKHOUSE_DB}.{spec.table}")
    print(f"[ingest] mode={mode}")
    print(f"[ingest] hdfs={path}")

    if spec.allow_empty and not hdfs_has_parquet(path):
        print(f"[ingest] optional dataset has no parquet files yet; skip: {path}")
        return

    rows = smoke_test(path, spec)
    print(f"[ingest] hdfs rows visible to ClickHouse: {rows:,}")
    if rows == 0:
        if spec.allow_empty:
            print(f"[ingest] optional dataset has 0 rows; skip insert: {spec.dataset}")
            return
        raise RuntimeError(f"No parquet rows found for dataset {spec.dataset}: {hdfs_glob(path, spec)}")

    if mode == "full_refresh":
        execute_sql(f"TRUNCATE TABLE IF EXISTS {CLICKHOUSE_DB}.{spec.table}")
    elif mode not in {"replace_latest", "append"}:
        raise ValueError(f"Unsupported ingest mode: {mode}")

    columns = ", ".join(spec.columns)
    select_exprs = ",\n    ".join(spec.select_exprs)
    sql = (
        f"INSERT INTO {CLICKHOUSE_DB}.{spec.table} ({columns})\n"
        f"SELECT\n    {select_exprs}\n"
        f"FROM hdfs('{hdfs_glob(path, spec)}', 'Parquet')"
    )
    execute_sql(sql)

    count_body = execute_sql(f"SELECT count() FROM {CLICKHOUSE_DB}.{spec.table}").strip()
    print(f"[ingest] ClickHouse table count: {count_body}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest staged HDFS Parquet datasets into ClickHouse.")
    parser.add_argument(
        "dataset",
        choices=sorted(TABLE_SPECS),
        help="Dataset/table contract name to ingest.",
    )
    parser.add_argument(
        "--mode",
        choices=["full_refresh", "replace_latest", "append"],
        default=None,
        help="Override the default ingest mode for the dataset.",
    )
    args = parser.parse_args()
    ingest_dataset_to_clickhouse(args.dataset, args.mode)


if __name__ == "__main__":
    main()
