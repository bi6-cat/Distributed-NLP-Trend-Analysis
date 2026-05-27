#!/bin/bash
set -euo pipefail

# Wrapper for the standardized HDFS -> ClickHouse ingest contract.
# Usage:
#   scripts/ingest_hdfs_to_clickhouse.sh stg_posts_core
#   scripts/ingest_hdfs_to_clickhouse.sh stg_keyword_freq --mode append

DATASET="${1:-stg_posts_core}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "[INFO] Ingest dataset: $DATASET"
echo "[INFO] CLICKHOUSE_HOST=${CLICKHOUSE_HOST:-clickhouse}"
echo "[INFO] CLICKHOUSE_DB=${CLICKHOUSE_DB:-tech_radar}"
echo "[INFO] HDFS_STAGED_ROOT=${HDFS_STAGED_ROOT:-hdfs://namenode:9000/user/${HDFS_USER:-${HADOOP_USER_NAME:-root}}/staged}"

python "$PROJECT_ROOT/scripts/hdfs_to_clickhouse.py" "$DATASET" "$@"
