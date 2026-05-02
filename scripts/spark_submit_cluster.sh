#!/usr/bin/env bash
# =============================================================================
# spark_submit_cluster.sh — Submit NLP Sentiment Job lên Spark cluster
#
# Yêu cầu M2 cung cấp trước:
#   - SPARK_MASTER_URL   : địa chỉ Spark master (vd: spark://192.168.1.10:7077)
#   - HDFS_NAMENODE      : địa chỉ HDFS NameNode (vd: hdfs://192.168.1.10:9000)
#   - Python path trên worker nodes (mặc định: /opt/conda/envs/nlp-trend/bin/python)
#
# Cách dùng:
#   # Bước 1: Upload model + data files lên HDFS (chạy 1 lần)
#   bash scripts/spark_submit_cluster.sh --upload-only
#
#   # Bước 2: Đóng gói Python code thành zip, submit job
#   bash scripts/spark_submit_cluster.sh
#
#   # Chạy với số executor tuỳ chỉnh
#   bash scripts/spark_submit_cluster.sh --executors 3
# =============================================================================

set -euo pipefail

export HADOOP_USER_NAME=zett

# ── Cấu hình cluster — điền từ docs/CLUSTER_INFO.md ─────────────────────────
# Master Node: 192.168.56.11 (Spark Master + HDFS NameNode)
# Storage Node: 192.168.56.14 (ClickHouse)
SPARK_MASTER="${SPARK_MASTER_URL:-spark://192.168.56.11:7077}"
HDFS_BASE="${HDFS_NAMENODE:-hdfs://192.168.56.11:9000}"
WORKER_PYTHON="${WORKER_PYTHON_PATH:-/opt/miniconda/envs/nlp-trend/bin/python}"
CLICKHOUSE_HOST="${CLICKHOUSE_HOST:-192.168.56.14}"

# ── Cấu hình job ─────────────────────────────────────────────────────────────
NUM_EXECUTORS="${NUM_EXECUTORS:-2}"
EXECUTOR_CORES="${EXECUTOR_CORES:-5}"
EXECUTOR_MEMORY="${EXECUTOR_MEMORY:-3g}"
DRIVER_MEMORY="${DRIVER_MEMORY:-2g}"

# ── Đường dẫn local (relative từ project root) ───────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ZIP_PATH="$PROJECT_ROOT/dist/nlp_trend.zip"

# ── Đường dẫn HDFS ───────────────────────────────────────────────────────────
# Dùng /user/zett/ — khớp với upload_to_hdfs.py (HDFS_USER=zett)
HDFS_RAW="$HDFS_BASE/user/zett/raw_data"         # nơi M1 đã upload CSV
HDFS_STAGED="$HDFS_BASE/user/zett/staged"         # output sau khi xử lý
HDFS_MODEL="$HDFS_BASE/user/zett/models/phobert_finetuned/final"
HDFS_SLANG="$HDFS_BASE/user/zett/ref/slang_dict.json"
HDFS_STOPWORDS="$HDFS_BASE/user/zett/ref/stopwords_vi.txt"
# Đọc cả 3 nguồn CSV — M1 upload dạng CSV (không phải JSON, không partition by date)
HDFS_INPUT_VOZ="$HDFS_RAW/voz"
HDFS_INPUT_VATVO="$HDFS_RAW/vatvo"
HDFS_INPUT_VNE="$HDFS_RAW/vnexpress"
HDFS_OUTPUT="$HDFS_STAGED/sentiment/"

# =============================================================================
# Hàm tiện ích
# =============================================================================

log() { echo "[$(date '+%H:%M:%S')] $*"; }

upload_to_hdfs() {
    log "Upload model lên HDFS..."
    /opt/hadoop/bin/hdfs dfs -mkdir -p "$HDFS_MODEL"
    # /opt/hadoop/bin/hdfs dfs -put -f "$PROJECT_ROOT/models/phobert_finetuned/final" \
    #     "$(dirname "$HDFS_MODEL")/"

    log "Upload data files (slang + stopwords) lên HDFS..."
    /opt/hadoop/bin/hdfs dfs -mkdir -p "$(dirname "$HDFS_SLANG")"
    /opt/hadoop/bin/hdfs dfs -put -f "$PROJECT_ROOT/data/slang_dict.json"   "$HDFS_SLANG"
    /opt/hadoop/bin/hdfs dfs -put -f "$PROJECT_ROOT/data/stopwords_vi.txt"  "$HDFS_STOPWORDS"

    log "Upload hoàn tất."
    /opt/hadoop/bin/hdfs dfs -ls "$HDFS_MODEL" || true
}

build_zip() {
    log "Đóng gói Python code → $ZIP_PATH"
    mkdir -p "$PROJECT_ROOT/dist"
    cd "$PROJECT_ROOT"
    # Thêm schemas/ — chứa VozAdapter, VatVoAdapter, VnExpressAdapter, models.py
    zip -r "$ZIP_PATH" preprocessing/ schemas/ spark_jobs/ models/ \
        -x "**/__pycache__/*" -x "**/*.pyc" -x "models/phobert_finetuned/*"
    log "Zip size: $(du -sh "$ZIP_PATH" | cut -f1)"
}

submit_job() {
    log "Submit Spark job..."
    log "  Master       : $SPARK_MASTER"
    log "  Executors    : $NUM_EXECUTORS x $EXECUTOR_CORES cores x $EXECUTOR_MEMORY"
    log "  HDFS VOZ     : $HDFS_INPUT_VOZ"
    log "  HDFS VatVo   : $HDFS_INPUT_VATVO"
    log "  HDFS VnE     : $HDFS_INPUT_VNE"
    log "  HDFS output  : $HDFS_OUTPUT"
    log "  ClickHouse   : $CLICKHOUSE_HOST:8123"

    /opt/spark/bin/spark-submit \
        --master "$SPARK_MASTER" \
        --deploy-mode client \
        --num-executors "$NUM_EXECUTORS" \
        --executor-cores "$EXECUTOR_CORES" \
        --executor-memory "$EXECUTOR_MEMORY" \
        --driver-memory  "$DRIVER_MEMORY" \
        \
        --conf "spark.pyspark.python=$WORKER_PYTHON" \
        --conf "spark.pyspark.driver.python=$WORKER_PYTHON" \
        --conf "spark.executorEnv.PYSPARK_PYTHON=$WORKER_PYTHON" \
        --conf "spark.python.worker.faulthandler.enabled=true" \
        --conf "spark.sql.shuffle.partitions=200" \
        --conf "spark.serializer=org.apache.spark.serializer.KryoSerializer" \
        \
        --py-files "$ZIP_PATH" \
        \
        --conf "spark.executorEnv.NLP_MODEL_PATH=$HDFS_MODEL" \
        --conf "spark.executorEnv.NLP_SLANG_DICT=$HDFS_SLANG" \
        --conf "spark.executorEnv.NLP_STOPWORDS=$HDFS_STOPWORDS" \
        --conf "spark.executorEnv.HDFS_INPUT=$HDFS_INPUT_VOZ" \
        --conf "spark.executorEnv.HDFS_OUTPUT=$HDFS_OUTPUT" \
        --conf "spark.executorEnv.CLICKHOUSE_HOST=$CLICKHOUSE_HOST" \
        --conf "spark.executorEnv.CLICKHOUSE_PORT=8123" \
        --conf "spark.executorEnv.CLICKHOUSE_DB=tech_radar" \
        --conf "spark.executorEnv.CLICKHOUSE_USER=default" \
        \
        "$PROJECT_ROOT/spark_jobs/cleaning_job.py"
}

# =============================================================================
# Main
# =============================================================================

UPLOAD_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --upload-only) UPLOAD_ONLY=true ;;
        --executors=*) NUM_EXECUTORS="${arg#*=}" ;;
    esac
done

if [ "$UPLOAD_ONLY" = true ]; then
    upload_to_hdfs
    exit 0
fi

build_zip
upload_to_hdfs
submit_job
