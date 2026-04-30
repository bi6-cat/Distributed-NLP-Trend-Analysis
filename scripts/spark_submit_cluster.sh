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

# ── Cấu hình cluster — M2 điền vào ──────────────────────────────────────────
SPARK_MASTER="${SPARK_MASTER_URL:-spark://CLUSTER_MASTER:7077}"
HDFS_BASE="${HDFS_NAMENODE:-hdfs://CLUSTER_MASTER:9000}"
WORKER_PYTHON="${WORKER_PYTHON_PATH:-/opt/conda/envs/nlp-trend/bin/python}"

# ── Cấu hình job ─────────────────────────────────────────────────────────────
NUM_EXECUTORS="${NUM_EXECUTORS:-3}"
EXECUTOR_CORES="${EXECUTOR_CORES:-4}"
EXECUTOR_MEMORY="${EXECUTOR_MEMORY:-6g}"
DRIVER_MEMORY="${DRIVER_MEMORY:-4g}"

# ── Đường dẫn local (relative từ project root) ───────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ZIP_PATH="$PROJECT_ROOT/dist/nlp_trend.zip"

# ── Đường dẫn HDFS ───────────────────────────────────────────────────────────
HDFS_MODEL="$HDFS_BASE/models/phobert_finetuned/final"
HDFS_SLANG="$HDFS_BASE/data/ref/slang_dict.json"
HDFS_STOPWORDS="$HDFS_BASE/data/ref/stopwords_vi.txt"
HDFS_INPUT="$HDFS_BASE/data/raw/voz/date=*/*.json"
HDFS_OUTPUT="$HDFS_BASE/data/staged/voz_sentiment/"

# =============================================================================
# Hàm tiện ích
# =============================================================================

log() { echo "[$(date '+%H:%M:%S')] $*"; }

upload_to_hdfs() {
    log "Upload model lên HDFS..."
    hdfs dfs -mkdir -p "$HDFS_BASE/models/phobert_finetuned"
    hdfs dfs -put -f "$PROJECT_ROOT/models/phobert_finetuned/final" \
        "$HDFS_BASE/models/phobert_finetuned/"

    log "Upload data files lên HDFS..."
    hdfs dfs -mkdir -p "$HDFS_BASE/data/ref"
    hdfs dfs -put -f "$PROJECT_ROOT/data/slang_dict.json"   "$HDFS_SLANG"
    hdfs dfs -put -f "$PROJECT_ROOT/data/stopwords_vi.txt"  "$HDFS_STOPWORDS"

    log "Upload hoàn tất."
    hdfs dfs -ls "$HDFS_BASE/models/phobert_finetuned/final"
}

build_zip() {
    log "Đóng gói Python code → $ZIP_PATH"
    mkdir -p "$PROJECT_ROOT/dist"
    cd "$PROJECT_ROOT"
    zip -r "$ZIP_PATH" preprocessing/ spark_jobs/ models/ \
        -x "**/__pycache__/*" -x "**/*.pyc" -x "models/phobert_finetuned/*"
    log "Zip size: $(du -sh "$ZIP_PATH" | cut -f1)"
}

submit_job() {
    log "Submit Spark job..."
    log "  Master       : $SPARK_MASTER"
    log "  Executors    : $NUM_EXECUTORS x $EXECUTOR_CORES cores x $EXECUTOR_MEMORY"
    log "  HDFS input   : $HDFS_INPUT"
    log "  HDFS output  : $HDFS_OUTPUT"

    spark-submit \
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
        --conf "spark.executorEnv.HDFS_INPUT=$HDFS_INPUT" \
        --conf "spark.executorEnv.HDFS_OUTPUT=$HDFS_OUTPUT" \
        \
        "$PROJECT_ROOT/spark_jobs/preprocess_job.py"
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
