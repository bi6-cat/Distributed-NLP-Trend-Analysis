"""
Task 3.1 — Tích hợp PhoBERT vào Spark qua mapPartitions.

Đọc stg_posts_core từ HDFS Parquet (M2 output) → PhoBERT inference →
ghi kết quả vào ClickHouse table stg_posts_nlp.


Cách chạy trên cluster:
    spark-submit \\
        --master spark://192.168.56.11:7077 \\
        --executor-memory 2g \\
        --total-executor-cores 4 \\
        --py-files dist/nlp_trend.zip \\
        --conf spark.jars=/opt/spark/jars/clickhouse-jdbc.jar \\
        --conf spark.executorEnv.NLP_MODEL_PATH=hdfs://192.168.56.11:9000/user/zett/models/phobert_finetuned/final \\
        --conf spark.executorEnv.NLP_MODEL_VERSION=phobert_v1 \\
        --conf spark.executorEnv.CLICKHOUSE_HOST=192.168.56.14 \\
        spark_jobs/sentiment_job.py

Biến môi trường:
    HDFS_INPUT         : path Parquet stg_posts_core
    NLP_MODEL_PATH     : path tới PhoBERT checkpoint (HDFS hoặc local)
    NLP_MODEL_VERSION  : version string ghi vào model_version
    CLICKHOUSE_HOST    : IP storage node
    CLICKHOUSE_PORT    : HTTP port ClickHouse      (mặc định 8123)
    CLICKHOUSE_DB      : database name             (mặc định tech_radar)
    CLICKHOUSE_USER    : user                      (mặc định default)
    CLICKHOUSE_PASS    : password                  (mặc định '')
    HDFS_COPY_TIMEOUT  : timeout copy model (giây) (mặc định 300)
    LIMIT_SAMPLES      : giới hạn số record test   (mặc định None)
"""

import os
import shutil
import threading
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import (
    StructType, StructField,
    StringType, FloatType, TimestampType,
)

HDFS_BASE     = os.environ.get("HDFS_BASE", "hdfs://namenode:9000")
HDFS_USER     = os.environ.get("HDFS_USER", os.environ.get("HADOOP_USER_NAME", "root"))
HDFS_HOME     = os.environ.get("HDFS_HOME", f"/user/{HDFS_USER}")
HDFS_INPUT    = os.environ.get("HDFS_INPUT",        f"{HDFS_BASE}{HDFS_HOME}/staged/stg_posts_core")
MODEL_PATH    = os.environ.get("NLP_MODEL_PATH",    f"{HDFS_BASE}{HDFS_HOME}/models/phobert_finetuned/final")
MODEL_VERSION = os.environ.get("NLP_MODEL_VERSION", "phobert_v1")

CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST", "192.168.56.14")
CLICKHOUSE_PORT = os.environ.get("CLICKHOUSE_PORT", "8123")
CLICKHOUSE_DB   = os.environ.get("CLICKHOUSE_DB",   "tech_radar")
CLICKHOUSE_USER = os.environ.get("CLICKHOUSE_USER", "root")
CLICKHOUSE_PASS = os.environ.get("CLICKHOUSE_PASS", "root")

HDFS_COPY_TIMEOUT = int(os.environ.get("HDFS_COPY_TIMEOUT", "300"))   # FIX ⑤
LIMIT_SAMPLES     = os.environ.get("LIMIT_SAMPLES", None)
SENTIMENT_BATCH_SIZE = int(os.environ.get("SENTIMENT_BATCH_SIZE", "64"))
SENTIMENT_MAX_LENGTH = int(os.environ.get("SENTIMENT_MAX_LENGTH", "256"))
SENTIMENT_TORCH_THREADS = int(os.environ.get("SENTIMENT_TORCH_THREADS", "1"))


def log_progress(percent: int, message: str) -> None:
    print(f"[PROGRESS] {percent:>3}% | {message}", flush=True)


def _format_eta(seconds: float) -> str:
    """Format thời gian còn lại thành chuỗi ngắn gọn cho log."""
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _count_with_progress_monitor(spark, df, progress_start=35, progress_end=75):
    """
    Chạy count() trên background thread và poll Spark StatusTracker để in tiến độ.

    Lý do:
    - Log bên executor thường không stream đều về Airflow.
    - Driver vẫn nhìn thấy stage/task progress qua StatusTracker.
    """
    sc = spark.sparkContext
    tracker = sc.statusTracker()
    job_group = f"sentiment-inference-{int(time.time())}"
    poll_interval = 5
    result = {"count": None, "error": None}

    def _run_count():
        try:
            sc.setJobGroup(job_group, "PhoBERT inference count")
            result["count"] = df.count()
        except Exception as exc:  # pragma: no cover - surfaced to caller
            result["error"] = exc

    worker = threading.Thread(target=_run_count, daemon=True)
    worker.start()

    started_at = time.time()
    last_progress = -1

    while worker.is_alive():
        active_stage_ids = list(tracker.getActiveStageIds())
        total_tasks = 0
        completed_tasks = 0
        active_tasks = 0

        for stage_id in active_stage_ids:
            info = tracker.getStageInfo(stage_id)
            if not info:
                continue
            total_tasks += int(getattr(info, "numTasks", 0) or 0)
            completed_tasks += int(getattr(info, "numCompletedTasks", 0) or 0)
            active_tasks += int(getattr(info, "numActiveTasks", 0) or 0)

        if total_tasks > 0:
            stage_ratio = min(max(completed_tasks / total_tasks, 0.0), 1.0)
            progress = progress_start + (progress_end - progress_start) * stage_ratio
            progress_int = int(progress)

            if progress_int != last_progress:
                if completed_tasks == 0:
                    print(
                        f"[PROGRESS] {progress:>5.1f}% | PhoBERT inference tasks "
                        f"{completed_tasks}/{total_tasks} | active {active_tasks} | "
                        "warming up workers, ETA chưa xác định",
                        flush=True,
                    )
                else:
                    elapsed = max(time.time() - started_at, 1e-6)
                    eta_seconds = elapsed / stage_ratio - elapsed
                    print(
                        f"[PROGRESS] {progress:>5.1f}% | PhoBERT inference tasks "
                        f"{completed_tasks}/{total_tasks} | active {active_tasks} | "
                        f"ETA {_format_eta(eta_seconds)}",
                        flush=True,
                    )
                last_progress = progress_int
        else:
            elapsed = time.time() - started_at
            if int(elapsed) // poll_interval != int(max(elapsed - poll_interval, 0)) // poll_interval:
                print(
                    "[PROGRESS]  35% | PhoBERT inference submitted, waiting for Spark stage metrics...",
                    flush=True,
                )

        worker.join(timeout=poll_interval)

    if result["error"] is not None:
        raise result["error"]
    return result["count"]


# Schema đầu ra — khớp với ClickHouse table stg_posts_nlp
OUTPUT_SCHEMA = StructType([
    StructField("post_id",         StringType(),    False),
    StructField("sentiment_label", StringType(),    True),
    StructField("sentiment_score", FloatType(),     True),
    StructField("model_version",   StringType(),    True),
    StructField("predicted_at",    TimestampType(), True),
])

# ---------------------------------------------------------------------------
# FIX ②: Global model cache — sống xuyên suốt executor process
#
# mapPartitions gọi process_partition nhiều lần trên cùng 1 executor process.
# Dict này tồn tại ở process level → model chỉ load 1 lần duy nhất mỗi executor,
# bất kể có bao nhiêu partition được xử lý.
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}
_RESOLVED_MODEL_PATH: dict = {}   # cache kaggle/hdfs → local path, per executor process


def _is_valid_model_dir(path: str) -> bool:
    """Kiểm tra thư mục model có đủ file tối thiểu để load."""
    if not path or not os.path.isdir(path):
        return False

    required_any = [
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
    ]
    required_all = [
        "config.json",
        "tokenizer_config.json",
    ]

    for fname in required_all:
        if not os.path.exists(os.path.join(path, fname)):
            return False

    if not any(os.path.exists(os.path.join(path, fname)) for fname in required_any):
        return False

    return True


def _prepare_kaggle_model(handle: str) -> str:
    """
    Tải model từ Kaggle vào thư mục local ổn định với file lock.

    Lý do:
    - cache của kagglehub có thể bị nhiều python worker đụng cùng lúc
    - worker khác có thể đọc trúng file safetensors chưa tải/copy xong
    """
    import fcntl
    import tempfile
    from pathlib import Path

    safe_name = handle.replace("/", "__").replace(":", "__")
    base_dir = os.path.join(tempfile.gettempdir(), "kaggle_models")
    final_dir = os.path.join(base_dir, safe_name)
    tmp_dir = final_dir + ".tmp"
    lock_path = final_dir + ".lock"

    os.makedirs(base_dir, exist_ok=True)

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if _is_valid_model_dir(final_dir):
                print(f"[MODEL] Reusing prepared Kaggle model: {final_dir}", flush=True)
                return final_dir

            shutil.rmtree(final_dir, ignore_errors=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)

            os.environ["HOME"] = tempfile.gettempdir()
            os.environ["KAGGLE_CONFIG_DIR"] = tempfile.gettempdir()
            os.environ["KAGGLEHUB_CACHE"] = os.path.join(tempfile.gettempdir(), "kaggle_cache")

            import kagglehub
            print(f"[MODEL] Downloading from Kaggle: {handle}", flush=True)
            downloaded_base = kagglehub.model_download(handle)

            source_dir = None
            for root_dir, _, files in os.walk(downloaded_base):
                if "config.json" in files and (
                    "model.safetensors" in files
                    or "pytorch_model.bin" in files
                    or "model.safetensors.index.json" in files
                ):
                    source_dir = root_dir
                    break

            if source_dir is None:
                raise RuntimeError(f"Cannot locate usable model files under {downloaded_base}")

            shutil.copytree(source_dir, tmp_dir)
            if not _is_valid_model_dir(tmp_dir):
                raise RuntimeError(f"Downloaded model is incomplete at {tmp_dir}")

            Path(tmp_dir).replace(final_dir)
            print(f"[MODEL] Prepared Kaggle model at: {final_dir}", flush=True)
            return final_dir
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _stage_local_model(source_path: str, local_name: str) -> str:
    """
    Copy model local từ bind mount sang tmp nội bộ container để tránh I/O chậm
    trên filesystem Windows/WSL khi load safetensors lớn.
    """
    import fcntl
    import tempfile

    if not _is_valid_model_dir(source_path):
        raise RuntimeError(f"Local model path không hợp lệ: {source_path}")

    normalized_source = os.path.abspath(source_path)
    temp_root = tempfile.gettempdir()
    if normalized_source.startswith(temp_root):
        return normalized_source

    local = os.path.join(temp_root, local_name)
    lock_path = local + ".lock"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if _is_valid_model_dir(local):
                print(f"[MODEL] Reusing staged local model: {local}", flush=True)
                return local

            shutil.rmtree(local, ignore_errors=True)
            print(f"[MODEL] Staging local model {source_path} -> {local}", flush=True)
            shutil.copytree(source_path, local)

            if not _is_valid_model_dir(local):
                raise RuntimeError(f"Staged local model is incomplete at {local}")

            return local
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _hdfs_to_local(hdfs_path: str, local_name: str) -> str:
    """
    Copy model từ HDFS về local temp dir của executor.

    FIX ④: Dùng file lock (lockfile) để tránh race condition khi
    executor-cores > 1 — nhiều thread cùng check và copy đồng thời.

    FIX ⑤: Timeout cấu hình qua HDFS_COPY_TIMEOUT (default 300s).
    """
    if not hdfs_path.startswith("hdfs://"):
        print(f"[HDFS] Path là local: {hdfs_path}", flush=True)
        return hdfs_path

    import fcntl
    import shutil
    import subprocess
    import tempfile

    local     = os.path.join(tempfile.gettempdir(), local_name)
    lock_path = local + ".lock"

    # Lock để serialize giữa các thread trong cùng executor process
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)   # blocking exclusive lock

        try:
            # Double-check sau khi có lock: thread khác có thể đã copy xong
            if os.path.exists(os.path.join(local, "config.json")):
                print(f"[HDFS] Reusing cached model: {local}", flush=True)
                return local

            # Xoá directory cũ nếu không đầy đủ
            if os.path.exists(local):
                print(f"[HDFS] Xoá directory cũ: {local}", flush=True)
                shutil.rmtree(local, ignore_errors=True)

            hdfs_bin = "/opt/hadoop/bin/hdfs"
            print(f"[HDFS] Copy {hdfs_path} → {local} (timeout={HDFS_COPY_TIMEOUT}s)", flush=True)

            result = subprocess.run(
                [hdfs_bin, "dfs", "-get", hdfs_path, local],
                capture_output=True,
                text=True,
                timeout=HDFS_COPY_TIMEOUT,   # FIX ⑤
            )

            if result.returncode != 0:
                print(f"[HDFS STDOUT] {result.stdout}", flush=True)
                print(f"[HDFS STDERR] {result.stderr}", flush=True)
                raise RuntimeError(
                    f"HDFS copy thất bại (code {result.returncode}): {result.stderr}"
                )

            # Kiểm tra file bắt buộc
            required_files = ["config.json", "tokenizer_config.json", "vocab.txt", "bpe.codes"]
            for fname in required_files:
                fpath = os.path.join(local, fname)
                if not os.path.exists(fpath):
                    raise RuntimeError(f"Thiếu file model: {fpath}")

            print(f"[HDFS] Copy hoàn tất, files verified ✓", flush=True)

        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    return local


def _get_model(model_path: str, device):
    """
    FIX ②: Load model 1 lần duy nhất mỗi executor process.

    Lần đầu gọi: load từ disk → lưu vào _MODEL_CACHE.
    Các lần sau: trả về từ cache ngay lập tức (không đọc disk).
    """
    global _MODEL_CACHE

    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]

    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    print(f"[MODEL] Loading từ {model_path} ...", flush=True)
    t0 = time.time()

    try:
        config    = AutoConfig.from_pretrained(model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            config=config,
            local_files_only=True,
            use_fast=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            local_files_only=True,
        )
    except Exception:
        # Nếu cache local đang hỏng thì lần sau caller có thể tái tạo lại thư mục model.
        _MODEL_CACHE.pop(model_path, None)
        raise
    model.to(device)
    model.eval()

    _MODEL_CACHE[model_path] = (tokenizer, model)
    print(f"[MODEL] Loaded ✓  ({time.time() - t0:.1f}s)", flush=True)

    return tokenizer, model


def _chunked(iterator, size: int):
    """
    FIX ③: Yield từng batch nhỏ từ iterator mà không materialize toàn bộ.

    Thay thế list(iterator) — tránh load toàn partition vào RAM cùng lúc.
    """
    import itertools
    it = iter(iterator)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


# ---------------------------------------------------------------------------
# Inference worker — chạy trên executor
# ---------------------------------------------------------------------------

def process_partition(iterator, partition_id=None):
    """
    Chạy trên mỗi Spark executor (lazy init).

    Đầu vào : Row từ stg_posts_core (segmented_text từ M2)
    Đầu ra  : tuple theo OUTPUT_SCHEMA (5 cột)

    Cải tiến:
      - Model load 1 lần / executor process (FIX ②)
      - Stream qua iterator theo batch, không load hết RAM (FIX ③)
    """
    import os
    import sys
    import traceback
    from datetime import datetime, timezone

    # Offline mode: không gọi HuggingFace Hub
    os.environ["HF_HUB_OFFLINE"]      = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        from pyspark import SparkFiles
        root = SparkFiles.getRootDirectory()
        if root not in sys.path:
            sys.path.insert(0, root)
    except Exception:
        pass

    kaggle_handle = os.environ.get("KAGGLE_MODEL_HANDLE")
    model_path    = os.environ.get("NLP_MODEL_PATH",    "models/phobert_finetuned/final")
    model_version = os.environ.get("NLP_MODEL_VERSION", "phobert_v1")

    if _is_valid_model_dir(model_path):
        model_path = _stage_local_model(model_path, "phobert_finetuned_local")
        print(f"[MODEL] Using local model path: {model_path}", flush=True)
    elif kaggle_handle:
        if kaggle_handle in _RESOLVED_MODEL_PATH:
            model_path = _RESOLVED_MODEL_PATH[kaggle_handle]
            print(f"[MODEL] Reusing cached Kaggle path: {model_path}", flush=True)
        else:
            model_path = _prepare_kaggle_model(kaggle_handle)
            _RESOLVED_MODEL_PATH[kaggle_handle] = model_path
            print(f"[MODEL] Local Kaggle model path: {model_path}", flush=True)
    else:
        # Copy HDFS model về local (với lock, với retry timeout đúng)
        model_path = _hdfs_to_local(model_path, "phobert_finetuned")

    try:
        import torch
    except ImportError as e:
        raise RuntimeError(f"Worker không import được torch: {e}")

    LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
    batch_size = max(1, int(os.environ.get("SENTIMENT_BATCH_SIZE", SENTIMENT_BATCH_SIZE)))
    max_length = max(16, int(os.environ.get("SENTIMENT_MAX_LENGTH", SENTIMENT_MAX_LENGTH)))
    torch_threads = max(1, int(os.environ.get("SENTIMENT_TORCH_THREADS", SENTIMENT_TORCH_THREADS)))
    torch.set_num_threads(torch_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer, model = _get_model(model_path, device)
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Không load được model: {e}")

    rows = list(iterator)
    total_rows = len(rows)
    if total_rows == 0:
        return

    partition_label = partition_id if partition_id is not None else "?"
    print(
        f"[PROGRESS]  35% | PhoBERT partition {partition_label} inference start: "
        f"{total_rows:,} records",
        flush=True,
    )

    processed_rows = 0
    next_partition_progress = 25
    partition_started_at = time.time()

    print(
        f"[BENCHMARK] PhoBERT partition {partition_label}: "
        f"batch_size={batch_size}, max_length={max_length}, torch_threads={torch_threads}",
        flush=True,
    )

    for batch in _chunked(rows, batch_size):

        texts = []
        for row in batch:
            text = getattr(row, "segmented_text", None) or getattr(row, "body", None) or ""
            texts.append(text if isinstance(text, str) else str(text))

        encoding = tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            ).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)

        predicted_at = datetime.now(timezone.utc).replace(tzinfo=None)

        for i, row in enumerate(batch):
            yield (
                str(getattr(row, "post_id", "") or ""),
                LABEL_MAP[int(preds[i])],
                float(probs[i][int(preds[i])]),
                model_version,
                predicted_at,
            )

        processed_rows += len(batch)
        partition_progress = int(processed_rows * 100 / total_rows)
        if partition_progress >= next_partition_progress or processed_rows == total_rows:
            elapsed = max(time.time() - partition_started_at, 1e-6)
            throughput = processed_rows / elapsed
            remaining_rows = max(total_rows - processed_rows, 0)
            eta_seconds = remaining_rows / throughput if throughput > 0 else 0.0
            print(
                f"[PROGRESS]  35%-75% | PhoBERT partition {partition_label}: "
                f"{partition_progress:>3}% ({processed_rows:,}/{total_rows:,}) | "
                f"{throughput:.1f} rec/s | ETA { _format_eta(eta_seconds) }",
                flush=True,
            )
            while next_partition_progress <= partition_progress:
                next_partition_progress += 25


# ---------------------------------------------------------------------------
# FIX ①: Ghi ClickHouse qua foreachPartition — không collect() về driver
# FIX ⑦: Retry 3 lần với exponential backoff, check HTTP status
# ---------------------------------------------------------------------------

def _write_partition_to_clickhouse(rows, host, port, db, user, password, table):
    """
    Ghi 1 partition vào ClickHouse qua HTTP API (TabSeparated).
    Chạy trực tiếp trên executor — không kéo dữ liệu về driver.

    FIX ⑦: Retry với backoff khi gặp lỗi mạng hoặc 5xx.
    """
    import time
    import urllib.parse
    import urllib.request

    MAX_RETRY    = 3
    RETRY_BACKOFF = [2, 5, 10]   # giây chờ giữa các lần retry
    WRITE_BATCH  = 1000          # số dòng mỗi lần INSERT

    def _insert(lines: list[str]) -> None:
        body  = "\n".join(lines).encode("utf-8")
        query = (
            f"INSERT INTO {db}.{table} "
            f"(post_id, sentiment_label, sentiment_score, model_version, predicted_at) "
            f"FORMAT TabSeparated"
        )
        url = (
            f"http://{host}:{port}/"
            f"?query={urllib.parse.quote(query)}"
            f"&user={urllib.parse.quote(user)}"
            f"&password={urllib.parse.quote(password)}"
        )

        for attempt in range(MAX_RETRY):
            try:
                req = urllib.request.Request(url, data=body, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    status = resp.status
                    content = resp.read().decode("utf-8", errors="replace")

                # FIX ⑦: Check HTTP status — urlopen không raise trên 5xx
                if status >= 400:
                    raise RuntimeError(
                        f"ClickHouse trả về HTTP {status}: {content[:200]}"
                    )
                return  # thành công

            except Exception as e:
                if attempt < MAX_RETRY - 1:
                    wait = RETRY_BACKOFF[attempt]
                    print(
                        f"[CH] Lần {attempt+1}/{MAX_RETRY} thất bại: {e}. "
                        f"Thử lại sau {wait}s ...",
                        flush=True,
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"ClickHouse insert thất bại sau {MAX_RETRY} lần: {e}"
                    ) from e

    # Ghi theo batch nhỏ để tránh HTTP request quá lớn
    buffer = []
    total  = 0

    for r in rows:
        predicted_at = (
            r.predicted_at.strftime("%Y-%m-%d %H:%M:%S")
            if r.predicted_at else "1970-01-01 00:00:00"
        )
        buffer.append("\t".join([
            str(r.post_id or ""),
            str(r.sentiment_label or ""),
            str(r.sentiment_score or 0.0),
            str(r.model_version or ""),
            predicted_at,
        ]))

        if len(buffer) >= WRITE_BATCH:
            _insert(buffer)
            total  += len(buffer)
            buffer  = []

    if buffer:
        _insert(buffer)
        total += len(buffer)

    if total > 0:
        print(f"[CH] Partition ghi {total} dòng ✓", flush=True)


def write_to_clickhouse(df, host, port, db, user, password, table="stg_posts_nlp"):
    """
    FIX ①: Ghi DataFrame vào ClickHouse qua foreachPartition.

    Mỗi executor tự ghi partition của mình — driver không giữ dữ liệu.
    Scale tuyến tính với số executor, không bị giới hạn bởi driver memory.
    """
    df.foreachPartition(
        lambda rows: _write_partition_to_clickhouse(
            rows, host, port, db, user, password, table
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log_progress(5, "Starting Spark session")
    spark = (
        SparkSession.builder
        .appName("NLP_SentimentJob")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    log_progress(10, "Reading staged posts")
    print(f"[INFO] Đọc stg_posts_core từ: {HDFS_INPUT}")
    raw_df = spark.read.parquet(HDFS_INPUT)

    if LIMIT_SAMPLES is not None:
        try:
            raw_df = raw_df.limit(int(LIMIT_SAMPLES))
            print(f"[INFO] Giới hạn {LIMIT_SAMPLES} mẫu")
        except ValueError:
            pass

    # Đảm bảo các cột bắt buộc tồn tại
    for col_name in ("post_id", "segmented_text", "body"):
        if col_name not in raw_df.columns:
            raw_df = raw_df.withColumn(col_name, lit(None))

    # ── Inference ────────────────────────────────────────────────────────────

    log_progress(25, "Preparing sentiment inference partitions")
    target_partitions = max(1, spark.sparkContext.defaultParallelism)
    raw_df = raw_df.repartition(target_partitions)
    print(f"[INFO] Repartition sentiment input to {target_partitions} partitions")
    processed_rdd = raw_df.rdd.mapPartitionsWithIndex(
        lambda partition_id, rows: process_partition(rows, partition_id)
    )
    processed_df  = spark.createDataFrame(processed_rdd, OUTPUT_SCHEMA)
    processed_df  = processed_df.cache()

    # FIX ⑥: t_start ngay trước action thật sự (count trigger toàn pipeline)
    n_executors    = spark.sparkContext.defaultParallelism
    executor_mem   = spark.conf.get("spark.executor.memory",  "?")
    executor_cores = spark.conf.get("spark.executor.cores",   "?")

    print(
        f"[BENCHMARK] defaultParallelism={n_executors}, "
        f"executor.memory={executor_mem}, executor.cores={executor_cores}"
    )

    log_progress(35, "Running PhoBERT inference")
    t_infer_start = time.time()
    n_records     = _count_with_progress_monitor(spark, processed_df)
    t_infer_end   = time.time()
    log_progress(75, f"Inference complete: {n_records:,} records")

    infer_elapsed  = t_infer_end - t_infer_start
    infer_throughput = n_records / infer_elapsed if infer_elapsed > 0 else 0.0

    print(f"[BENCHMARK] ===== INFERENCE REPORT =====")
    print(f"[BENCHMARK] Records processed : {n_records:,}")
    print(f"[BENCHMARK] Inference time    : {infer_elapsed:.2f} s")
    print(f"[BENCHMARK] Throughput        : {infer_throughput:.1f} records/s")
    print(f"[BENCHMARK] Parallelism       : {n_executors} slots")
    print(f"[BENCHMARK] ===========================")

    # ── Ghi ClickHouse ───────────────────────────────────────────────────────

    log_progress(80, "Writing sentiment rows to ClickHouse")
    t_write_start = time.time()

    write_to_clickhouse(
        processed_df,
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        db=CLICKHOUSE_DB,
        user=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASS,
    )

    t_write_end   = time.time()
    write_elapsed = t_write_end - t_write_start
    log_progress(95, "ClickHouse write complete")

    # FIX ⑥: Tách biệt thời gian inference vs write
    print(f"[BENCHMARK] ===== WRITE REPORT =====")
    print(f"[BENCHMARK] Write time        : {write_elapsed:.2f} s")
    print(f"[BENCHMARK] Write throughput  : {n_records / write_elapsed if write_elapsed > 0 else 0:.1f} records/s")
    print(f"[BENCHMARK] Total time        : {infer_elapsed + write_elapsed:.2f} s")
    print(f"[BENCHMARK] ===========================")

    processed_df.unpersist()
    spark.stop()
    log_progress(100, "Sentiment job complete")


if __name__ == "__main__":
    main()
