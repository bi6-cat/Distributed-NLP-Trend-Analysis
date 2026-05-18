"""
dags/processing_dag.py — Airflow DAG: Tích hợp CMS Keyword Streaming

Task 2.5 (Member 3 — Phase 2)
Dự án: Vietnamese Social Media Trend & Controversy Analysis System

File này chứa 2 DAGs:

1. daily_processing_pipeline (từ Member 2 — STUB/TODO):
    Schedule: 0 2 * * * (2:00 AM hàng ngày)
    Pipeline: crawl → spark_cleaning → lda_topic_modeling → load_to_clickhouse
    → Member 3 tích hợp LDA task vào pipeline chính

2. cms_keyword_streaming (Member 3):
    Schedule: */15 * * * * (mỗi 15 phút)
    Pipeline: read_new_data → run_cms_update → export_top_keywords
    → Đếm tần suất keyword streaming bằng Count-Min Sketch

CMS Streaming Architecture:
    Mỗi 15 phút, DAG:
    1. Đọc dữ liệu mới từ HDFS (incremental — chỉ file mới từ lần chạy trước)
    2. Tokenize + đếm keywords bằng Count-Min Sketch
    3. Merge với CMS state cũ (load từ pickle trên HDFS)
    4. Lưu CMS state mới (pickle) và export top-50 keywords → ClickHouse

    Tại sao CMS thay vì exact count?
    → Bộ nhớ cố định O(d×w) ≈ 80KB, bất kể bao nhiêu keyword
    → Phù hợp cho streaming liên tục 24/7 không lo OOM
    → Ref: Cormode & Muthukrishnan (2005), CS246 Stanford

Phụ thuộc:
    - Member 2: HDFS + Spark cluster + Airflow sẵn sàng
    - Member 1: Dữ liệu raw trên HDFS (crawl liên tục)
    - algorithms/count_min_sketch.py (Task 2.3 — Member 3)

Cấu hình Airflow:
    Đặt file này vào thư mục $AIRFLOW_HOME/dags/
    Hoặc cấu hình dags_folder trong airflow.cfg
"""

import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

# ============================================================================
# CẤU HÌNH CHUNG
# ============================================================================

logger = logging.getLogger("processing_dag")

# ── Đường dẫn HDFS ──
# TODO [Member 2]: Xác nhận đường dẫn HDFS thực tế
HDFS_STAGED_PATH: str = "/data/staged/"
HDFS_CMS_STATE_PATH: str = "/data/results/cms/cms_state.pkl"
HDFS_CMS_TOPK_PATH: str = "/data/results/cms/top_keywords.json"

# ── Đường dẫn local (dev/test khi chưa có HDFS) ──
LOCAL_STAGED_PATH: str = "data/fake/"
LOCAL_CMS_STATE_PATH: str = "output/cms/cms_state.pkl"
LOCAL_CMS_TOPK_PATH: str = "output/cms/top_keywords.json"

# ── CMS Config (theo TECH_STACK.md Section 5.3) ──
CMS_DEPTH: int = 5       # d = 5 hàm hash
CMS_WIDTH: int = 2048    # w = 2048 chiều rộng
CMS_TOP_K: int = 50      # Xuất top-50 keywords

# ── Chế độ chạy ──
# True: dùng local file system (dev/test)
# False: dùng HDFS (production trên cluster)
USE_LOCAL: bool = True    # TODO [Member 2]: Đổi thành False khi deploy cluster

# ── ClickHouse config ──
# TODO [Member 2/5]: Cấu hình ClickHouse connection
CLICKHOUSE_HOST: str = "localhost"
CLICKHOUSE_PORT: int = 8123
CLICKHOUSE_DB: str = "nlp_db"
CLICKHOUSE_TABLE: str = "stg_keyword_freq"


# ============================================================================
# DAG DEFAULT ARGS
# ============================================================================

default_args: Dict[str, Any] = {
    "owner": "member3-ml-engineer",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "start_date": days_ago(1),
}


# ============================================================================
# HELPER FUNCTIONS — CMS Operations
# ============================================================================

def _get_cms_state_path() -> str:
    """Lấy đường dẫn CMS state tùy chế độ (local/HDFS)."""
    return LOCAL_CMS_STATE_PATH if USE_LOCAL else HDFS_CMS_STATE_PATH


def _load_cms_state() -> "CountMinSketch":
    """
    Load CMS state từ pickle file.

    Nếu file không tồn tại (lần chạy đầu tiên), tạo CMS mới.
    Strategy: pickle trên local/HDFS — đơn giản, đủ cho Phase 2.
    Phase 3: có thể migrate sang ClickHouse BLOB nếu cần.

    Returns:
        CountMinSketch đã khôi phục hoặc mới.
    """
    # Import tại đây để tránh circular dependency khi Airflow parse DAG
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from algorithms.count_min_sketch import CountMinSketch

    state_path = _get_cms_state_path()

    if USE_LOCAL:
        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                cms = CountMinSketch.deserialize(f.read())
            logger.info(f"Loaded CMS state from {state_path} "
                        f"(total_count={cms.total_count:,})")
            return cms
    else:
        # TODO [Member 2]: Implement HDFS read
        # from hdfs import InsecureClient
        # client = InsecureClient("http://master-node:9870")
        # with client.read(state_path) as reader:
        #     cms = CountMinSketch.deserialize(reader.read())
        # return cms
        pass

    logger.info(f"No existing CMS state — creating new (d={CMS_DEPTH}, w={CMS_WIDTH})")
    return CountMinSketch(d=CMS_DEPTH, w=CMS_WIDTH)


def _save_cms_state(cms) -> None:
    """
    Lưu CMS state ra pickle file.

    Args:
        cms: CountMinSketch cần lưu.
    """
    state_path = _get_cms_state_path()

    if USE_LOCAL:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "wb") as f:
            f.write(cms.serialize())
        logger.info(f"Saved CMS state → {state_path} "
                    f"(total_count={cms.total_count:,})")
    else:
        # TODO [Member 2]: Implement HDFS write
        # from hdfs import InsecureClient
        # client = InsecureClient("http://master-node:9870")
        # with client.write(state_path, overwrite=True) as writer:
        #     writer.write(cms.serialize())
        pass


def _tokenize_text(text: str, stopwords: set, slang_dict: dict) -> List[str]:
    """
    Tokenize văn bản tiếng Việt (tái sử dụng logic từ lda_job.py).

    Pipeline: lowercase → clean → slang normalize → word_tokenize → stopword removal

    Args:
        text: Văn bản gốc.
        stopwords: Set stopwords tiếng Việt.
        slang_dict: Dict mapping slang → chuẩn.

    Returns:
        List[str] tokens sạch.
    """
    import re
    from underthesea import word_tokenize

    if not text or not isinstance(text, str):
        return []

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(
        r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
        r"ìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữự"
        r"ỳýỷỹỵđ_]",
        " ",
        text,
    )

    words = text.split()
    words = [slang_dict.get(w, w) for w in words]
    text = " ".join(words)

    try:
        tokens = word_tokenize(text, format="text").split()
    except Exception:
        tokens = text.split()

    return [
        t for t in tokens
        if t not in stopwords and not t.isdigit() and len(t) >= 2
    ]


# ============================================================================
# AIRFLOW TASK FUNCTIONS — CMS Keyword Streaming
# ============================================================================

def task_read_new_data(**context) -> List[str]:
    """
    Task 1: Đọc dữ liệu mới (incremental) từ HDFS hoặc local.

    Strategy:
        - Dùng execution_date từ Airflow context để xác định time window
        - Chỉ đọc file mới trong 15 phút gần nhất (avoid reprocessing)
        - Local mode: đọc toàn bộ CSV (cho dev/test)

    Returns:
        List[str]: Danh sách văn bản mới cần xử lý.
    """
    import pandas as pd

    texts: List[str] = []

    if USE_LOCAL:
        # ── LOCAL MODE: đọc toàn bộ CSV (dev/test) ──
        for csv_file in ["posts_5k.csv", "comments_5k.csv"]:
            csv_path = os.path.join(LOCAL_STAGED_PATH, csv_file)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, encoding="utf-8")
                text_cols = [
                    c for c in ["tieu_de", "noi_dung", "comment", "title", "content"]
                    if c in df.columns
                ]
                for _, row in df.iterrows():
                    parts = [str(row[c]) for c in text_cols if pd.notna(row[c])]
                    combined = " ".join(parts).strip()
                    if combined:
                        texts.append(combined)
        logger.info(f"[LOCAL] Read {len(texts):,} texts from {LOCAL_STAGED_PATH}")
    else:
        # ── CLUSTER MODE: đọc Parquet tăng dần từ HDFS ──
        # TODO [Member 2]: Implement incremental read
        # execution_date = context["execution_date"]
        # window_start = execution_date - timedelta(minutes=15)
        # df = spark.read.parquet(HDFS_STAGED_PATH) \
        #     .filter(F.col("crawl_timestamp") >= window_start)
        # texts = [row.text for row in df.select("text").collect()]
        logger.warning("[CLUSTER] HDFS read not yet implemented — using empty list")

    # Push danh sách texts qua XCom để task tiếp theo dùng
    context["ti"].xcom_push(key="new_texts", value=texts[:10000])
    # Giới hạn 10K texts per XCom để tránh quá tải (XCom lưu trong DB Airflow)
    logger.info(f"Pushed {min(len(texts), 10000):,} texts to XCom")

    return texts


def task_run_cms_update(**context) -> Dict[str, Any]:
    """
    Task 2: Cập nhật Count-Min Sketch với keywords từ dữ liệu mới.

    Pipeline:
        1. Load CMS state cũ từ pickle (hoặc tạo mới nếu lần đầu)
        2. Pull texts mới từ XCom (task trước đó)
        3. Tokenize → đếm mỗi keyword vào CMS
        4. Lưu CMS state mới ra pickle

    Returns:
        Dict chứa thống kê: total_count, n_new_tokens, ...
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Load tài nguyên NLP
    stopwords_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "stopwords_vi.txt",
    )
    slang_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "slang_dict.json",
    )

    stopwords: set = set()
    try:
        with open(stopwords_path, "r", encoding="utf-8") as f:
            stopwords = {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        logger.warning(f"Stopwords not found: {stopwords_path}")

    slang_dict: dict = {}
    try:
        with open(slang_path, "r", encoding="utf-8") as f:
            slang_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning(f"Slang dict not found or invalid: {slang_path}")

    # 1. Load CMS state cũ
    cms = _load_cms_state()
    count_before = cms.total_count

    # 2. Pull texts mới từ XCom
    texts: List[str] = context["ti"].xcom_pull(
        task_ids="read_new_data", key="new_texts"
    ) or []
    logger.info(f"Pulled {len(texts):,} texts from XCom")

    if not texts:
        logger.info("No new texts — skipping CMS update.")
        context["ti"].xcom_push(key="cms_updated", value=False)
        return {"total_count": cms.total_count, "n_new_tokens": 0}

    # 3. Tokenize & update CMS
    n_new_tokens = 0
    all_keywords: List[str] = []

    for text in texts:
        tokens = _tokenize_text(text, stopwords, slang_dict)
        for token in tokens:
            cms.add(token)
            n_new_tokens += 1
            all_keywords.append(token)

    # 4. Lưu CMS state mới
    _save_cms_state(cms)

    # Push danh sách unique keywords cho task export
    unique_keywords = list(set(all_keywords))
    context["ti"].xcom_push(key="unique_keywords", value=unique_keywords[:5000])
    context["ti"].xcom_push(key="cms_updated", value=True)

    stats = {
        "total_count": cms.total_count,
        "count_before": count_before,
        "n_new_tokens": n_new_tokens,
        "n_unique_keywords": len(unique_keywords),
        "epsilon": cms.epsilon,
        "delta": cms.delta,
    }
    logger.info(f"CMS updated: +{n_new_tokens:,} tokens, "
                f"total={cms.total_count:,}")
    return stats


def task_export_top_keywords(**context) -> Optional[List[Dict]]:
    """
    Task 3: Xuất top-K keywords từ CMS → ClickHouse / local JSON.

    Output format:
        [{"keyword": "iphone", "estimated_count": 1234, "rank": 1}, ...]

    Cluster mode:
        → Ghi vào ClickHouse bảng stg_keyword_freq
        → Member 5 dùng cho dashboard Streamlit

    Local mode:
        → Ghi ra file JSON (dev/test)

    Returns:
        List[Dict] top-K keywords hoặc None nếu không có update.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Kiểm tra có update không
    cms_updated = context["ti"].xcom_pull(
        task_ids="run_cms_update", key="cms_updated"
    )
    if not cms_updated:
        logger.info("CMS was not updated — skipping export.")
        return None

    # Load CMS state
    cms = _load_cms_state()

    # Pull unique keywords làm candidates cho top_k
    unique_keywords: List[str] = context["ti"].xcom_pull(
        task_ids="run_cms_update", key="unique_keywords"
    ) or []

    if not unique_keywords:
        logger.warning("No unique keywords to query.")
        return None

    # Lấy top-K keywords
    top_keywords = cms.top_k(unique_keywords, k=CMS_TOP_K)

    # Format output
    results: List[Dict] = []
    for rank, (keyword, count) in enumerate(top_keywords, start=1):
        results.append({
            "keyword": keyword,
            "estimated_count": count,
            "rank": rank,
            "timestamp": datetime.utcnow().isoformat(),
        })

    # Xuất kết quả
    if USE_LOCAL:
        # ── LOCAL: ghi JSON ──
        output_path = LOCAL_CMS_TOPK_PATH
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Exported top-{CMS_TOP_K} keywords → {output_path}")
    else:
        # ── CLUSTER: ghi ClickHouse ──
        # TODO [Member 2/5]: Implement ClickHouse writer
        # Cần cài clickhouse-connect hoặc dùng Spark JDBC
        # ─────────────────────────────────────────────────
        # import clickhouse_connect
        # client = clickhouse_connect.get_client(
        #     host=CLICKHOUSE_HOST,
        #     port=CLICKHOUSE_PORT,
        #     database=CLICKHOUSE_DB,
        # )
        # client.insert(
        #     CLICKHOUSE_TABLE,
        #     data=[[r["keyword"], r["estimated_count"], r["rank"],
        #            r["timestamp"]] for r in results],
        #     column_names=["keyword", "estimated_count", "rank", "timestamp"],
        # )
        # ─────────────────────────────────────────────────
        logger.warning("[CLUSTER] ClickHouse write not yet implemented")

    # Log top-5 cho monitoring
    logger.info(f"Top-5 keywords:")
    for r in results[:5]:
        logger.info(f"  #{r['rank']}: {r['keyword']} ({r['estimated_count']:,})")

    return results


# ============================================================================
# DAG 1: CMS KEYWORD STREAMING (Member 3 — chạy mỗi 15 phút)
# ============================================================================

with DAG(
    dag_id="cms_keyword_streaming",
    default_args=default_args,
    description=(
        "Count-Min Sketch keyword frequency streaming — "
        "đếm tần suất keyword mỗi 15 phút bằng CMS (CS246). "
        "Member 3 — Task 2.5"
    ),
    schedule_interval="*/15 * * * *",   # Mỗi 15 phút
    catchup=False,
    max_active_runs=1,                   # Chỉ chạy 1 instance tại 1 thời điểm
    tags=["member3", "cms", "streaming", "phase2"],
) as cms_dag:

    start = DummyOperator(task_id="start")

    # Task 1: Đọc dữ liệu mới
    read_data = PythonOperator(
        task_id="read_new_data",
        python_callable=task_read_new_data,
        provide_context=True,
    )

    # Task 2: Cập nhật CMS
    update_cms = PythonOperator(
        task_id="run_cms_update",
        python_callable=task_run_cms_update,
        provide_context=True,
    )

    # Task 3: Xuất top keywords
    export_keywords = PythonOperator(
        task_id="export_top_keywords",
        python_callable=task_export_top_keywords,
        provide_context=True,
    )

    end = DummyOperator(task_id="end")

    # DAG flow: start → read → update → export → end
    start >> read_data >> update_cms >> export_keywords >> end


# ============================================================================
# DAG 2: DAILY PROCESSING PIPELINE (Member 2 chính — Member 3 tích hợp LDA)
# ============================================================================

with DAG(
    dag_id="daily_processing_pipeline",
    default_args=default_args,
    description=(
        "Pipeline xử lý hàng ngày: crawl → clean → LDA → sentiment → score → ClickHouse. "
        "Member 2 (chính) + Member 3 (LDA task) + Member 4 (sentiment task)"
    ),
    schedule_interval="0 2 * * *",       # 2:00 AM hàng ngày
    catchup=False,
    max_active_runs=1,
    tags=["pipeline", "daily", "phase2"],
) as daily_dag:

    pipeline_start = DummyOperator(task_id="pipeline_start")

    # ── Task từ Member 1: Crawl ──
    # Thay bằng crawl thực tế qua BashOperator
    # Note: Truyền biến môi trường HDFS_HOST để python script tự nhận dạng hostname docker
    crawl_sources = BashOperator(
        task_id="crawl_sources",
        bash_command=(
            # SKIP TẠM THỜI: Dùng data có sẵn để tập trung vào phần processing
            # "export HDFS_HOST='namenode' && "
            # "python3 /opt/airflow/crawlers/vnexpress.py || true && "
            # "python3 /opt/airflow/crawlers/voz.py || true && "
            # "python3 /opt/airflow/crawlers/vatvo.py || true && "
            # "python3 /opt/airflow/crawlers/upload_to_hdfs.py"
            "export HDFS_HOST='namenode' PYTHONUNBUFFERED=1 && "
            "echo '[SKIP] crawlers - dùng data có sẵn' && "
            "python3 -u /opt/airflow/crawlers/upload_to_hdfs.py"
        ),
        execution_timeout=timedelta(hours=2),
    )

    # ── Task từ Member 2: Spark Cleaning + Dedup LSH ──
    from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

    # Note: SparkSubmitOperator yêu cầu connection 'spark_default' đã được tạo với host 'spark://spark-master:7077'
    # Hoặc chúng ta override trực tiếp tham số:
    spark_cleaning = SparkSubmitOperator(
        task_id="spark_cleaning",
        application="/opt/airflow/spark_jobs/cleaning_job.py",
        conn_id="spark_default",
        conf={"spark.master": "spark://spark-master:7077", "spark.pyspark.python": "/opt/bitnami/python/bin/python3", "spark.pyspark.driver.python": "/usr/local/bin/python3", "spark.executorEnv.PYSPARK_PYTHON": "/opt/bitnami/python/bin/python3", "spark.executorEnv.PYTHONPATH": "/opt/airflow"},
        executor_memory="12g",
        env_vars={
            "HDFS_INPUT": "hdfs://namenode:9000/user/zett/raw_data",
            "HDFS_OUTPUT": "hdfs://namenode:9000/user/zett/staged/sentiment/",
            "CLICKHOUSE_HOST": "clickhouse",
            "PYTHONPATH": "/opt/airflow",
        }
    )

    # ── Task từ Member 3: LDA Topic Modeling ──
    lda_topic_modeling = SparkSubmitOperator(
        task_id="lda_topic_modeling",
        application="/opt/airflow/spark_jobs/lda_job.py",
        conn_id="spark_default",
        conf={"spark.master": "spark://spark-master:7077", "spark.pyspark.python": "/opt/bitnami/python/bin/python3", "spark.pyspark.driver.python": "/usr/local/bin/python3", "spark.executorEnv.PYSPARK_PYTHON": "/opt/bitnami/python/bin/python3", "spark.executorEnv.PYTHONPATH": "/opt/airflow"},
        executor_memory="12g",
        application_args=[
            "--input-path", "hdfs://namenode:9000/user/zett/staged/",
            "--output-path", "hdfs://namenode:9000/user/zett/results/lda/",
            "--k", "20"
        ]
    )

    # ── Task từ Member 4: Sentiment Analysis ──
    sentiment_analysis = SparkSubmitOperator(
        task_id="sentiment_analysis",
        application="/opt/airflow/spark_jobs/sentiment_job.py",
        conn_id="spark_default",
        conf={"spark.master": "spark://spark-master:7077", "spark.pyspark.python": "/opt/bitnami/python/bin/python3", "spark.pyspark.driver.python": "/usr/local/bin/python3", "spark.executorEnv.PYSPARK_PYTHON": "/opt/bitnami/python/bin/python3", "spark.executorEnv.PYTHONPATH": "/opt/airflow", "spark.executorEnv.KAGGLE_MODEL_HANDLE": "nquanggnguyn/phobert-/transformers/default"},
        executor_memory="12g",
        env_vars={
            "HDFS_INPUT": "hdfs://namenode:9000/user/zett/staged/",
            "CLICKHOUSE_HOST": "clickhouse",
            "KAGGLE_MODEL_HANDLE": "nquanggnguyn/phobert-/transformers/default",
            "PYTHONPATH": "/opt/airflow",
        }
    )

    # ── Task từ Member 5: Trend Scoring (bằng dbt) ──
    dbt_transform = BashOperator(
        task_id="dbt_transform",
        bash_command="cd /opt/airflow/warehouse/dbt_project && dbt run --profiles-dir .",
    )

    pipeline_end = DummyOperator(task_id="pipeline_end")

    # ── DAG Flow ──
    # crawl → clean → [LDA + sentiment song song] → dbt 
    # (Scoring & Load to Clickhouse đã nằm trong dbt & NLP jobs)
    pipeline_start >> crawl_sources >> spark_cleaning
    spark_cleaning >> [lda_topic_modeling, sentiment_analysis]
    [lda_topic_modeling, sentiment_analysis] >> dbt_transform >> pipeline_end
