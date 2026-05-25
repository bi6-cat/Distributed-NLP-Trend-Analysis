"""
dags/processing_dag.py — Airflow DAG: Tích hợp CMS Keyword Streaming

Task 2.5 (Member 3 — Phase 2)
Dự án: Vietnamese Social Media Trend & Controversy Analysis System

File này chứa 2 DAGs:

1. full_processing_pipeline:
    crawl → spark_cleaning → lda_topic_modeling → sentiment_analysis
    ├→ save_lda_to_clickhouse
    ├→ cms_keyword_counting
    └→ dbt_transform → pipeline_end
    Nguồn dữ liệu chung: stg_posts_core (cleaning output)

2. bertopic_weekly_inference (Chủ nhật 3:00 AM):
    load_model → run_inference_pipeline (stg_posts_core 7 ngày) → save_topics_to_clickhouse
    Nguồn dữ liệu: stg_posts_core (cùng nguồn với LDA, filter 7 ngày)

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
from datetime import datetime, timedelta, timezone
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


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}

HDFS_NAMENODE: str = os.getenv("HDFS_NAMENODE", "namenode:9000")
WEBHDFS_HOST: str = os.getenv("WEBHDFS_HOST", "namenode:9870")
WEBHDFS_HOSTNAME: str = WEBHDFS_HOST.split(":")[0]
HDFS_USER: str = os.getenv("HDFS_USER", os.getenv("HADOOP_USER_NAME", "root"))
HDFS_USER_DIR: str = f"/user/{HDFS_USER}"
HDFS_URI_PREFIX: str = f"hdfs://{HDFS_NAMENODE}"
SPARK_MASTER: str = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")

# ── Đường dẫn HDFS ──
# stg_posts_core là nguồn duy nhất dùng chung cho LDA và BERTopic
STAGED_HDFS_PATH:    str = f"{HDFS_URI_PREFIX}{HDFS_USER_DIR}/staged/stg_posts_core"
RAW_HDFS_PATH:       str = f"{HDFS_URI_PREFIX}{HDFS_USER_DIR}/raw_data"
LDA_HDFS_OUTPUT:          str = f"{HDFS_URI_PREFIX}{HDFS_USER_DIR}/results/lda"
HDFS_CMS_STATE_PATH:      str = "/data/results/cms/cms_state.pkl"
HDFS_CMS_TOPK_PATH:       str = "/data/results/cms/top_keywords.json"
HDFS_SILVER_POST_TOPICS:  str = "/data/silver/post_topics"

# ── Đường dẫn local (Docker volume /opt/airflow) ──
# cleaning ghi ra đây → LDA và BERTopic đọc từ đây (cùng nguồn)
STAGED_LOCAL_PATH:   str = os.getenv("STAGED_LOCAL_PATH", "/opt/airflow/data/preprocessed/stg_posts_core")
RAW_LOCAL_PATH:      str = os.getenv("RAW_LOCAL_PATH", "/opt/airflow/crawlers/data")
LDA_LOCAL_OUTPUT:    str = os.getenv("LDA_LOCAL_OUTPUT", "/opt/airflow/output/lda")
LOCAL_SENTIMENT_MODEL_PATH: str = os.getenv("LOCAL_SENTIMENT_MODEL_PATH", "/opt/airflow/models/phobert_finetuned/final")
LOCAL_STOPWORDS_PATH: str = os.getenv("LOCAL_STOPWORDS_PATH", "/opt/airflow/data/stopwords_vi.txt")
LOCAL_SLANG_DICT_PATH: str = os.getenv("LOCAL_SLANG_DICT_PATH", "/opt/airflow/data/slang_dict.json")
LOCAL_CMS_STATE_PATH: str = "output/cms/cms_state.pkl"
LOCAL_CMS_TOPK_PATH:  str = "output/cms/top_keywords.json"

# Alias cho CMS tasks (giữ backward compat)
HDFS_STAGED_PATH: str = STAGED_HDFS_PATH
LOCAL_STAGED_PATH: str = "data/fake/"

# ── CMS Config (theo TECH_STACK.md Section 5.3) ──
CMS_DEPTH: int = 5       # d = 5 hàm hash
CMS_WIDTH: int = 2048    # w = 2048 chiều rộng
CMS_TOP_K: int = 50      # Xuất top-50 keywords

# ── Chế độ chạy ──
# True: dùng local file system (dev/test)
# False: dùng HDFS (production trên cluster)
USE_LOCAL: bool = _env_bool("USE_LOCAL", False)

# ── ClickHouse config ──
# TODO [Member 2/5]: Cấu hình ClickHouse connection
CLICKHOUSE_HOST: str = os.getenv("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_PORT: int = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CLICKHOUSE_DB: str = os.getenv("CLICKHOUSE_DB", "tech_radar")
CLICKHOUSE_TABLE: str = os.getenv("CLICKHOUSE_TABLE", "stg_keyword_freq")
CLICKHOUSE_USER: str = os.getenv("CLICKHOUSE_USER", "root")
CLICKHOUSE_PASSWORD: str = os.getenv("CLICKHOUSE_PASSWORD", "root")


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

def _load_stg_posts_core_window(window_minutes: int = 15):
    """
    Đọc stg_posts_core và filter 15 phút gần nhất theo crawled_at.
    Dùng chung cho task_read_new_data (CMS streaming).

    Local: đọc parquet dir STAGED_LOCAL_PATH.
    Cluster: đọc qua WebHDFS từ STAGED_HDFS_PATH.

    Returns:
        DataFrame với columns: clean_text (hoặc segmented_text), source, crawled_at
    """
    import io
    import re
    import pandas as pd
    import requests as _req

    window_start = datetime.now(tz=timezone.utc) - timedelta(minutes=window_minutes)

    def _filter_and_select(df: pd.DataFrame) -> pd.DataFrame:
        if "crawled_at" in df.columns:
            df["crawled_at"] = pd.to_datetime(df["crawled_at"], utc=True, errors="coerce")
            df = df[df["crawled_at"] >= window_start].copy()
        text_col = next(
            (c for c in ["clean_text", "segmented_text", "body"] if c in df.columns),
            None,
        )
        if text_col is None:
            return pd.DataFrame(columns=["text", "source"])
        df = df.rename(columns={text_col: "text"})
        src = df["source"].astype(str) if "source" in df.columns else pd.Series("unknown", index=df.index)
        df["source"] = src.fillna("unknown").replace("nan", "unknown")
        return df[["text", "source"]].dropna(subset=["text"])

    def _list_hdfs_parquet_files_recursive(webhdfs_base: str, hdfs_dir: str) -> List[str]:
        resp = _req.get(
            f"{webhdfs_base}{hdfs_dir}?op=LISTSTATUS&user.name={HDFS_USER}",
            timeout=30,
        )
        resp.raise_for_status()

        parquet_files: List[str] = []
        for status in resp.json()["FileStatuses"]["FileStatus"]:
            path_suffix = status["pathSuffix"]
            child_path = f"{hdfs_dir.rstrip('/')}/{path_suffix}"
            if status["type"] == "FILE" and path_suffix.endswith(".parquet"):
                parquet_files.append(child_path)
            elif status["type"] == "DIRECTORY":
                parquet_files.extend(_list_hdfs_parquet_files_recursive(webhdfs_base, child_path))
        return parquet_files

    def _extract_source_from_hdfs_path(path: str) -> Optional[str]:
        match = re.search(r"/source=([^/]+)/", path)
        return match.group(1) if match else None

    if USE_LOCAL:
        if not os.path.isdir(STAGED_LOCAL_PATH):
            logger.warning(f"[LOCAL] {STAGED_LOCAL_PATH} chưa tồn tại — chạy cleaning trước")
            return pd.DataFrame(columns=["text", "source"])
        df = pd.read_parquet(STAGED_LOCAL_PATH)
        logger.info(f"[LOCAL] Loaded stg_posts_core: {len(df):,} rows")
        return _filter_and_select(df)
    else:
        webhdfs = f"http://{WEBHDFS_HOST}/webhdfs/v1"
        hdfs_path = f"{HDFS_USER_DIR}/staged/stg_posts_core"
        files = _list_hdfs_parquet_files_recursive(webhdfs, hdfs_path)
        dfs = []
        for fname in files:
            resp = _req.get(
                f"{webhdfs}{fname}?op=OPEN&user.name={HDFS_USER}",
                allow_redirects=True,
                timeout=120,
            )
            resp.raise_for_status()
            part_df = pd.read_parquet(io.BytesIO(resp.content))
            if "source" not in part_df.columns:
                source_name = _extract_source_from_hdfs_path(fname)
                if source_name:
                    part_df["source"] = source_name
            dfs.append(part_df)
        if not dfs:
            logger.warning("[CLUSTER] stg_posts_core: không có parquet file")
            return pd.DataFrame(columns=["text", "source"])
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"[CLUSTER] Loaded stg_posts_core: {len(df):,} rows")
        return _filter_and_select(df)


def task_read_new_data(**context) -> None:
    """
    Task 1: Đọc stg_posts_core incremental (15 phút gần nhất).

    Nguồn: stg_posts_core — cùng bảng cleaning_job ghi, LDA + BERTopic đọc.
    Push XCom:
        new_texts   — list[str] text content (tối đa 10K)
        new_sources — list[str] source per text (voz/tinhte/vnexpress/youtube)
    """
    df = _load_stg_posts_core_window(window_minutes=15)
    # Giới hạn 10K để tránh quá tải XCom (lưu trong Airflow metadata DB)
    df = df.head(10000)
    texts   = df["text"].tolist()
    sources = df["source"].tolist()
    logger.info(f"New data: {len(texts):,} texts | sources: {df['source'].value_counts().to_dict()}")
    context["ti"].xcom_push(key="new_texts",   value=texts)
    context["ti"].xcom_push(key="new_sources", value=sources)


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

    # 2. Pull texts + sources từ XCom
    texts: List[str] = context["ti"].xcom_pull(
        task_ids="read_new_data", key="new_texts"
    ) or []
    sources: List[str] = context["ti"].xcom_pull(
        task_ids="read_new_data", key="new_sources"
    ) or ["unknown"] * len(texts)
    logger.info(f"Pulled {len(texts):,} texts from XCom")

    if not texts:
        logger.info("No new texts — skipping CMS update.")
        context["ti"].xcom_push(key="cms_updated", value=False)
        return {"total_count": cms.total_count, "n_new_tokens": 0}

    # 3. Tokenize & update CMS; track per-source keyword sets
    n_new_tokens = 0
    # source_keywords: {source: set_of_unique_keywords}
    source_keywords: Dict[str, set] = {}

    for text, src in zip(texts, sources):
        tokens = _tokenize_text(text, stopwords, slang_dict)
        for token in tokens:
            cms.add(token)
            n_new_tokens += 1
            source_keywords.setdefault(src, set()).add(token)

    # 4. Lưu CMS state mới
    _save_cms_state(cms)

    # Push candidates cho export: {source: [unique_keywords]}
    source_kw_export = {src: list(kws)[:2000] for src, kws in source_keywords.items()}
    all_unique = list({kw for kws in source_keywords.values() for kw in kws})
    context["ti"].xcom_push(key="source_keywords", value=source_kw_export)
    context["ti"].xcom_push(key="unique_keywords", value=all_unique[:5000])
    context["ti"].xcom_push(key="cms_updated", value=True)

    stats = {
        "total_count": cms.total_count,
        "count_before": count_before,
        "n_new_tokens": n_new_tokens,
        "n_unique_keywords": len(all_unique),
        "epsilon": cms.epsilon,
        "delta": cms.delta,
    }
    logger.info(f"CMS updated: +{n_new_tokens:,} tokens, "
                f"total={cms.total_count:,}")
    return stats


def task_export_top_keywords(**context) -> Optional[List[Dict]]:
    """
    Task 3: Xuất top-K keywords per source từ CMS → ClickHouse stg_keyword_freq.

    Schema: keyword | window_start | window_end | estimated_count | source
    Mỗi source (voz/tinhte/vnexpress/youtube) có top-K riêng trong cùng window.
    """
    import pandas as pd

    cms_updated = context["ti"].xcom_pull(task_ids="run_cms_update", key="cms_updated")
    if not cms_updated:
        logger.info("CMS was not updated — skipping export.")
        return None

    cms = _load_cms_state()

    # source_keywords: {source: [unique_keywords]} từ task trước
    source_keywords: Dict[str, List[str]] = context["ti"].xcom_pull(
        task_ids="run_cms_update", key="source_keywords"
    ) or {}

    if not source_keywords:
        logger.warning("No source_keywords in XCom.")
        return None

    execution_date = context.get("execution_date") or datetime.now(tz=timezone.utc)
    window_end   = execution_date if isinstance(execution_date, datetime) else datetime.now(tz=timezone.utc)
    window_start = window_end - timedelta(minutes=15)
    # strip tz for ClickHouse DateTime (no timezone support)
    ws = window_start.replace(tzinfo=None)
    we = window_end.replace(tzinfo=None)

    # Build per-source top-K rows (spec: stg_keyword_freq)
    results: List[Dict] = []
    for src, candidates in source_keywords.items():
        for keyword, count in cms.top_k(candidates, k=CMS_TOP_K):
            results.append({
                "keyword":         keyword,
                "window_start":    ws,
                "window_end":      we,
                "estimated_count": int(count),
                "source":          src,
            })

    if not results:
        logger.warning("top_k returned no results.")
        return None

    # ── Ghi ClickHouse stg_keyword_freq ──
    ch_host = os.environ.get("CLICKHOUSE_HOST", "clickhouse")
    try:
        import clickhouse_connect
        client = clickhouse_connect.get_client(
            host=ch_host, port=CLICKHOUSE_PORT,
            username=os.environ.get("CLICKHOUSE_USER", "root"),
            password=os.environ.get("CLICKHOUSE_PASSWORD", "root"),
            database=CLICKHOUSE_DB,
        )
        df_out = pd.DataFrame(results)
        df_out["estimated_count"] = df_out["estimated_count"].astype("int64")
        client.insert_df(CLICKHOUSE_TABLE, df_out)
        logger.info(f"Inserted {len(df_out)} rows → {CLICKHOUSE_DB}.{CLICKHOUSE_TABLE}")
    except Exception as exc:
        # Fallback local JSON khi test mà chưa có ClickHouse
        logger.warning(f"ClickHouse unavailable ({exc}) — writing fallback JSON")
        os.makedirs(os.path.dirname(LOCAL_CMS_TOPK_PATH), exist_ok=True)
        with open(LOCAL_CMS_TOPK_PATH, "w", encoding="utf-8") as f:
            json.dump(
                [{**r, "window_start": r["window_start"].isoformat(),
                       "window_end":   r["window_end"].isoformat()} for r in results],
                f, ensure_ascii=False, indent=2,
            )
        logger.info(f"Fallback JSON → {LOCAL_CMS_TOPK_PATH}")

    # Log top-5 cho monitoring
    logger.info("Top-5 keywords:")
    for i, r in enumerate(results[:5], start=1):
        logger.info(f"  #{i}: {r['keyword']} ({r['estimated_count']:,})")

    return results


# cms_keyword_streaming DAG đã được gộp vào full_processing_pipeline
# (task cms_keyword_counting chạy sau spark_cleaning, song song với LDA + sentiment)


# ============================================================================
# TASK — CMS daily batch (dùng trong full_processing_pipeline)
# ============================================================================

def task_run_cms_daily() -> None:
    """
    CMS keyword counting — chạy sau sentiment_analysis trong full_processing_pipeline.

    Input:  stg_posts_core (STAGED_LOCAL_PATH hoặc STAGED_HDFS_PATH)
            — cùng nguồn với LDA và BERTopic, window 24h (crawled_at hôm nay)
            — cột dùng: clean_text (hoặc segmented_text), source, crawled_at

    Output: ClickHouse tech_radar.stg_keyword_freq
            schema: keyword | window_start | window_end | estimated_count | source
            — top-K keywords per source (voz/tinhte/vnexpress/youtube)
            — fallback ghi JSON local nếu ClickHouse chưa sẵn sàng

    Reuses: _load_stg_posts_core_window, _tokenize_text,
            _load_cms_state, _save_cms_state, cms.top_k
    """
    import sys as _sys
    import pandas as _pd

    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # ── 1. Input: stg_posts_core 24h window ──
    df = _load_stg_posts_core_window(window_minutes=24 * 60)
    if df.empty:
        logger.warning("[CMS] Không có dữ liệu mới trong 24h — skip.")
        return
    logger.info(f"[CMS] Input: {len(df):,} rows | sources: {df['source'].value_counts().to_dict()}")

    # ── 2. Load NLP resources (stopwords + slang) ──
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stopwords: set = set()
    slang_dict: dict = {}
    try:
        with open(os.path.join(root, "data", "stopwords_vi.txt"), encoding="utf-8") as f:
            stopwords = {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        logger.warning("[CMS] stopwords_vi.txt not found — proceeding without")
    try:
        with open(os.path.join(root, "data", "slang_dict.json"), encoding="utf-8") as f:
            slang_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("[CMS] slang_dict.json not found — proceeding without")

    # ── 3. Tokenize + CMS update (per-source tracking) ──
    # One CMS per source so top_k returns source-specific counts, not global counts
    from algorithms.count_min_sketch import CountMinSketch as _CMS
    source_cms: Dict[str, object] = {}
    source_keywords: Dict[str, set] = {}

    for _, row in df.iterrows():
        tokens = _tokenize_text(str(row["text"]), stopwords, slang_dict)
        src = str(row.get("source", "unknown"))
        if src not in source_cms:
            source_cms[src] = _CMS(d=5, w=4096)
        for token in tokens:
            source_cms[src].add(token)
            source_keywords.setdefault(src, set()).add(token)

    _save_cms_state(_load_cms_state())
    total_unique = sum(len(v) for v in source_keywords.values())
    logger.info(f"[CMS] CMS updated — {total_unique:,} unique tokens across {len(source_cms)} sources")

    # ── 4. Build output rows: top-K per source → stg_keyword_freq schema ──
    now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    window_start = now.replace(hour=0, minute=0, second=0, microsecond=0)  # midnight today

    results: List[Dict] = []
    for src, candidates in source_keywords.items():
        for keyword, count in source_cms[src].top_k(list(candidates), k=CMS_TOP_K):
            results.append({
                "keyword":         keyword,           # String
                "window_start":    window_start,      # DateTime (midnight)
                "window_end":      now,               # DateTime (now)
                "estimated_count": int(count),        # Int64
                "source":          src,               # LowCardinality String
            })

    if not results:
        logger.warning("[CMS] top_k returned no results — skip insert.")
        return

    # ── 5. Output: insert vào ClickHouse stg_keyword_freq ──
    ch_host = os.environ.get("CLICKHOUSE_HOST", "clickhouse")
    try:
        import clickhouse_connect
        client = clickhouse_connect.get_client(
            host=ch_host, port=CLICKHOUSE_PORT,
            username=os.environ.get("CLICKHOUSE_USER", "root"),
            password=os.environ.get("CLICKHOUSE_PASSWORD", "root"),
            database=CLICKHOUSE_DB,
        )
        df_out = _pd.DataFrame(results)
        df_out["estimated_count"] = df_out["estimated_count"].astype("int64")
        client.insert_df(CLICKHOUSE_TABLE, df_out)
        logger.info(f"[CMS] Inserted {len(df_out)} rows → {CLICKHOUSE_DB}.{CLICKHOUSE_TABLE}")
    except Exception as exc:
        # Fallback JSON khi test mà ClickHouse chưa up
        logger.warning(f"[CMS] ClickHouse unavailable ({exc}) — fallback JSON")
        os.makedirs(os.path.dirname(LOCAL_CMS_TOPK_PATH), exist_ok=True)
        with open(LOCAL_CMS_TOPK_PATH, "w", encoding="utf-8") as f:
            json.dump(
                [{**r,
                  "window_start": r["window_start"].isoformat(),
                  "window_end":   r["window_end"].isoformat()} for r in results],
                f, ensure_ascii=False, indent=2,
            )
        logger.info(f"[CMS] Fallback → {LOCAL_CMS_TOPK_PATH}")


# ============================================================================
# HELPER FUNCTION — post_topics → HDFS silver layer
# ============================================================================

def task_save_post_topics_to_hdfs(
    model_type: str = "lda",
    local_src: Optional[str] = None,
    hdfs_src: Optional[str] = None,
    **context,
) -> None:
    """
    Ghi post_topic_assignment lên HDFS silver layer.

    Output path: /data/silver/post_topics/date={YYYY-MM-DD}/part-{model_type}-0.parquet
    LDA và BERTopic cùng thư mục date=, khác tên file → Spark đọc chung, không overwrite.
    Phân biệt model qua cột model_type đã có sẵn trong data.

    Args:
        model_type: "lda" hoặc "bertopic" — dùng làm suffix tên file.
        local_src:  path local tới post_topic_assignment.parquet.
                    Mặc định: LDA_LOCAL_OUTPUT/post_topic_assignment.parquet
        hdfs_src:   HDFS directory chứa part files parquet.
                    Mặc định: {HDFS_USER_DIR}/results/lda/post_topic_assignment
    """
    import io
    import requests as _req
    import pandas as _pd

    date_str: str = context["ds"]
    silver_dir = f"{HDFS_SILVER_POST_TOPICS}/date={date_str}"
    part_path  = f"{silver_dir}/part-{model_type}-0.parquet"

    _local_src = local_src or os.path.join(LDA_LOCAL_OUTPUT, "post_topic_assignment.parquet")
    _hdfs_src  = hdfs_src  or f"{HDFS_USER_DIR}/results/lda/post_topic_assignment"

    # ── Đọc post_topics ──────────────────────────────────────────────────────
    if USE_LOCAL:
        post_topics_df = _pd.read_parquet(_local_src)
    else:
        webhdfs = f"http://{WEBHDFS_HOST}/webhdfs/v1"
        resp = _req.get(
            f"{webhdfs}{_hdfs_src}?op=LISTSTATUS&user.name={HDFS_USER}",
            timeout=30,
        )
        resp.raise_for_status()
        parts = [
            s["pathSuffix"]
            for s in resp.json()["FileStatuses"]["FileStatus"]
            if s["type"] == "FILE" and s["pathSuffix"].endswith(".parquet")
        ]
        if not parts:
            raise FileNotFoundError(f"Không tìm thấy parquet files trong: {_hdfs_src}")
        post_topics_df = _pd.concat(
            [
                _pd.read_parquet(io.BytesIO(
                    _req.get(
                        f"{webhdfs}{_hdfs_src}/{p}?op=OPEN&user.name={HDFS_USER}",
                        allow_redirects=True, timeout=120,
                    ).content
                ))
                for p in parts
            ],
            ignore_index=True,
        )

    # ── Ghi lên HDFS silver ──────────────────────────────────────────────────
    buf = io.BytesIO()
    post_topics_df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)

    if USE_LOCAL:
        local_dir = f"/opt/airflow/data/silver/post_topics/date={date_str}"
        os.makedirs(local_dir, exist_ok=True)
        with open(os.path.join(local_dir, f"part-{model_type}-0.parquet"), "wb") as f:
            f.write(buf.getvalue())
        logger.info(f"[silver/{model_type}] post_topics → local {local_dir} ({len(post_topics_df)} rows)")
        return

    webhdfs = f"http://{WEBHDFS_HOST}/webhdfs/v1"
    _req.put(
        f"{webhdfs}{silver_dir}?op=MKDIRS&user.name={HDFS_USER}",
        timeout=30,
    ).raise_for_status()

    redir = _req.put(
        f"{webhdfs}{part_path}?op=CREATE&user.name={HDFS_USER}&overwrite=true",
        allow_redirects=False,
        timeout=30,
    )
    upload_url = redir.headers.get("Location") if redir.status_code == 307 else None
    if upload_url:
        _req.put(
            upload_url,
            data=buf.getvalue(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=120,
        ).raise_for_status()
    else:
        redir.raise_for_status()

    logger.info(f"[silver/{model_type}] post_topics → hdfs:{part_path} ({len(post_topics_df)} rows)")


# ============================================================================
# HELPER FUNCTION — LDA → ClickHouse (dùng trong full_processing_pipeline)
# ============================================================================

def task_save_lda_to_clickhouse() -> None:
    """
    Đọc LDA Parquet rồi insert vào ClickHouse.

    Local/Docker: đọc parquet từ LDA_LOCAL_OUTPUT (volume mount).
    Cluster:      đọc parquet từ LDA_HDFS_OUTPUT qua WebHDFS.
    """
    import io
    import sys as _sys
    import pandas as _pd
    import requests as _req

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, root)
    from scripts.save_topics_to_ch import (
        create_tables,
        export_handoff_summary,
        get_ch_client,
        insert_stg_post_topics,
        insert_stg_topics,
        verify_load,
    )

    def _read_hdfs_parquet_dataset(hdfs_dir: str) -> _pd.DataFrame:
        webhdfs = f"http://{WEBHDFS_HOST}/webhdfs/v1"
        list_url = f"{webhdfs}{hdfs_dir}?op=LISTSTATUS&user.name={HDFS_USER}"
        resp = _req.get(list_url, timeout=30)
        resp.raise_for_status()

        parts = [
            status["pathSuffix"]
            for status in resp.json()["FileStatuses"]["FileStatus"]
            if status["type"] == "FILE" and status["pathSuffix"].endswith(".parquet")
        ]
        if not parts:
            raise FileNotFoundError(f"Không tìm thấy parquet part files trong HDFS dir: {hdfs_dir}")

        dfs = []
        for part in parts:
            open_url = f"{webhdfs}{hdfs_dir}/{part}?op=OPEN&user.name={HDFS_USER}"
            part_resp = _req.get(open_url, allow_redirects=True, timeout=120)
            part_resp.raise_for_status()
            dfs.append(_pd.read_parquet(io.BytesIO(part_resp.content)))
        return _pd.concat(dfs, ignore_index=True)

    ch_host = os.environ.get("CLICKHOUSE_HOST", CLICKHOUSE_HOST)
    ch_user = os.environ.get("CLICKHOUSE_USER", CLICKHOUSE_USER)
    ch_pass = os.environ.get("CLICKHOUSE_PASSWORD", os.environ.get("CLICKHOUSE_PASS", CLICKHOUSE_PASSWORD))

    if USE_LOCAL:
        topics_df = _pd.read_parquet(os.path.join(LDA_LOCAL_OUTPUT, "topics.parquet"))
        post_topics_df = _pd.read_parquet(os.path.join(LDA_LOCAL_OUTPUT, "post_topic_assignment.parquet"))
    else:
        topics_df = _read_hdfs_parquet_dataset(f"{HDFS_USER_DIR}/results/lda/topics")
        post_topics_df = _read_hdfs_parquet_dataset(f"{HDFS_USER_DIR}/results/lda/post_topic_assignment")

    client = get_ch_client(
        host=ch_host,
        port=CLICKHOUSE_PORT,
        database=CLICKHOUSE_DB,
        username=ch_user,
        password=ch_pass,
    )
    create_tables(client, CLICKHOUSE_DB)
    insert_stg_topics(client, topics_df, CLICKHOUSE_DB)
    if not post_topics_df.empty:
        insert_stg_post_topics(client, post_topics_df, CLICKHOUSE_DB)
    verify_load(client, CLICKHOUSE_DB)
    export_handoff_summary(
        topics_df,
        os.path.join(root, "output", "handoff", "topic_summary_for_m5.csv"),
    )
    logger.info("[LDA] Topics pushed to ClickHouse.")


# ============================================================================
# DAG 2: FULL PROCESSING PIPELINE (historical/full reprocessing)
# ============================================================================

with DAG(
    dag_id="full_processing_pipeline",
    default_args=default_args,
    description=(
        "Full processing pipeline: crawl → clean → LDA → sentiment "
        "→ ClickHouse landing tables → dbt transform."
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
            f"export HDFS_HOST='{WEBHDFS_HOSTNAME}' HDFS_USER='{HDFS_USER}' PYTHONUNBUFFERED=1 && "
            # Tạm thời tắt crawl mạng để chạy data có sẵn:
            # "python3 -u /opt/airflow/crawlers/vnexpress.py || true && "
            # "python3 -u /opt/airflow/crawlers/voz.py || true && "
            # "python3 -u /opt/airflow/crawlers/vatvo.py || true && "
            "python3 -u /opt/airflow/crawlers/upload_to_hdfs.py"
        ),
        execution_timeout=timedelta(hours=2),
    )

    spark_common_conf = (
        f"--master {SPARK_MASTER} "
        "--conf spark.pyspark.python=/opt/bitnami/python/bin/python3 "
        "--conf spark.pyspark.driver.python=/usr/local/bin/python3 "
        "--conf spark.executorEnv.PYSPARK_PYTHON=/opt/bitnami/python/bin/python3 "
        "--conf spark.executorEnv.PYTHONPATH=/opt/airflow "
        f"--conf spark.executorEnv.HADOOP_USER_NAME={HDFS_USER} "
        f"--conf spark.executorEnv.HDFS_USER={HDFS_USER} "
        "--conf spark.hadoop.fs.permissions.umask-mode=000 "
        "--executor-memory 12g "
    )

    # ── Task từ Member 2: Spark Cleaning + Dedup LSH ──
    spark_cleaning = BashOperator(
        task_id="spark_cleaning",
        bash_command=(
            "export PYTHONPATH='/opt/airflow' "
            f"HADOOP_USER_NAME='{HDFS_USER}' "
            f"HDFS_INPUT='{RAW_LOCAL_PATH if USE_LOCAL else RAW_HDFS_PATH}' "
            f"HDFS_OUTPUT='{STAGED_LOCAL_PATH if USE_LOCAL else STAGED_HDFS_PATH}' "
            f"HDFS_USER='{HDFS_USER}' "
            f"CLICKHOUSE_HOST='{CLICKHOUSE_HOST}' "
            "&& spark-submit "
            f"{spark_common_conf}"
            "/opt/airflow/spark_jobs/cleaning_job.py"
        ),
        execution_timeout=timedelta(hours=2),
    )

    # ── Task từ Member 3: LDA Topic Modeling ──
    lda_local_flag = "--local" if USE_LOCAL else ""
    lda_topic_modeling = BashOperator(
        task_id="lda_topic_modeling",
        bash_command=(
            "export PYTHONPATH='/opt/airflow' "
            f"HADOOP_USER_NAME='{HDFS_USER}' "
            f"HDFS_USER='{HDFS_USER}' "
            "&& spark-submit "
            f"{spark_common_conf}"
            "/opt/airflow/spark_jobs/lda_job.py "
            f"--input-path '{STAGED_LOCAL_PATH if USE_LOCAL else STAGED_HDFS_PATH}' "
            f"--output-path '{LDA_LOCAL_OUTPUT if USE_LOCAL else LDA_HDFS_OUTPUT}' "
            "--k 20 "
            f"--stopwords-path '{LOCAL_STOPWORDS_PATH if USE_LOCAL else f'{HDFS_URI_PREFIX}{HDFS_USER_DIR}/ref/stopwords_vi.txt'}' "
            f"--slang-dict-path '{LOCAL_SLANG_DICT_PATH if USE_LOCAL else f'{HDFS_URI_PREFIX}{HDFS_USER_DIR}/ref/slang_dict.json'}' "
            f"{lda_local_flag}"
        ),
        execution_timeout=timedelta(hours=2),
    )

    # ── Task từ Member 4: Sentiment Analysis ──
    sentiment_analysis = BashOperator(
        task_id="sentiment_analysis",
        bash_command=(
            "export PYTHONPATH='/opt/airflow' "
            f"HADOOP_USER_NAME='{HDFS_USER}' "
            f"HDFS_INPUT='{STAGED_LOCAL_PATH if USE_LOCAL else STAGED_HDFS_PATH}' "
            f"HDFS_USER='{HDFS_USER}' "
            f"CLICKHOUSE_HOST='{CLICKHOUSE_HOST}' "
            f"CLICKHOUSE_DB='{CLICKHOUSE_DB}' "
            f"CLICKHOUSE_USER='{CLICKHOUSE_USER}' "
            f"CLICKHOUSE_PASS='{CLICKHOUSE_PASSWORD}' "
            f"NLP_MODEL_PATH='{LOCAL_SENTIMENT_MODEL_PATH}' "
            "NLP_MODEL_VERSION='phobert_v1' "
            "&& spark-submit "
            f"{spark_common_conf}"
            f"--conf spark.executorEnv.CLICKHOUSE_DB={CLICKHOUSE_DB} "
            f"--conf spark.executorEnv.CLICKHOUSE_USER={CLICKHOUSE_USER} "
            f"--conf spark.executorEnv.CLICKHOUSE_PASS={CLICKHOUSE_PASSWORD} "
            f"--conf spark.executorEnv.NLP_MODEL_PATH={LOCAL_SENTIMENT_MODEL_PATH} "
            "--conf spark.executorEnv.NLP_MODEL_VERSION=phobert_v1 "
            "/opt/airflow/spark_jobs/sentiment_job.py"
        ),
        execution_timeout=timedelta(hours=2),
    )

    # ── Task từ Member 5: Trend Scoring (bằng dbt) ──
    dbt_transform = BashOperator(
        task_id="dbt_transform",
        bash_command="cd /opt/airflow/warehouse/dbt_project && dbt run --profiles-dir .",
    )

    # ── Task từ Member 3: Push LDA results → ClickHouse ──
    save_lda_to_ch = PythonOperator(
        task_id="save_lda_to_clickhouse",
        python_callable=task_save_lda_to_clickhouse,
    )

    # ── Task từ Member 3: Ghi post_topics → HDFS silver layer ──
    save_post_topics_hdfs = PythonOperator(
        task_id="save_post_topics_to_hdfs",
        python_callable=task_save_post_topics_to_hdfs,
        op_kwargs={"model_type": "lda"},
        provide_context=True,
    )

    # ── Task từ Member 3: CMS keyword frequency (daily batch) ──
    run_cms = PythonOperator(
        task_id="cms_keyword_counting",
        python_callable=task_run_cms_daily,
    )

    pipeline_end = DummyOperator(task_id="pipeline_end")

    # ── DAG Flow ──
    # crawl → clean → LDA → sentiment
    #                    ├→ save_lda_to_ch ──┬──► dbt_transform → end
    #                    └→ save_post_topics─┘
    # clean ─────────────────────────────────────→ CMS ──► dbt_transform
    pipeline_start >> crawl_sources >> spark_cleaning
    spark_cleaning >> lda_topic_modeling
    lda_topic_modeling >> sentiment_analysis
    sentiment_analysis >> [save_lda_to_ch, save_post_topics_hdfs] >> dbt_transform
    sentiment_analysis >> run_cms >> dbt_transform
    dbt_transform >> pipeline_end


# ============================================================================
# DAG 3: BERTOPIC WEEKLY INFERENCE (Member 3 — chạy Chủ nhật 3h sáng)
# ============================================================================
# Tại sao dùng PythonOperator (không phải SparkSubmitOperator)?
#   - BERTopic dùng PyTorch + HDBSCAN — không chạy trên Spark executor
#   - Inference chạy single-machine: load model → batch transform → export
#   - 1 tuần data ≈ 50K-200K docs — đủ với driver node 16-32GB RAM
#   - Spark chỉ dùng cho LDA vì LDA gensim không scale → Spark MLlib
# ============================================================================

def task_bertopic_load_model(**context) -> None:
    """
    Task 1: Load BERTopic model từ HDFS → lưu path vào XCom.

    Không load cả model vào XCom (quá lớn) — chỉ lưu đường dẫn tmp dir.
    Để model sống trong driver memory, dùng một PythonOperator duy nhất
    chạy toàn bộ pipeline (xem task_bertopic_run_pipeline bên dưới).
    """
    logger.info("[BERTopic] Checking model path...")
    model_path = (
        HDFS_MODEL_PATH if not USE_LOCAL else LOCAL_BERTOPIC_MODEL_PATH
    )
    logger.info(f"Model path: {model_path}")
    context["ti"].xcom_push(key="model_path", value=model_path)


def task_bertopic_run_pipeline() -> None:
    """
    Task 2 (chính): Load model → đọc staged data → inference → export.

    Chạy toàn bộ trong 1 task để tránh phải serialize model qua XCom.
    PythonOperator giữ process sống suốt runtime — model load 1 lần.
    """
    import sys as _sys
    _sys.path.insert(
        0,
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    from spark_jobs.bertopic_inference_job import main as bertopic_main

    bertopic_main(
        # Cùng nguồn với LDA: stg_posts_core do cleaning_job tạo ra
        input_path=STAGED_LOCAL_PATH if USE_LOCAL else STAGED_HDFS_PATH,
        model_path=None,
        output_path=None,
        local=USE_LOCAL,
        window_days=7,     # chạy Chủ nhật → lấy 7 ngày tích lũy trong tuần
    )
    logger.info("[BERTopic] Weekly inference pipeline done.")


def task_save_bertopic_to_clickhouse() -> None:
    """
    Task 3: Push BERTopic parquet results → ClickHouse tech_radar.

    Reads output/bertopic_inference/{post_topic_assignment,topics}.parquet
    and inserts into stg_post_topics + stg_topics.
    Runs after task_bertopic_run_pipeline so parquet files are already written.
    """
    import sys as _sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _sys.path.insert(0, root)
    from scripts.save_topics_to_ch import main as save_to_ch

    # Inside Docker the service hostname is always "clickhouse", regardless of USE_LOCAL
    ch_host = os.environ.get("CLICKHOUSE_HOST", "clickhouse")
    save_to_ch(
        input_dir="output/bertopic_inference",
        host=ch_host,
        port=CLICKHOUSE_PORT,
    )
    logger.info("[BERTopic] Topics pushed to ClickHouse.")


# ── Cấu hình path riêng cho BERTopic ──
HDFS_MODEL_PATH: str = f"{HDFS_USER_DIR}/models/bertopic/bertopic_model"
LOCAL_BERTOPIC_MODEL_PATH: str = "output/task3.1_bertopic/output/bertopic_model"

with DAG(
    dag_id="bertopic_weekly_inference",
    default_args=default_args,
    description=(
        "BERTopic weekly inference — chạy Chủ nhật 3h sáng. "
        "Load model PhoBERT từ HDFS, inference 7 ngày dữ liệu, "
        "export stg_post_topics + stg_topics. Member 3 — Task 3.x"
    ),
    schedule_interval="0 3 * * 0",        # Chủ nhật 3:00 AM
    catchup=False,
    max_active_runs=1,
    tags=["member3", "bertopic", "weekly", "inference"],
) as bertopic_dag:

    bt_start = DummyOperator(task_id="bertopic_start")

    # Task 1: Kiểm tra model tồn tại (lightweight check)
    bt_check_model = PythonOperator(
        task_id="check_model_path",
        python_callable=task_bertopic_load_model,
        provide_context=True,
    )

    # Task 2: Chạy toàn bộ inference pipeline → parquet
    bt_run = PythonOperator(
        task_id="run_inference_pipeline",
        python_callable=task_bertopic_run_pipeline,
        execution_timeout=timedelta(hours=3),   # tối đa 3h cho 200K docs
    )

    # Task 3a: Push parquet kết quả → ClickHouse tech_radar
    bt_save_ch = PythonOperator(
        task_id="save_topics_to_clickhouse",
        python_callable=task_save_bertopic_to_clickhouse,
    )

    # Task 3b: Ghi post_topics → HDFS silver layer (cùng path với LDA, khác partition)
    bt_save_hdfs = PythonOperator(
        task_id="save_post_topics_to_hdfs",
        python_callable=task_save_post_topics_to_hdfs,
        op_kwargs={
            "model_type": "bertopic",
            "local_src": "output/bertopic_inference/post_topic_assignment.parquet",
            "hdfs_src": f"{HDFS_USER_DIR}/results/bertopic/post_topics",
        },
        provide_context=True,
    )

    bt_end = DummyOperator(task_id="bertopic_end")

    bt_start >> bt_check_model >> bt_run >> [bt_save_ch, bt_save_hdfs] >> bt_end
