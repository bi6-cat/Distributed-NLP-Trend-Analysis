"""
spark_jobs/bertopic_inference_job.py — BERTopic Weekly Inference Job

Chạy mỗi Chủ nhật lúc 3h sáng (via Airflow PythonOperator).
KHÔNG dùng Spark vì BERTopic chạy single-machine Python.

Pipeline:
    1. Load BERTopic model từ HDFS (hoặc local)
    2. Đọc staged posts từ HDFS (1 tuần gần nhất)
    3. Preprocess text (title + segmented_text)
    4. Chạy model.transform() để assign topics
    5. Export Parquet theo spec schema → HDFS/local

Output schema (theo docs/data_flow_schema_evolution.md):
    stg_post_topics:  post_id, topic_id, topic_probability, model_type, predicted_at
    stg_topics:       topic_id, label, top_keywords, coherence_score, model_version, created_at

Tại sao không dùng Spark?
    - BERTopic dùng PyTorch + HDBSCAN — không chạy được trên Spark executor
    - Inference load model 1 lần rồi batch-process → đủ nhanh trên driver node
    - 1 tuần data ≈ 50K-200K docs — vừa RAM driver node (16-32GB)

Author: Member 3 (ML Engineer)
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bertopic_inference_job")

# ── Đường dẫn HDFS — khớp với output của cleaning_job.py (M2) ──
# cleaning_job ghi ra: hdfs://namenode:9000/user/zett/staged/stg_posts_core/
# LDA đọc từ:          hdfs://namenode:9000/user/zett/staged/  (cùng nguồn)
HDFS_NAMENODE: str       = os.getenv("HDFS_NAMENODE", "namenode:9000")
WEBHDFS_HOST: str        = os.getenv("WEBHDFS_HOST",  "namenode:9870")
HDFS_STAGED_PATH: str    = "/user/zett/staged/stg_posts_core"   # Parquet từ cleaning_job
HDFS_MODEL_PATH: str     = "/user/zett/models/bertopic/bertopic_model"
HDFS_OUTPUT_TOPICS: str  = "/user/zett/results/bertopic/post_topics/"
HDFS_OUTPUT_TOPIC_DEFS: str = "/user/zett/results/bertopic/topics/"

LOCAL_MODEL_PATH: str   = "output/task3.1_bertopic/output/bertopic_model"
LOCAL_OUTPUT_PATH: str  = "output/bertopic_inference/"
HF_MODEL_REPO: str      = os.getenv("HF_MODEL_REPO", "ABCDHAQ/Bertopic")

BATCH_SIZE: int    = 512   # docs per encode batch
MAX_TEXT_LEN: int  = 500   # ký tự — khớp với giới hạn khi training
MODEL_VERSION: str = "bertopic_v1"
TOP_N_TOPICS: int  = 50    # giữ top-N topic theo số doc, remap phần còn lại → -1


# ============================================================================
# HDFS HELPERS  — dùng WebHDFS REST API, không cần Hadoop CLI
# ============================================================================

def _webhdfs_read_parquet(hdfs_path: str) -> pd.DataFrame:
    """
    Đọc Parquet directory từ HDFS qua WebHDFS HTTP API.

    Không cần Hadoop CLI — chỉ cần requests + pyarrow (đã có trong requirements).
    Tương đương: hdfs dfs -get <hdfs_path> /tmp/... + pd.read_parquet(...)

    Args:
        hdfs_path: Đường dẫn trong HDFS, ví dụ /user/zett/staged/stg_posts_core
    """
    import io
    import requests as _req

    base = f"http://{WEBHDFS_HOST}/webhdfs/v1"

    # List files trong directory
    r = _req.get(f"{base}{hdfs_path}?op=LISTSTATUS", timeout=30)
    r.raise_for_status()
    statuses = r.json()["FileStatuses"]["FileStatus"]
    parquet_files = [s["pathSuffix"] for s in statuses
                     if s["pathSuffix"].endswith(".parquet") and s["type"] == "FILE"]

    if not parquet_files:
        raise FileNotFoundError(f"Không tìm thấy file .parquet tại HDFS: {hdfs_path}")

    logger.info(f"WebHDFS: đọc {len(parquet_files)} parquet file(s) từ {hdfs_path}")
    dfs = []
    for fname in parquet_files:
        file_path = f"{hdfs_path}/{fname}"
        resp = _req.get(f"{base}{file_path}?op=OPEN", allow_redirects=True, timeout=300)
        resp.raise_for_status()
        dfs.append(pd.read_parquet(io.BytesIO(resp.content)))

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"WebHDFS: đọc xong {len(df):,} rows từ HDFS")
    return df


def _webhdfs_upload(local_path: str, hdfs_path: str) -> None:
    """Ghi file từ local lên HDFS qua WebHDFS."""
    import requests as _req

    base = f"http://{WEBHDFS_HOST}/webhdfs/v1"

    parent = os.path.dirname(hdfs_path)
    _req.put(f"{base}{parent}?op=MKDIRS", timeout=30)

    r = _req.put(f"{base}{hdfs_path}?op=CREATE&overwrite=true",
                 allow_redirects=False, timeout=30)
    if r.status_code == 307:
        upload_url = r.headers["Location"]
        with open(local_path, "rb") as f:
            resp = _req.put(upload_url, data=f, timeout=300)
        resp.raise_for_status()
        logger.info(f"WebHDFS upload: {local_path} → {hdfs_path}")
    else:
        raise RuntimeError(f"WebHDFS CREATE redirect failed: {r.status_code}")


def _webhdfs_download_dir(hdfs_dir: str, local_dir: str) -> None:
    """
    Download toàn bộ HDFS directory về local qua WebHDFS.
    Dùng để tải model artifacts (không cần Hadoop CLI).
    """
    import io
    import requests as _req

    base = f"http://{WEBHDFS_HOST}/webhdfs/v1"
    os.makedirs(local_dir, exist_ok=True)

    r = _req.get(f"{base}{hdfs_dir}?op=LISTSTATUS", timeout=30)
    r.raise_for_status()
    statuses = r.json()["FileStatuses"]["FileStatus"]

    for s in statuses:
        name = s["pathSuffix"]
        if s["type"] == "FILE":
            resp = _req.get(f"{base}{hdfs_dir}/{name}?op=OPEN",
                            allow_redirects=True, timeout=300)
            resp.raise_for_status()
            local_file = os.path.join(local_dir, name)
            with open(local_file, "wb") as f:
                f.write(resp.content)
            logger.info(f"WebHDFS download: {hdfs_dir}/{name} → {local_file}")
        elif s["type"] == "DIRECTORY":
            _webhdfs_download_dir(f"{hdfs_dir}/{name}", os.path.join(local_dir, name))


# ============================================================================
# DATA LOADING
# ============================================================================

def load_staged_posts(
    hdfs_path: str,
    window_days: int = 7,
    local: bool = False,
    local_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Đọc staged posts cần inference.

    Local mode: đọc từ stg_posts_core.csv (toàn bộ hoặc filter theo created_at).
    Cluster mode: đọc Parquet từ HDFS, filter 7 ngày gần nhất.

    Returns:
        DataFrame với columns: post_id, clean_text
    """
    if local:
        # local_csv có thể là:
        # - Spark parquet directory (output của cleaning_job): /opt/airflow/data/preprocessed/stg_posts_core/
        # - CSV file (dev/test thủ công):                     data/preprocessed/stg_posts_core.csv
        data_path = local_csv or "data/preprocessed/stg_posts_core.csv"
        if os.path.isdir(data_path):
            logger.info(f"[LOCAL] Loading posts from parquet dir: {data_path}")
            df = pd.read_parquet(data_path)
        else:
            logger.info(f"[LOCAL] Loading posts from {data_path}")
            df = pd.read_csv(data_path, low_memory=False)
    else:
        # Cluster mode: đọc Parquet từ HDFS qua WebHDFS (cùng nguồn với LDA)
        df = _webhdfs_read_parquet(hdfs_path)

        # Filter theo window
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=window_days)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
            df = df[df["created_at"] >= cutoff]
            logger.info(f"Filtered to last {window_days} days: {len(df):,} rows")

    # Build clean_text (giống notebook cell 4)
    def _build_text(row) -> str:
        title = str(row.get("title", "") or "").strip()
        seg   = str(row.get("segmented_text", "") or "").strip()
        body  = str(row.get("body", "") or "").strip()
        text  = seg if seg else body
        if title and title.lower() not in text.lower():
            text = f"{title} {text}"
        return text.strip()[:MAX_TEXT_LEN]

    df["clean_text"] = df.apply(_build_text, axis=1)
    df = df[df["clean_text"].str.len() >= 10].copy()
    df["post_id"] = df["post_id"].astype(str)

    logger.info(f"Ready for inference: {len(df):,} posts")
    return df[["post_id", "clean_text"]]


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_bertopic_model(
    hdfs_model_path: str,
    local: bool = False,
    local_model_path: Optional[str] = None,
    hf_repo: Optional[str] = None,
):
    """
    Load VietnameseBERTopicModel từ HuggingFace Hub, local, hoặc HDFS.

    Thứ tự ưu tiên:
        1. HuggingFace Hub (hf_repo được set) — dùng trên server thật
        2. Local path (local=True) — dùng khi dev/Docker
        3. HDFS (mặc định) — dùng khi cluster thật có HDFS

    Returns:
        VietnameseBERTopicModel instance.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    from models.bertopic_model import VietnameseBERTopicModel

    repo = hf_repo or HF_MODEL_REPO

    if repo:
        logger.info(f"[HuggingFace] Loading BERTopic from {repo}")
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("vinai/phobert-base")
        topic_model = BERTopic.load(repo, embedding_model=embedding_model)
        # Wrap vào VietnameseBERTopicModel để tương thích pipeline
        wrapper = VietnameseBERTopicModel.__new__(VietnameseBERTopicModel)
        wrapper.topic_model = topic_model
        wrapper.topics_ = topic_model.topics_
        wrapper.probs_ = None
        logger.info(f"[HuggingFace] Loaded — {len(topic_model.get_topic_info())} topics")
        return wrapper

    if local:
        model_dir = local_model_path or LOCAL_MODEL_PATH
        logger.info(f"[LOCAL] Loading BERTopic model from {model_dir}")
        return VietnameseBERTopicModel.load(model_dir, verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_model_dir = os.path.join(tmpdir, "bertopic_model")
        _webhdfs_download_dir(hdfs_model_path, local_model_dir)
        logger.info(f"[CLUSTER] Loading BERTopic model from tmp")
        return VietnameseBERTopicModel.load(local_model_dir, verbose=False)


# ============================================================================
# INFERENCE
# ============================================================================

def _filter_top_topics(
    post_topics_df: pd.DataFrame,
    topics_df: pd.DataFrame,
    top_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Giữ top-N topics theo số post, remap phần còn lại về topic_id = -1.

    Lý do: BERTopic v1 sinh ra ~214 topics (rất rộng). Chỉ top-N topics
    có đủ document để có ý nghĩa thống kê khi hiển thị trên dashboard.
    """
    counts = post_topics_df[post_topics_df["topic_id"] != -1]["topic_id"].value_counts()
    keep_ids = set(counts.head(top_n).index.tolist())

    post_topics_df = post_topics_df.copy()
    mask_out = ~post_topics_df["topic_id"].isin(keep_ids) & (post_topics_df["topic_id"] != -1)
    post_topics_df.loc[mask_out, "topic_id"]          = -1
    post_topics_df.loc[mask_out, "topic_probability"] = 0.0

    topics_df = topics_df[topics_df["topic_id"].isin(keep_ids)].copy()

    n_remapped = int(mask_out.sum())
    logger.info(
        f"Top-{top_n} filter: kept {len(keep_ids)} topics, "
        f"remapped {n_remapped:,} posts → outlier (-1)"
    )
    return post_topics_df, topics_df


def run_inference(
    model,
    posts_df: pd.DataFrame,
    top_n_topics: int = TOP_N_TOPICS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chạy BERTopic .transform() và trả về 2 DataFrames theo spec schema.

    Args:
        model:         VietnameseBERTopicModel đã load.
        posts_df:      DataFrame với columns [post_id, clean_text].
        top_n_topics:  Giữ tối đa N topics lớn nhất; phần còn lại → topic -1.
                       Set 0 để tắt filter.

    Returns:
        post_topics_df: stg_post_topics schema
        topics_df: stg_topics schema
    """
    documents = posts_df["clean_text"].tolist()
    post_ids  = posts_df["post_id"].tolist()

    logger.info(f"Running inference on {len(documents):,} documents...")
    topics, probs = model.topic_model.transform(documents)
    logger.info("Inference done.")

    now = datetime.now(tz=timezone.utc)

    # ── post_topic_assignment (stg_post_topics) ──
    if probs is not None and hasattr(probs, "shape") and len(probs.shape) == 2:
        topic_probs = probs.max(axis=1).tolist()
    else:
        topic_probs = [1.0] * len(topics)

    post_topics_df = pd.DataFrame({
        "post_id":           post_ids,
        "topic_id":          [int(t) for t in topics],
        "topic_probability": [float(p) for p in topic_probs],
        "model_type":        "bertopic",
        "predicted_at":      now,
    })

    # ── topics lookup (stg_topics) ──
    topics_rows = []
    for tid in sorted(set(topics)):
        if tid == -1:
            continue
        words = [w for w, _ in model.topic_model.get_topic(tid)]
        topics_rows.append({
            "topic_id":       int(tid),
            "label":          "_".join(words[:3]),
            "top_keywords":   words,
            "coherence_score": None,   # không tính lại coherence khi inference
            "model_version":  MODEL_VERSION,
            "created_at":     now,
        })
    topics_df = pd.DataFrame(topics_rows)

    n_topics   = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(t == -1 for t in topics)
    logger.info(
        f"Raw results: {n_topics} topics, {n_outliers} outliers "
        f"({n_outliers / len(topics) * 100:.1f}%)"
    )

    if top_n_topics and top_n_topics > 0:
        post_topics_df, topics_df = _filter_top_topics(post_topics_df, topics_df, top_n_topics)

    return post_topics_df, topics_df


# ============================================================================
# OUTPUT
# ============================================================================

def export_results(
    post_topics_df: pd.DataFrame,
    topics_df: pd.DataFrame,
    hdfs_output_topics: str,
    hdfs_output_topic_defs: str,
    local: bool = False,
    local_output_path: Optional[str] = None,
) -> None:
    """
    Ghi kết quả ra Parquet → HDFS hoặc local.

    Cluster: ghi parquet vào tmp dir rồi upload HDFS.
    Local: ghi thẳng ra local_output_path.
    """
    if local:
        out_dir = Path(local_output_path or LOCAL_OUTPUT_PATH)
        out_dir.mkdir(parents=True, exist_ok=True)

        pt_path = out_dir / "post_topic_assignment.parquet"
        td_path = out_dir / "topics.parquet"
        post_topics_df.to_parquet(pt_path, index=False)
        topics_df.to_parquet(td_path, index=False)
        logger.info(f"[LOCAL] Wrote {len(post_topics_df):,} rows → {pt_path}")
        logger.info(f"[LOCAL] Wrote {len(topics_df)} topics → {td_path}")

    else:
        # Cluster: ghi parquet ra tmp rồi upload lên HDFS qua WebHDFS
        with tempfile.TemporaryDirectory() as tmpdir:
            pt_local = os.path.join(tmpdir, "post_topic_assignment.parquet")
            td_local = os.path.join(tmpdir, "topics.parquet")

            post_topics_df.to_parquet(pt_local, index=False)
            topics_df.to_parquet(td_local, index=False)

            _webhdfs_upload(pt_local, hdfs_output_topics + "post_topic_assignment.parquet")
            _webhdfs_upload(td_local, hdfs_output_topic_defs + "topics.parquet")

        logger.info(
            f"[CLUSTER] Uploaded {len(post_topics_df):,} post-topic rows "
            f"and {len(topics_df)} topic defs to HDFS"
        )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main(
    input_path: Optional[str] = None,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    local: bool = False,
    window_days: int = 7,
    top_n_topics: int = TOP_N_TOPICS,
) -> None:
    """
    Entry point — dùng được từ cả argparse CLI lẫn Airflow PythonOperator.

    Args:
        input_path:    HDFS staged path hoặc local CSV path.
        model_path:    HDFS model path hoặc local model dir.
        output_path:   HDFS output prefix hoặc local output dir.
        local:         True = local mode (dev/test), False = HDFS cluster.
        window_days:   Số ngày dữ liệu cần inference (default: 7).
        top_n_topics:  Giữ top-N topics lớn nhất (0 = không lọc).
    """
    logger.info("=" * 60)
    logger.info("BERTopic Weekly Inference Job")
    logger.info(f"Mode     : {'LOCAL' if local else 'CLUSTER'}")
    logger.info(f"Window   : last {window_days} days")
    logger.info(f"Top-N    : {top_n_topics if top_n_topics else 'disabled'}")
    logger.info("=" * 60)

    # 1. Load model
    model = load_bertopic_model(
        hdfs_model_path=model_path or HDFS_MODEL_PATH,
        local=local,
        local_model_path=model_path if local else None,
    )

    # 2. Load staged posts
    posts_df = load_staged_posts(
        hdfs_path=input_path or HDFS_STAGED_PATH,
        window_days=window_days,
        local=local,
        local_csv=input_path if local else None,
    )

    if posts_df.empty:
        logger.warning("No posts found — exiting.")
        return

    # 3. Inference
    post_topics_df, topics_df = run_inference(model, posts_df, top_n_topics=top_n_topics)

    # 4. Export
    export_results(
        post_topics_df=post_topics_df,
        topics_df=topics_df,
        hdfs_output_topics=output_path or HDFS_OUTPUT_TOPICS,
        hdfs_output_topic_defs=HDFS_OUTPUT_TOPIC_DEFS,
        local=local,
        local_output_path=output_path if local else None,
    )

    logger.info("BERTopic inference job completed.")


# ── CLI mode ──
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BERTopic weekly inference job"
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="HDFS staged path hoặc local CSV (default: dùng hằng số HDFS_STAGED_PATH)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="HDFS model path hoặc local model dir (default: HDFS_MODEL_PATH)",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="HDFS output prefix hoặc local output dir (default: HDFS_OUTPUT_TOPICS)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Chạy local mode (dev/test, không cần HDFS)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="Số ngày dữ liệu cần inference (default: 7)",
    )
    parser.add_argument(
        "--top-n-topics",
        type=int,
        default=TOP_N_TOPICS,
        help=f"Giữ top-N topics lớn nhất, remap phần còn lại → -1 (default: {TOP_N_TOPICS}, 0 = tắt)",
    )
    args = parser.parse_args()

    main(
        input_path=args.input_path,
        model_path=args.model_path,
        output_path=args.output_path,
        local=args.local,
        window_days=args.window_days,
        top_n_topics=args.top_n_topics,
    )
