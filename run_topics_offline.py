"""Run BERTopic locally on a GPU PC and write pipeline-compatible Parquet.

This script is intentionally independent from Spark, HDFS, Airflow, and
ClickHouse. Copy ``stg_posts_core`` to a GPU machine, run topic modeling here,
then copy ``stg_post_topics`` and ``stg_topics`` back to the pipeline server.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
OUTPUT_POST_TOPIC_COLUMNS = [
    "post_id",
    "topic_id",
    "topic_probability",
    "model_type",
    "predicted_at",
]
OUTPUT_TOPIC_COLUMNS = [
    "topic_id",
    "label",
    "top_keywords",
    "coherence_score",
    "model_version",
    "created_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline GPU BERTopic training/inference for stg_posts_core."
    )
    parser.add_argument("--input", required=True, help="Path to stg_posts_core parquet directory or file.")
    parser.add_argument("--output", required=True, help="Output root containing stg_post_topics and stg_topics.")
    parser.add_argument("--model-out", default=None, help="Optional directory to save the trained BERTopic model.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None, help="Default: auto detect.")
    parser.add_argument("--batch-size", type=int, default=32, help="SentenceTransformer encode batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument("--sample", type=int, default=None, help="Optional random sample size for faster training.")
    parser.add_argument("--max-text-len", type=int, default=800)
    parser.add_argument("--min-text-len", type=int, default=20)
    parser.add_argument("--min-topic-size", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--top-n-words", type=int, default=12)
    parser.add_argument("--nr-topics", default="auto", help="BERTopic nr_topics value. Use 0/none to disable.")
    parser.add_argument("--top-n-topics", type=int, default=80, help="Keep biggest N topics; 0 disables filtering.")
    parser.add_argument("--model-version", default="bertopic_bge_m3_v1")
    parser.add_argument("--part-size", type=int, default=100_000, help="Rows per post-topic parquet part.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true", help="Use already-downloaded HF models only.")
    return parser.parse_args()


def _clean_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _first_non_empty(row, columns: Iterable[str]) -> str:
    for column in columns:
        if column in row:
            value = str(row.get(column) or "").strip()
            if value and value.lower() != "nan":
                return value
    return ""


def build_topic_text(row, max_text_len: int) -> str:
    topic_text = _first_non_empty(row, ("topic_text",))
    title = _first_non_empty(row, ("title",))
    body = _first_non_empty(row, ("segmented_text", "body", "clean_text"))

    text = topic_text or body
    if title and title.lower() not in text.lower():
        text = f"{title}. {text}"
    return _clean_space(text)[:max_text_len]


def load_input(args: argparse.Namespace):
    import pandas as pd

    df = pd.read_parquet(args.input)
    if args.limit is not None:
        df = df.head(args.limit).copy()
    if args.sample is not None and len(df) > args.sample:
        df = df.sample(n=args.sample, random_state=args.seed).copy()

    if "post_id" not in df.columns:
        raise ValueError("Input dataset must contain post_id.")

    df["post_id"] = df["post_id"].fillna("").astype(str)
    df["topic_text_offline"] = df.apply(lambda row: build_topic_text(row, args.max_text_len), axis=1)
    df = df[(df["post_id"] != "") & (df["topic_text_offline"].str.len() >= args.min_text_len)].copy()
    df = df.drop_duplicates(subset=["post_id"], keep="first")
    return df[["post_id", "topic_text_offline"]]


def parse_nr_topics(value: str):
    if value.lower() in {"0", "none", "false", "no", "off"}:
        return None
    if value.lower() == "auto":
        return "auto"
    return int(value)


def fit_bertopic(documents: list[str], args: argparse.Namespace):
    import torch
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model = SentenceTransformer(args.embedding_model, device=device_name)

    embeddings = embedding_model.encode(
        documents,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    umap_model = UMAP(
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=args.seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_topic_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    # Thêm bộ lọc stop words cực mạnh để loại bỏ rác hệ thống và văn phong diễn đàn Voz
    stop_words = [
        # System, English noise & hash tags
        "friend", "said", "click", "expand", "to_expand", "attachments", "jpg", "jpeg", "png", "screenshot", 
        "sent", "from", "sent_from", "txt", "com", "www", "link", "your", "far", "away", "last",
        "screenshot_2024", "screenshot_2025", "screenshot_2026", "com_shopee", "việt_nam", "vietnam", "vn",
        "tel", "vesa", "ras", "vilog", "using", "_com_shopee", "com", "search", "app", "_com",
        # Months
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        # Voz specific & junk
        "vozfapp", "vozvnapp", "deptraitube", "aquarius", "gadibo", "subcribe", "yuuki", "shopee", "master",
        "vozer", "voz", "thím", "mừng_thím", "trư", "hichic", "làng", "đại_hiệp",
        # VN forum generic / Stop words
        "đồng_ý", "cảm_ơn", "quyết_định", "cuối_cùng", "kết_bạn", "chúc_mừng", "đừng", "tao", "hả", 
        "nha", "nhá", "mai", "thử", "ổn", "anh_em", "anh_chị_em", "múc", "nè", "hẹ", "final", 
        "phân_vân", "xong", "khuyên", "cố", "dấu", "ý_kiến", "bac", "cám", "ơn", "cám_ơn", "khong", 
        "nãy", "đọc", "post", "kỹ", "khúc", "vấn_đề", "chờ", "đợi", "chờ_đợi", "mãi", "triệu", "đồng", 
        "gửi", "bàn", "ngã", "sấp", "ngã_sấp", "hả_friend", "á_friend", "chuyên_nghiệp", "máy", "chế", "độ",
        "giá", "rẻ", "tiền", "cục", "sợ", "kiểu", "bình_thường", "thằng", "câu", "hôm", "coi", "tui",
        "nhấn", "chê", "khen", "ảo", "bây_giờ", "sài", "tóp", "thả", "tim", "hàng", "áp_đặt", "phán",
        "đường_dẫn", "hư", "sợ_hãi", "tặng", "deal", "giếng", "ốc", "khơi", "vạn", "bình_luận", "mang_tiếng"
    ]

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.6,
        token_pattern=r"(?u)\b[a-zA-Z_À-ỹ]{3,}\b", # Bỏ các từ <= 2 ký tự (như 26, sờ) và hash số/kí tự đặc biệt
        stop_words=stop_words,
    )

    ctfidf_model = ClassTfidfTransformer(
        reduce_frequent_words=True,
        bm25_weighting=True
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        top_n_words=args.top_n_words,
        nr_topics=parse_nr_topics(args.nr_topics),
        calculate_probabilities=True,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(documents, embeddings=embeddings)
    return topic_model, topics, probs


def topic_probabilities(topics: list[int], probs) -> list[float]:
    import numpy as np

    if probs is not None and hasattr(probs, "shape") and len(probs.shape) == 2:
        return np.max(probs, axis=1).astype(float).tolist()
    return [0.0 if int(topic) == -1 else 1.0 for topic in topics]


def build_outputs(topic_model, post_ids: list[str], topics: list[int], probs, args: argparse.Namespace):
    import pandas as pd

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    post_topics_df = pd.DataFrame(
        {
            "post_id": post_ids,
            "topic_id": [int(topic) for topic in topics],
            "topic_probability": topic_probabilities(topics, probs),
            "model_type": "bertopic",
            "predicted_at": now,
        },
        columns=OUTPUT_POST_TOPIC_COLUMNS,
    )
    post_topics_df = post_topics_df.astype(
        {
            "post_id": "string",
            "topic_id": "int32",
            "topic_probability": "float32",
            "model_type": "string",
        }
    )

    topic_ids = sorted(topic_id for topic_id in set(post_topics_df["topic_id"].tolist()) if topic_id != -1)
    topic_rows = []
    for topic_id in topic_ids:
        words = [word for word, _ in (topic_model.get_topic(topic_id) or [])]
        topic_rows.append(
            {
                "topic_id": int(topic_id),
                "label": "_".join(words[:3]),
                "top_keywords": words,
                "coherence_score": None,
                "model_version": args.model_version,
                "created_at": now,
            }
        )
    topics_df = pd.DataFrame(topic_rows, columns=OUTPUT_TOPIC_COLUMNS)
    if topics_df.empty:
        topics_df = pd.DataFrame(
            {
                "topic_id": pd.Series(dtype="int32"),
                "label": pd.Series(dtype="string"),
                "top_keywords": pd.Series(dtype="object"),
                "coherence_score": pd.Series(dtype="float32"),
                "model_version": pd.Series(dtype="string"),
                "created_at": pd.Series(dtype="datetime64[ns]"),
            },
            columns=OUTPUT_TOPIC_COLUMNS,
        )
    else:
        topics_df = topics_df.astype(
            {
                "topic_id": "int32",
                "label": "string",
                "coherence_score": "float32",
                "model_version": "string",
            }
        )

    if args.top_n_topics and args.top_n_topics > 0:
        counts = post_topics_df[post_topics_df["topic_id"] != -1]["topic_id"].value_counts()
        keep_ids = set(counts.head(args.top_n_topics).index.tolist())
        remap_mask = ~post_topics_df["topic_id"].isin(keep_ids) & (post_topics_df["topic_id"] != -1)
        post_topics_df.loc[remap_mask, "topic_id"] = -1
        post_topics_df.loc[remap_mask, "topic_probability"] = 0.0
        topics_df = topics_df[topics_df["topic_id"].isin(keep_ids)].copy()
        print(f"[topics] kept top {len(keep_ids)} topics, remapped {int(remap_mask.sum()):,} rows to -1")

    return post_topics_df, topics_df


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_outputs(post_topics_df: pd.DataFrame, topics_df: pd.DataFrame, args: argparse.Namespace) -> None:
    output_root = Path(args.output)
    post_topics_dir = output_root / "stg_post_topics"
    topics_dir = output_root / "stg_topics"
    reset_dir(post_topics_dir)
    reset_dir(topics_dir)

    if post_topics_df.empty:
        post_topics_df.to_parquet(post_topics_dir / "part-00000.parquet", index=False)
    else:
        for idx, start in enumerate(range(0, len(post_topics_df), args.part_size)):
            part = post_topics_df.iloc[start:start + args.part_size]
            part.to_parquet(post_topics_dir / f"part-{idx:05d}.parquet", index=False)

    topics_df.to_parquet(topics_dir / "part-00000.parquet", index=False)
    print(f"[topics] wrote {len(post_topics_df):,} rows -> {post_topics_dir}")
    print(f"[topics] wrote {len(topics_df):,} topics -> {topics_dir}")


def main() -> None:
    args = parse_args()
    import torch

    started = time.time()
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[topics] input={args.input}")
    print(f"[topics] output={args.output}")
    print(f"[topics] embedding_model={args.embedding_model}")
    print(f"[topics] device={device_name} batch_size={args.batch_size}")
    if device_name == "cuda":
        print(f"[topics] gpu={torch.cuda.get_device_name(0)}")

    df = load_input(args)
    if df.empty:
        raise RuntimeError("No usable rows after filtering topic text.")

    documents = df["topic_text_offline"].tolist()
    post_ids = df["post_id"].tolist()
    print(f"[topics] loaded rows={len(df):,}")

    topic_model, topics, probs = fit_bertopic(documents, args)
    post_topics_df, topics_df = build_outputs(topic_model, post_ids, topics, probs, args)
    write_outputs(post_topics_df, topics_df, args)

    if args.model_out:
        model_path = Path(args.model_out)
        reset_dir(model_path)
        topic_model.save(str(model_path), serialization="safetensors", save_ctfidf=True)
        print(f"[topics] saved model -> {model_path}")

    elapsed = max(time.time() - started, 1e-6)
    assigned = int((post_topics_df["topic_id"] != -1).sum())
    print(f"[topics] assigned={assigned:,}/{len(post_topics_df):,} elapsed={elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
