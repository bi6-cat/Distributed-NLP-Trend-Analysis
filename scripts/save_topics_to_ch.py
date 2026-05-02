#!/usr/bin/env python3
"""
scripts/save_topics_to_ch.py — Task 3.5 (Member 3 — Phase 3)

Mục tiêu:
    Cung cấp `topic_id` và `topic_label` cho Member 5 dùng trong dashboard
    bằng cách load kết quả LDA/BERTopic vào ClickHouse.

    Đây là bước HANDOFF chính thức từ M3 → M5:
        M3 chạy script này → ClickHouse nhận data → M5 query từ dashboard

Pipeline:
    1. Đọc stg_topics CSV (output của lda_job.py hoặc bertopic_model.py)
    2. Đọc stg_post_topics CSV (topic assignments per post)
    3. Tạo bảng ClickHouse nếu chưa có (idempotent DDL)
    4. Insert data vào tech_radar.stg_topics + tech_radar.stg_post_topics
    5. Xuất handoff summary CSV cho M5 tham khảo

Schema load vào ClickHouse (data_flow_schema_evolution.md — Stage 5):

    tech_radar.stg_topics:
        topic_id        Int32
        label           String
        top_keywords    Array(String)
        coherence_score Nullable(Float32)
        model_version   String
        created_at      DateTime DEFAULT now()

    tech_radar.stg_post_topics:
        post_id           String
        topic_id          Int32
        topic_probability Float32
        model_type        LowCardinality(String)   -- 'lda' | 'bertopic'
        predicted_at      DateTime
        loaded_at         DateTime DEFAULT now()

Cách chạy:
    # Load LDA output (sau khi chạy lda_job.py)
    python scripts/save_topics_to_ch.py \\
        --topics-csv output/lda/stg_topics.csv \\
        --post-topics-csv output/lda/post_topic_assignment.csv \\
        --host storage-node --port 8123 \\
        --database tech_radar

    # Load BERTopic output (sau khi chạy bertopic_model.py)
    python scripts/save_topics_to_ch.py \\
        --topics-csv output/bertopic_model/stg_topics.csv \\
        --post-topics-csv output/bertopic_model/stg_post_topics.csv \\
        --host storage-node --port 8123 \\
        --database tech_radar

    # Local dev (dùng ClickHouse chạy Docker)
    python scripts/save_topics_to_ch.py \\
        --topics-csv output/lda/stg_topics.csv \\
        --post-topics-csv output/lda/post_topic_assignment.csv \\
        --host localhost --port 8123 \\
        --database tech_radar --dry-run

Phụ thuộc:
    pip install clickhouse-connect pandas

Phụ thuộc từ các thành viên:
    - M2: ClickHouse đã setup trên storage-node (Hard blocker)
    - M3: Đã chạy lda_job.py hoặc bertopic_model.save() để có CSV output
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("save_topics_to_ch")


# ============================================================================
# DDL — CREATE TABLE IF NOT EXISTS
# ============================================================================

DDL_STG_TOPICS = """
CREATE TABLE IF NOT EXISTS {database}.stg_topics (
    topic_id        Int32,
    label           String,
    top_keywords    Array(String),
    coherence_score Nullable(Float32),
    model_version   String,
    created_at      DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(created_at)
ORDER BY (topic_id, model_version)
COMMENT 'Topic lookup table — loaded by M3 (save_topics_to_ch.py)';
"""

DDL_STG_POST_TOPICS = """
CREATE TABLE IF NOT EXISTS {database}.stg_post_topics (
    post_id           String,
    topic_id          Int32,
    topic_probability Float32,
    model_type        LowCardinality(String),
    predicted_at      DateTime,
    loaded_at         DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(loaded_at)
ORDER BY (post_id)
COMMENT 'Post-to-topic assignments — loaded by M3 (save_topics_to_ch.py)';
"""


# ============================================================================
# ĐỌC & VALIDATE CSV
# ============================================================================

def load_stg_topics_csv(filepath: str) -> pd.DataFrame:
    """
    Đọc và validate file stg_topics.csv.

    Cột bắt buộc (theo stg_topics schema):
        topic_id, label, top_keywords, coherence_score, model_version, created_at

    Xử lý đặc biệt:
        - top_keywords: có thể là JSON string ("[\"iphone\",\"pin\"]")
          hoặc pipe-separated ("iphone|pin|camera") — cả hai đều được parse.
        - coherence_score: NULL/None được giữ nguyên.

    Args:
        filepath: Đường dẫn tới file CSV.

    Returns:
        pd.DataFrame đã validated với đúng dtype.

    Raises:
        FileNotFoundError, ValueError nếu CSV không đúng schema.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"stg_topics CSV không tìm thấy: {filepath}")

    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"  Đọc stg_topics: {filepath} — {len(df):,} rows")

    required_cols = {"topic_id", "label", "top_keywords", "model_version"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"stg_topics CSV thiếu cột: {missing}. "
            f"Có: {list(df.columns)}"
        )

    # Parse top_keywords: JSON string → List[str]
    def _parse_keywords(val) -> List[str]:
        if pd.isna(val) or val == "":
            return []
        if isinstance(val, list):
            return [str(w) for w in val]
        val = str(val).strip()
        # Thử JSON trước
        if val.startswith("["):
            try:
                parsed = json.loads(val)
                return [str(w) for w in parsed]
            except json.JSONDecodeError:
                pass
        # Fallback: pipe-separated
        return [w.strip() for w in val.split("|") if w.strip()]

    df["top_keywords"] = df["top_keywords"].apply(_parse_keywords)

    # Ép kiểu
    df["topic_id"] = df["topic_id"].astype("int32")
    df["label"] = df["label"].fillna("").astype(str)
    df["model_version"] = df["model_version"].fillna("lda_v1").astype(str)

    if "coherence_score" not in df.columns:
        df["coherence_score"] = None
    else:
        df["coherence_score"] = pd.to_numeric(df["coherence_score"], errors="coerce")

    if "created_at" not in df.columns:
        df["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(
        f"  stg_topics: {len(df):,} topics — "
        f"models: {df['model_version'].unique().tolist()}"
    )
    return df


def load_stg_post_topics_csv(filepath: str) -> pd.DataFrame:
    """
    Đọc và validate file stg_post_topics CSV.

    Cột bắt buộc (theo stg_post_topics schema):
        post_id, topic_id, topic_probability, model_type, predicted_at

    Tương thích với cả 2 format:
        - output của lda_job.py  : "post_topic_assignment.csv"
        - output của bertopic_model.py: "stg_post_topics.csv"

    Args:
        filepath: Đường dẫn tới file CSV.

    Returns:
        pd.DataFrame đã validated.

    Raises:
        FileNotFoundError, ValueError nếu CSV không đúng schema.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"stg_post_topics CSV không tìm thấy: {filepath}")

    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(f"  Đọc stg_post_topics: {filepath} — {len(df):,} rows")

    required_cols = {"post_id", "topic_id", "topic_probability", "model_type", "predicted_at"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"stg_post_topics CSV thiếu cột: {missing}. "
            f"Có: {list(df.columns)}"
        )

    # Ép kiểu
    df["post_id"] = df["post_id"].astype(str)
    df["topic_id"] = df["topic_id"].astype("int32")
    df["topic_probability"] = pd.to_numeric(df["topic_probability"], errors="coerce").fillna(0.0).astype("float32")
    df["model_type"] = df["model_type"].fillna("lda").astype(str)

    valid_model_types = {"lda", "bertopic"}
    invalid = set(df["model_type"].unique()) - valid_model_types
    if invalid:
        logger.warning(f"  model_type không hợp lệ: {invalid} — mặc định 'lda'")
        df.loc[~df["model_type"].isin(valid_model_types), "model_type"] = "lda"

    # Loại bỏ fake IDs (doc_0, doc_1, ...) nếu có
    fake_mask = df["post_id"].str.match(r"^doc_\d+$", na=False)
    n_fake = fake_mask.sum()
    if n_fake > 0:
        logger.warning(
            f"  ⚠️  {n_fake:,} hàng có fake post_id (doc_0, doc_1, ...) — bị loại bỏ. "
            f"Hãy dùng export_post_topics(real_post_ids, ...) từ bertopic_model.py."
        )
        df = df[~fake_mask].reset_index(drop=True)

    logger.info(
        f"  stg_post_topics: {len(df):,} assignments — "
        f"model_type: {df['model_type'].unique().tolist()}"
    )
    return df


# ============================================================================
# CLICKHOUSE OPERATIONS
# ============================================================================

def get_ch_client(host: str, port: int, database: str, username: str, password: str):
    """
    Tạo ClickHouse client (clickhouse-connect).

    clickhouse-connect dùng HTTP interface (port 8123) —
    không cần driver JAR như JDBC.

    Args:
        host: Hostname hoặc IP của ClickHouse node.
        port: HTTP port (mặc định 8123).
        database: Tên database (tech_radar).
        username: ClickHouse user.
        password: ClickHouse password.

    Returns:
        clickhouse_connect.driver.Client instance.
    """
    try:
        import clickhouse_connect
    except ImportError:
        logger.error(
            "clickhouse-connect chưa cài. Chạy: pip install clickhouse-connect"
        )
        sys.exit(1)

    client = clickhouse_connect.get_client(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
    )
    logger.info(f"  ClickHouse connected: {host}:{port}/{database}")
    return client


def create_tables(client, database: str) -> None:
    """
    Tạo bảng ClickHouse nếu chưa tồn tại (idempotent).

    Dùng CREATE TABLE IF NOT EXISTS → an toàn khi chạy nhiều lần.

    Args:
        client: ClickHouse client.
        database: Tên database.
    """
    for ddl_template, table_name in [
        (DDL_STG_TOPICS, "stg_topics"),
        (DDL_STG_POST_TOPICS, "stg_post_topics"),
    ]:
        ddl = ddl_template.format(database=database)
        client.command(ddl)
        logger.info(f"  ✅ Table ready: {database}.{table_name}")


def insert_stg_topics(client, df: pd.DataFrame, database: str) -> None:
    """
    Insert dữ liệu từ DataFrame vào tech_radar.stg_topics.

    Dùng clickhouse_connect.insert() — gửi binary columnar format,
    nhanh hơn INSERT ... VALUES string nhiều lần.

    Args:
        client: ClickHouse client.
        df: DataFrame với schema stg_topics.
        database: Tên database.
    """
    # Chuẩn bị data theo đúng column order của table
    insert_df = df[["topic_id", "label", "top_keywords", "coherence_score", "model_version"]].copy()

    # ClickHouse nhận top_keywords là list Python → Array(String) tự động
    # qua clickhouse-connect

    n = len(insert_df)
    logger.info(f"  Inserting {n:,} rows → {database}.stg_topics ...")
    client.insert_df(
        f"{database}.stg_topics",
        insert_df,
        column_names=["topic_id", "label", "top_keywords", "coherence_score", "model_version"],
    )
    logger.info(f"  ✅ stg_topics: {n:,} rows inserted")


def insert_stg_post_topics(client, df: pd.DataFrame, database: str) -> None:
    """
    Insert dữ liệu từ DataFrame vào tech_radar.stg_post_topics.

    ReplacingMergeTree(loaded_at) — nếu M3 rerun, bản mới nhất sẽ
    thay thế bản cũ theo (post_id) ORDER BY key.

    Args:
        client: ClickHouse client.
        df: DataFrame với schema stg_post_topics.
        database: Tên database.
    """
    insert_df = df[["post_id", "topic_id", "topic_probability", "model_type", "predicted_at"]].copy()

    n = len(insert_df)
    logger.info(f"  Inserting {n:,} rows → {database}.stg_post_topics ...")
    client.insert_df(
        f"{database}.stg_post_topics",
        insert_df,
        column_names=["post_id", "topic_id", "topic_probability", "model_type", "predicted_at"],
    )
    logger.info(f"  ✅ stg_post_topics: {n:,} rows inserted")


def verify_load(client, database: str) -> None:
    """
    Kiểm tra nhanh số row sau khi insert (smoke test).

    Args:
        client: ClickHouse client.
        database: Tên database.
    """
    for table in ["stg_topics", "stg_post_topics"]:
        result = client.query(f"SELECT count() FROM {database}.{table}")
        count = result.first_row[0]
        logger.info(f"  Verify {database}.{table}: {count:,} rows total")


# ============================================================================
# HANDOFF SUMMARY CHO MEMBER 5
# ============================================================================

def export_handoff_summary(
    topics_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Xuất handoff summary CSV cho Member 5 dùng trong dashboard.

    File này giúp M5 biết chính xác:
        - Có bao nhiêu topic
        - topic_id và topic_label tương ứng
        - top_keywords để hiển thị word cloud
        - model_version để phân biệt LDA/BERTopic

    Format CSV (đơn giản, M5 có thể đọc trực tiếp hoặc import vào dashboard):
        topic_id, label, top_3_keywords, top_keywords_json, model_version

    Args:
        topics_df: DataFrame stg_topics đã load.
        output_path: Đường dẫn file handoff CSV.
    """
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    rows = []
    for _, row in topics_df.iterrows():
        kws: List[str] = row["top_keywords"] if isinstance(row["top_keywords"], list) else []
        top3 = " | ".join(kws[:3])
        rows.append({
            "topic_id": int(row["topic_id"]),
            "label": str(row["label"]),
            "top_3_keywords": top3,
            "top_keywords_json": json.dumps(kws, ensure_ascii=False),
            "coherence_score": row.get("coherence_score", None),
            "model_version": str(row["model_version"]),
        })

    handoff_df = pd.DataFrame(
        rows,
        columns=["topic_id", "label", "top_3_keywords", "top_keywords_json", "coherence_score", "model_version"],
    )
    handoff_df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(
        f"  ✅ Handoff summary → {output_path} "
        f"({len(handoff_df):,} topics) — dành cho Member 5"
    )


# ============================================================================
# DRY-RUN MODE (không cần ClickHouse)
# ============================================================================

def dry_run(topics_df: pd.DataFrame, post_topics_df: pd.DataFrame) -> None:
    """
    In preview data mà không cần kết nối ClickHouse thực.

    Dùng để validate CSV trước khi push lên cluster.

    Args:
        topics_df: DataFrame stg_topics.
        post_topics_df: DataFrame stg_post_topics.
    """
    logger.info("=" * 60)
    logger.info("DRY-RUN MODE — không insert vào ClickHouse")
    logger.info("=" * 60)

    logger.info(f"\n[stg_topics] {len(topics_df):,} rows")
    logger.info(f"  Columns: {topics_df.columns.tolist()}")
    logger.info(f"  dtypes:\n{topics_df.dtypes}")
    logger.info("\n  Preview (5 rows):")
    preview_topics = topics_df[["topic_id", "label", "model_version"]].head(5).to_string(index=False)
    for line in preview_topics.splitlines():
        logger.info(f"    {line}")

    logger.info(f"\n[stg_post_topics] {len(post_topics_df):,} rows")
    logger.info(f"  Columns: {post_topics_df.columns.tolist()}")
    logger.info(f"  dtypes:\n{post_topics_df.dtypes}")
    logger.info("\n  Preview (5 rows):")
    preview_posts = post_topics_df.head(5).to_string(index=False)
    for line in preview_posts.splitlines():
        logger.info(f"    {line}")

    logger.info("\n✅ Dry-run complete — data hợp lệ, sẵn sàng insert")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Task 3.5 (Member 3) — Load topic clusters vào ClickHouse "
            "để Member 5 dùng trong dashboard."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input CSVs
    parser.add_argument(
        "--topics-csv",
        type=str,
        default="output/lda/stg_topics.csv",
        help="Đường dẫn stg_topics CSV (output của lda_job.py hoặc bertopic_model.py).",
    )
    parser.add_argument(
        "--post-topics-csv",
        type=str,
        default="output/lda/post_topic_assignment.csv",
        help="Đường dẫn stg_post_topics CSV (output của lda_job.py hoặc bertopic_model.py).",
    )

    # ClickHouse connection
    parser.add_argument(
        "--host", type=str, default="localhost",
        help="ClickHouse hostname hoặc IP.",
    )
    parser.add_argument(
        "--port", type=int, default=8123,
        help="ClickHouse HTTP port.",
    )
    parser.add_argument(
        "--database", type=str, default="tech_radar",
        help="ClickHouse database name.",
    )
    parser.add_argument(
        "--username", type=str, default="default",
        help="ClickHouse username.",
    )
    parser.add_argument(
        "--password", type=str, default="",
        help="ClickHouse password.",
    )

    # Options
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chỉ validate và preview data, không insert vào ClickHouse.",
    )
    parser.add_argument(
        "--skip-post-topics", action="store_true",
        help="Chỉ load stg_topics, bỏ qua stg_post_topics (hữu ích khi chỉ update topic labels).",
    )
    parser.add_argument(
        "--handoff-output",
        type=str,
        default="output/handoff/topic_summary_for_m5.csv",
        help="Đường dẫn xuất handoff summary CSV cho Member 5.",
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """
    Entry point — chạy toàn bộ pipeline load topics vào ClickHouse.

    Pipeline:
        1. Đọc & validate stg_topics CSV
        2. Đọc & validate stg_post_topics CSV (trừ khi --skip-post-topics)
        3. Kết nối ClickHouse (trừ khi --dry-run)
        4. CREATE TABLE IF NOT EXISTS (idempotent)
        5. Insert stg_topics
        6. Insert stg_post_topics (trừ khi --skip-post-topics)
        7. Verify row count (smoke test)
        8. Xuất handoff summary CSV cho M5
    """
    args = parse_args()

    logger.info("=" * 60)
    logger.info("SAVE TOPICS TO CLICKHOUSE — Member 3 Task 3.5")
    logger.info(f"  topics_csv     : {args.topics_csv}")
    logger.info(f"  post_topics_csv: {args.post_topics_csv}")
    if not args.dry_run:
        logger.info(f"  clickhouse     : {args.host}:{args.port}/{args.database}")
    else:
        logger.info("  mode           : DRY-RUN (no DB write)")
    logger.info("=" * 60)

    # ── Bước 1: Đọc stg_topics ───────────────────────────────────
    logger.info("\n[1/5] Đọc stg_topics CSV...")
    topics_df = load_stg_topics_csv(args.topics_csv)

    # ── Bước 2: Đọc stg_post_topics ──────────────────────────────
    post_topics_df = None
    if not args.skip_post_topics:
        logger.info("\n[2/5] Đọc stg_post_topics CSV...")
        post_topics_df = load_stg_post_topics_csv(args.post_topics_csv)
    else:
        logger.info("\n[2/5] Skip stg_post_topics (--skip-post-topics).")

    # ── Dry-run: chỉ preview, không insert ───────────────────────
    if args.dry_run:
        dry_run(topics_df, post_topics_df if post_topics_df is not None else pd.DataFrame())
        logger.info("\n[Dry-run] Xuất handoff summary...")
        export_handoff_summary(topics_df, args.handoff_output)
        logger.info("\nDone. Dùng --dry-run=False để insert vào ClickHouse.")
        return

    # ── Bước 3: Kết nối ClickHouse ───────────────────────────────
    logger.info("\n[3/5] Kết nối ClickHouse...")
    client = get_ch_client(
        host=args.host,
        port=args.port,
        database=args.database,
        username=args.username,
        password=args.password,
    )

    # ── Bước 4: CREATE TABLE IF NOT EXISTS ────────────────────────
    logger.info("\n[4/5] Tạo bảng (idempotent)...")
    create_tables(client, args.database)

    # ── Bước 5 & 6: Insert data ───────────────────────────────────
    logger.info("\n[5/5] Insert data...")
    insert_stg_topics(client, topics_df, args.database)

    if post_topics_df is not None and len(post_topics_df) > 0:
        insert_stg_post_topics(client, post_topics_df, args.database)
    elif not args.skip_post_topics:
        logger.warning(
            "  ⚠️  stg_post_topics rỗng sau khi lọc fake IDs. "
            "Hãy chạy lda_job.py hoặc bertopic_model.export_post_topics() "
            "với post_ids thực từ HDFS."
        )

    # ── Verify ────────────────────────────────────────────────────
    logger.info("\n[Verify] Kiểm tra row count...")
    verify_load(client, args.database)

    # ── Handoff summary cho M5 ────────────────────────────────────
    logger.info("\n[Handoff] Xuất summary cho Member 5...")
    export_handoff_summary(topics_df, args.handoff_output)

    logger.info("\n" + "=" * 60)
    logger.info("✅ TASK 3.5 COMPLETE!")
    logger.info(f"  stg_topics   : {len(topics_df):,} topics → {args.database}.stg_topics")
    if post_topics_df is not None:
        logger.info(f"  stg_post_topics: {len(post_topics_df):,} rows → {args.database}.stg_post_topics")
    logger.info(f"  Handoff CSV  : {args.handoff_output}")
    logger.info("=" * 60)
    logger.info(
        "\n📢 HANDOFF M3 → M5:\n"
        "  Ping Member 5 với file handoff CSV và thông báo:\n"
        f"    ClickHouse table: {args.database}.stg_topics\n"
        "    Query: SELECT topic_id, label, top_keywords FROM tech_radar.stg_topics\n"
        "    Columns cho dashboard: topic_id (Int32), label (String), top_keywords (Array)\n"
    )


if __name__ == "__main__":
    main()
