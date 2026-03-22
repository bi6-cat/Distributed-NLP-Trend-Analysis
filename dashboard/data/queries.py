"""
ClickHouse query layer for the Tech Trend Radar dashboard.

If ClickHouse is unreachable the module falls back to synthetic demo data
so the dashboard can always be previewed.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import math
import random
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
_CONN_PARAMS: dict = {
    "host": "localhost",
    "port": 8123,
    "database": "dwh_prod",
    "user": "admin",
    "password": "clickhouse_secret",
}


def _get_client():
    """Return a clickhouse-connect client or None when unavailable."""
    try:
        import clickhouse_connect  # noqa: F811
        return clickhouse_connect.get_client(**_CONN_PARAMS)
    except Exception:
        return None


def _query(sql: str) -> Optional[pd.DataFrame]:
    """Execute *sql* and return a DataFrame, or None on failure."""
    client = _get_client()
    if client is None:
        return None
    try:
        result = client.query_df(sql)
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Demo-data generators (deterministic, seeded)
# ---------------------------------------------------------------------------
_RNG = np.random.RandomState(42)

DEMO_SOURCES = ["voz", "vnexpress", "youtube", "tinhte"]
DEMO_TOPICS = [
    (1, "iPhone 16 Pro Max", ["iphone", "apple", "camera", "giá", "review"]),
    (2, "NVIDIA RTX 5090", ["gpu", "nvidia", "rtx", "gaming", "hiệu năng"]),
    (3, "AI trong giáo dục", ["ai", "chatgpt", "giáo dục", "học sinh", "trường"]),
    (4, "VinFast VF9", ["vinfast", "ô tô điện", "vf9", "giá", "đánh giá"]),
    (5, "Samsung Galaxy S26", ["samsung", "galaxy", "s26", "snapdragon", "ai"]),
    (6, "Laptop sinh viên 2026", ["laptop", "sinh viên", "giá rẻ", "ram", "ssd"]),
    (7, "5G Việt Nam", ["5g", "viettel", "mobifone", "tốc độ", "phủ sóng"]),
    (8, "Robot hút bụi", ["robot", "hút bụi", "xiaomi", "ecovacs", "giá"]),
    (9, "Bảo mật dữ liệu", ["bảo mật", "hack", "lộ dữ liệu", "vpn", "password"]),
    (10, "Gemini vs ChatGPT", ["gemini", "chatgpt", "so sánh", "ai", "google"]),
    (11, "Đồng hồ thông minh", ["smartwatch", "apple watch", "samsung", "garmin", "sức khỏe"]),
    (12, "Máy ảnh mirrorless", ["máy ảnh", "sony", "canon", "mirrorless", "nhiếp ảnh"]),
]


def _demo_hours(days: int = 30) -> list[_dt.datetime]:
    """Generate hourly buckets for the last N days."""
    end = _dt.datetime.now().replace(minute=0, second=0, microsecond=0)
    return [end - _dt.timedelta(hours=i) for i in range(days * 24)]


def _make_fct_topic_activity(days: int = 30) -> pd.DataFrame:
    hours = _demo_hours(days)
    rows = []
    for tid, label, kw in DEMO_TOPICS:
        base_mentions = _RNG.randint(5, 60)
        for src in DEMO_SOURCES:
            src_factor = {"voz": 1.2, "vnexpress": 0.7, "youtube": 1.0, "tinhte": 0.5}[src]
            prev_mc = 0
            for h in reversed(hours):
                noise = _RNG.normal(0, 3)
                hour_factor = 1.0 + 0.3 * math.sin(h.hour / 24 * 2 * math.pi)
                mc = max(1, int(base_mentions * src_factor * hour_factor + noise))
                vel = mc
                acc = mc - prev_mc
                prev_mc = mc

                pos = _RNG.binomial(mc, 0.55)
                neg = _RNG.binomial(mc - pos, 0.4)
                neu = mc - pos - neg
                eng = int(mc * _RNG.uniform(1.5, 8.0))
                eng_norm = round(_RNG.uniform(0, 1), 4)
                ts = round(0.40 * vel + 0.30 * acc + 0.30 * eng_norm * 100, 2)

                neg_ratio = round(neg / max(mc, 1), 4)
                pos_ratio = round(pos / max(mc, 1), 4)

                rows.append({
                    "topic_id": tid,
                    "source": src,
                    "hour_bucket": h,
                    "bucket_date": h.date(),
                    "source_type": {"voz": "forum", "tinhte": "forum",
                                    "vnexpress": "news", "youtube": "video"}[src],
                    "topic_label": label,
                    "mention_count": mc,
                    "unique_authors": max(1, int(mc * 0.7)),
                    "engagement_sum": eng,
                    "velocity": vel,
                    "acceleration": acc,
                    "engagement_normalized": eng_norm,
                    "trend_score": ts,
                    "trend_rank": 0,
                    "pos_count": pos,
                    "neg_count": neg,
                    "neu_count": neu,
                    "neg_ratio": neg_ratio,
                    "pos_ratio": pos_ratio,
                    "avg_sentiment_confidence": round(_RNG.uniform(0.6, 0.95), 3),
                    "neg_ratio_24h_avg": round(neg_ratio + _RNG.normal(0, 0.02), 4),
                    "mention_7d_avg": round(base_mentions * src_factor, 2),
                    "mention_7d_stddev": round(abs(_RNG.normal(4, 1.5)), 2),
                    "volume_zscore": round(_RNG.normal(0, 1.2), 3),
                    "computed_at": _dt.datetime.now(),
                })
    df = pd.DataFrame(rows)
    # Assign trend_rank per hour_bucket
    df["trend_rank"] = df.groupby("hour_bucket")["trend_score"] \
        .rank(ascending=False, method="first").astype(int)
    return df


def _make_fct_crisis_events() -> pd.DataFrame:
    rows = []
    severities = ["HIGH", "MEDIUM", "LOW"]
    for i in range(22):
        sev = _RNG.choice(severities, p=[0.15, 0.35, 0.50])
        det = _dt.datetime.now() - _dt.timedelta(hours=_RNG.randint(1, 720))
        tids = _RNG.choice([t[0] for t in DEMO_TOPICS], size=_RNG.randint(1, 3),
                           replace=False).tolist()
        labels = [DEMO_TOPICS[t - 1][1] for t in tids]
        conds = list(_RNG.choice(["neg_spike", "volume_anomaly", "isolation_forest"],
                                 size=_RNG.randint(2, 3), replace=False))
        rows.append({
            "event_id": hashlib.md5(f"evt-{i}".encode()).hexdigest()[:12],
            "detected_at": det,
            "detected_date": det.date(),
            "severity": sev,
            "anomaly_score": round(_RNG.uniform(0.45, 0.98), 2),
            "trigger_conditions": conds,
            "neg_ratio": round(_RNG.uniform(0.25, 0.75), 2),
            "mention_velocity": round(_RNG.uniform(20, 400), 1),
            "evidence_doc_ids": [f"post_{_RNG.randint(1000,9999)}" for _ in range(3)],
            "affected_topics": tids,
            "affected_topic_labels": labels,
            "severity_rank": {"HIGH": 3, "MEDIUM": 2, "LOW": 1}[sev],
        })
    return pd.DataFrame(rows).sort_values("detected_at", ascending=False)


def _make_dim_topics() -> pd.DataFrame:
    rows = []
    for tid, label, kw in DEMO_TOPICS:
        rows.append({
            "topic_id": tid,
            "label": label,
            "top_keywords": kw,
            "coherence_score": round(_RNG.uniform(0.3, 0.7), 3),
            "model_version": "bertopic-v1",
            "total_mentions": _RNG.randint(5000, 80000),
            "unique_authors": _RNG.randint(500, 8000),
            "first_seen": _dt.date.today() - _dt.timedelta(days=_RNG.randint(20, 60)),
            "last_seen": _dt.date.today(),
            "active_days": _RNG.randint(15, 60),
        })
    return pd.DataFrame(rows)


def _make_keyword_freq() -> pd.DataFrame:
    words = [
        "iPhone", "Samsung", "AI", "ChatGPT", "Gemini", "laptop", "GPU",
        "VinFast", "5G", "robot", "camera", "giá rẻ", "đánh giá", "so sánh",
        "pin", "hiệu năng", "bảo mật", "hack", "Snapdragon", "OLED",
        "gaming", "RAM", "SSD", "sạc nhanh", "cập nhật",
    ]
    rows = []
    for w in words:
        for src in DEMO_SOURCES:
            rows.append({
                "keyword": w,
                "source": src,
                "estimated_count": _RNG.randint(50, 5000),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public query functions  (all @st.cache_data)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def get_topic_activity(
    start_date: _dt.date | None = None,
    end_date: _dt.date | None = None,
    sources: list[str] | None = None,
) -> pd.DataFrame:
    """Return fct_topic_activity, filtered."""
    if start_date is None:
        start_date = _dt.date.today() - _dt.timedelta(days=30)
    if end_date is None:
        end_date = _dt.date.today()

    src_clause = ""
    if sources:
        src_list = ", ".join(f"'{s}'" for s in sources)
        src_clause = f"AND source IN ({src_list})"

    sql = f"""
        SELECT *
        FROM fct_topic_activity
        WHERE bucket_date BETWEEN '{start_date}' AND '{end_date}'
        {src_clause}
        ORDER BY hour_bucket DESC, trend_score DESC
    """
    df = _query(sql)
    if df is None:
        df = _make_fct_topic_activity()
        df = df[(df["bucket_date"] >= start_date) & (df["bucket_date"] <= end_date)]
        if sources:
            df = df[df["source"].isin(sources)]
    return df


@st.cache_data(ttl=300, show_spinner=False)
def get_crisis_events(
    window_hours: int = 168,
    severities: list[str] | None = None,
) -> pd.DataFrame:
    """Return fct_crisis_events, filtered."""
    sev_clause = ""
    if severities:
        sev_list = ", ".join(f"'{s}'" for s in severities)
        sev_clause = f"AND severity IN ({sev_list})"

    sql = f"""
        SELECT *
        FROM fct_crisis_events
        WHERE detected_at >= now() - INTERVAL {window_hours} HOUR
        {sev_clause}
        ORDER BY detected_at DESC
    """
    df = _query(sql)
    if df is None:
        df = _make_fct_crisis_events()
        cutoff = _dt.datetime.now() - _dt.timedelta(hours=window_hours)
        df = df[df["detected_at"] >= cutoff]
        if severities:
            df = df[df["severity"].isin(severities)]
    return df


@st.cache_data(ttl=600, show_spinner=False)
def get_dim_topics() -> pd.DataFrame:
    """Return dim_topics."""
    sql = "SELECT * FROM dim_topics ORDER BY total_mentions DESC"
    df = _query(sql)
    if df is None:
        df = _make_dim_topics()
    return df


@st.cache_data(ttl=300, show_spinner=False)
def get_keyword_frequencies() -> pd.DataFrame:
    """Aggregate keyword frequencies (latest window)."""
    sql = """
        SELECT keyword, source,
               sum(estimated_count) AS estimated_count
        FROM dwh_prod.stg_keyword_freq
        WHERE window_start >= now() - INTERVAL 7 DAY
        GROUP BY keyword, source
        ORDER BY estimated_count DESC
        LIMIT 100
    """
    df = _query(sql)
    if df is None:
        df = _make_keyword_freq()
    return df


@st.cache_data(ttl=300, show_spinner=False)
def get_recent_posts(topic_id: int, limit: int = 25) -> pd.DataFrame:
    """Return recent posts for a specific topic."""
    sql = f"""
        SELECT post_id, source, author_name, sentiment_label,
               engagement, title,
               substring(body, 1, 200) AS excerpt,
               created_at
        FROM dwh_prod.stg_posts
        WHERE topic_id = {topic_id}
        ORDER BY created_at DESC
        LIMIT {limit}
    """
    df = _query(sql)
    if df is None:
        # Generate minimal demo posts
        rows = []
        labels = ["positive", "negative", "neutral"]
        for i in range(limit):
            rows.append({
                "post_id": f"p_{topic_id}_{i}",
                "source": random.choice(DEMO_SOURCES),
                "author_name": f"user_{random.randint(100,9999)}",
                "sentiment_label": random.choice(labels),
                "engagement": random.randint(10, 2000),
                "title": f"Bài viết #{i+1} về chủ đề {topic_id}",
                "excerpt": "Lorem ipsum dolor sit amet, đây là nội dung mẫu cho bài viết...",
                "created_at": _dt.datetime.now() - _dt.timedelta(hours=random.randint(1, 168)),
            })
        df = pd.DataFrame(rows)
    return df
