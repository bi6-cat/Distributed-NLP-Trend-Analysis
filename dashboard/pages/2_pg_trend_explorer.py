"""
Page 2 — Trend Explorer
Deep-dive into a specific topic's trajectory over time.
"""
from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components.sidebar import render_sidebar
from components.kpi_row import render_kpi_row
from components.chart_theme import (
    PRIMARY, POSITIVE, NEGATIVE, NEUTRAL, ACCENT_BLUE,
    ACCENT_PURPLE, SOURCE_COLORS, SENTIMENT_COLORS,
    apply_chart_style,
)
from data.queries import get_topic_activity, get_dim_topics, get_recent_posts

# ── Sidebar ─────────────────────────────────────────────────────
filters = render_sidebar()
df_all = get_topic_activity(filters.start_date, filters.end_date, filters.sources)
dim_topics = get_dim_topics()

# ── Page Header ─────────────────────────────────────────────────
st.markdown("## 🔥 Trend Explorer")
st.caption("Deep-dive into topic trajectories, sentiment evolution & engagement")

# ── Page-level Filters ──────────────────────────────────────────
fcol1, fcol2 = st.columns([3, 1], gap="large")

with fcol1:
    topic_options = (
        dim_topics[["topic_id", "label"]]
        .drop_duplicates()
        .sort_values("label")
    )
    topic_map = dict(zip(topic_options["label"], topic_options["topic_id"]))
    selected_labels = st.multiselect(
        "Select Topics (max 3)",
        options=list(topic_map.keys()),
        default=[list(topic_map.keys())[0]] if topic_map else [],
        max_selections=3,
    )
    selected_ids = [topic_map[lbl] for lbl in selected_labels]

with fcol2:
    granularity = st.radio(
        "Granularity",
        options=["Hourly", "Daily"],
        horizontal=True,
        index=1,
    )

if not selected_ids:
    st.info("👆 Select at least one topic to explore.")
    st.stop()

# ── Filter data ─────────────────────────────────────────────────
df = df_all[df_all["topic_id"].isin(selected_ids)].copy()

if df.empty:
    st.warning("No data for the selected topic(s) and filters.")
    st.stop()

# Aggregate by granularity
if granularity == "Daily":
    time_col = "bucket_date"
    df_ts = (
        df.groupby([time_col, "topic_label", "topic_id", "source"], as_index=False)
        .agg(
            mention_count=("mention_count", "sum"),
            velocity=("velocity", "sum"),
            acceleration=("acceleration", "sum"),
            engagement_sum=("engagement_sum", "sum"),
            engagement_normalized=("engagement_normalized", "mean"),
            trend_score=("trend_score", "max"),
            pos_count=("pos_count", "sum"),
            neg_count=("neg_count", "sum"),
            neu_count=("neu_count", "sum"),
        )
    )
else:
    time_col = "hour_bucket"
    df_ts = df.copy()

# Aggregate across sources for topic-level timeseries
df_topic_ts = (
    df_ts.groupby([time_col, "topic_label", "topic_id"], as_index=False)
    .agg(
        mention_count=("mention_count", "sum"),
        velocity=("velocity", "sum"),
        acceleration=("acceleration", "sum"),
        trend_score=("trend_score", "max"),
        pos_count=("pos_count", "sum"),
        neg_count=("neg_count", "sum"),
        neu_count=("neu_count", "sum"),
        engagement_normalized=("engagement_normalized", "mean"),
    )
)

# ── KPI Row ─────────────────────────────────────────────────────
latest = df_topic_ts.sort_values(time_col, ascending=False).head(len(selected_ids))
render_kpi_row([
    {"label": "🏆 Peak Trend Score",  "value": f"{latest['trend_score'].max():.1f}"},
    {"label": "⚡ Avg Velocity",      "value": f"{latest['velocity'].mean():.1f} /h"},
    {"label": "📈 Acceleration",      "value": f"{latest['acceleration'].mean():+.1f}"},
    {"label": "💬 Total Mentions",    "value": f"{int(df_topic_ts['mention_count'].sum()):,}"},
])

st.divider()

# ── Chart 1: Trend Score + Volume (Dual Axis) ──────────────────
st.markdown("#### 📊 Trend Score & Mention Volume")

fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

for lbl in selected_labels:
    topic_data = df_topic_ts[df_topic_ts["topic_label"] == lbl].sort_values(time_col)
    fig_dual.add_trace(
        go.Scatter(
            x=topic_data[time_col], y=topic_data["trend_score"],
            name=f"{lbl} — Score",
            mode="lines",
            line=dict(width=2.5),
        ),
        secondary_y=False,
    )
    fig_dual.add_trace(
        go.Bar(
            x=topic_data[time_col], y=topic_data["mention_count"],
            name=f"{lbl} — Volume",
            opacity=0.25,
        ),
        secondary_y=True,
    )

fig_dual.update_layout(
    yaxis_title="Trend Score",
    yaxis2_title="Mentions",
    xaxis_title="",
    barmode="overlay",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(apply_chart_style(fig_dual, height=420), use_container_width=True)

st.divider()

# ── Row 2: Sentiment Timeline + Sentiment by Source ─────────────
col_sent, col_src = st.columns([3, 2], gap="large")

with col_sent:
    st.markdown("#### 💬 Sentiment Over Time")
    # Use first selected topic for sentiment breakdown
    t_data = df_topic_ts[df_topic_ts["topic_id"] == selected_ids[0]].sort_values(time_col).copy()
    t_data["total"] = t_data["pos_count"] + t_data["neg_count"] + t_data["neu_count"]
    t_data["pos_pct"] = t_data["pos_count"] / t_data["total"].clip(lower=1) * 100
    t_data["neg_pct"] = t_data["neg_count"] / t_data["total"].clip(lower=1) * 100
    t_data["neu_pct"] = t_data["neu_count"] / t_data["total"].clip(lower=1) * 100

    fig_sent = go.Figure()
    for name, col, color in [
        ("Positive", "pos_pct", POSITIVE),
        ("Negative", "neg_pct", NEGATIVE),
        ("Neutral",  "neu_pct", NEUTRAL),
    ]:
        fig_sent.add_trace(go.Scatter(
            x=t_data[time_col], y=t_data[col],
            mode="lines", name=name,
            stackgroup="one", groupnorm="percent",
            line=dict(width=0.5, color=color),
            fillcolor=color,
        ))
    fig_sent.update_layout(
        yaxis_title="Share %", xaxis_title="",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(apply_chart_style(fig_sent, height=360), use_container_width=True)

with col_src:
    st.markdown("#### 📡 Sentiment by Source")
    src_sent = (
        df[df["topic_id"] == selected_ids[0]]
        .groupby("source", as_index=False)
        .agg(pos=("pos_count", "sum"), neg=("neg_count", "sum"), neu=("neu_count", "sum"))
    )
    fig_src_sent = go.Figure()
    for label_name, col_name, color in [
        ("Positive", "pos", POSITIVE),
        ("Negative", "neg", NEGATIVE),
        ("Neutral",  "neu", NEUTRAL),
    ]:
        fig_src_sent.add_trace(go.Bar(
            y=src_sent["source"], x=src_sent[col_name],
            name=label_name, orientation="h",
            marker_color=color,
        ))
    fig_src_sent.update_layout(
        barmode="stack",
        yaxis_title="", xaxis_title="Posts",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(apply_chart_style(fig_src_sent, height=360), use_container_width=True)

st.divider()

# ── Row 3: Velocity/Acceleration + Keywords ────────────────────
col_vel, col_kw = st.columns([3, 2], gap="large")

with col_vel:
    st.markdown("#### ⚡ Velocity & Acceleration")
    v_data = df_topic_ts[df_topic_ts["topic_id"] == selected_ids[0]].sort_values(time_col)
    fig_vel = go.Figure()
    fig_vel.add_trace(go.Scatter(
        x=v_data[time_col], y=v_data["velocity"],
        name="Velocity (V)", mode="lines",
        line=dict(color=ACCENT_BLUE, width=2),
    ))
    fig_vel.add_trace(go.Scatter(
        x=v_data[time_col], y=v_data["acceleration"],
        name="Acceleration (A)", mode="lines",
        line=dict(color=ACCENT_PURPLE, width=2, dash="dash"),
    ))
    fig_vel.add_hline(y=0, line_dash="dot", line_color="#4A5568", line_width=1)
    fig_vel.update_layout(
        xaxis_title="", yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(apply_chart_style(fig_vel, height=350), use_container_width=True)

with col_kw:
    st.markdown("#### 🏷️ Topic Keywords")
    topic_info = dim_topics[dim_topics["topic_id"] == selected_ids[0]]
    if not topic_info.empty:
        keywords = topic_info.iloc[0]["top_keywords"]
        if isinstance(keywords, str):
            import ast
            try:
                keywords = ast.literal_eval(keywords)
            except (ValueError, SyntaxError):
                keywords = keywords.split(",")

        tags_html = "".join(
            f"<span style='"
            f"background:linear-gradient(135deg, {PRIMARY}33, {ACCENT_PURPLE}33);"
            f"padding:8px 18px;border-radius:20px;margin:6px;display:inline-block;"
            f"border:1px solid {PRIMARY}88;font-size:1rem;'>"
            f"{kw}</span>"
            for kw in keywords
        )
        st.markdown(f"<div style='padding:12px 0'>{tags_html}</div>", unsafe_allow_html=True)

    # Also show topic metadata
    if not topic_info.empty:
        row = topic_info.iloc[0]
        st.markdown(f"""
        | Metric | Value |
        |---|---|
        | Coherence Score | `{row.get('coherence_score', 'N/A')}` |
        | Total Mentions | `{int(row.get('total_mentions', 0)):,}` |
        | Active Days | `{int(row.get('active_days', 0))}` |
        | Model | `{row.get('model_version', 'N/A')}` |
        """)

st.divider()

# ── Row 4: Recent Posts ─────────────────────────────────────────
with st.expander("📝 Recent Posts", expanded=False):
    posts_df = get_recent_posts(selected_ids[0])
    if not posts_df.empty:
        # Add emoji for sentiment
        emoji_map = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
        posts_df["sentiment"] = posts_df["sentiment_label"].map(
            lambda x: f"{emoji_map.get(x, '⚪')} {x.title()}"
        )
        st.dataframe(
            posts_df[["source", "author_name", "sentiment", "engagement", "title", "excerpt"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "source": st.column_config.TextColumn("Source", width="small"),
                "author_name": st.column_config.TextColumn("Author", width="small"),
                "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "engagement": st.column_config.NumberColumn("Engagement", format="%d"),
                "title": st.column_config.TextColumn("Title", width="medium"),
                "excerpt": st.column_config.TextColumn("Excerpt", width="large"),
            },
        )
    else:
        st.info("No posts available for this topic.")
