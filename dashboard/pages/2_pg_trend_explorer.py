"""
Page 2 — Trend & Sentiment Explorer
Deep-dive into a specific topic's trajectory over time.

Visuals:
  • Topic selector (st.selectbox from dim_topics)
  • KPI row (peak trend, velocity, acceleration, mentions)
  • Velocity line chart (last 7 days)
  • Sentiment stacked area chart (pos_count, neg_count, neu_count)
  • Velocity & Acceleration overlay
  • Word Cloud from top_keywords
  • Topic metadata table
"""
from __future__ import annotations

import ast
import io

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components.sidebar import render_sidebar
from components.kpi_row import render_kpi_row
from components.chart_theme import (
    PRIMARY, POSITIVE, NEGATIVE, NEUTRAL, ACCENT_BLUE,
    ACCENT_PURPLE, SOURCE_COLORS,
    apply_chart_style,
)
from data.queries import get_topic_activity, get_dim_topics

# ── Sidebar ─────────────────────────────────────────────────────
filters = render_sidebar()
df_all = get_topic_activity(filters.start_date, filters.end_date, filters.sources)
dim_topics = get_dim_topics()

# ── Page Header ─────────────────────────────────────────────────
st.markdown("## 🔥 Trend & Sentiment Explorer")
st.caption("Deep-dive into topic trajectories, sentiment evolution & engagement")

# ── Topic Selector (single selectbox from dim_topics) ───────────
topic_options = (
    dim_topics[["topic_id", "label"]]
    .drop_duplicates()
    .sort_values("label")
)
topic_map = dict(zip(topic_options["label"], topic_options["topic_id"]))

if not topic_map:
    st.warning("No topics available in dim_topics.")
    st.stop()

fcol1, fcol2 = st.columns([3, 1], gap="large")

with fcol1:
    selected_label = st.selectbox(
        "Select a Topic",
        options=list(topic_map.keys()),
        index=0,
    )
    selected_id = topic_map[selected_label]

with fcol2:
    granularity = st.radio(
        "Granularity",
        options=["Hourly", "Daily"],
        horizontal=True,
        index=1,
    )

# ── Filter data for selected topic ──────────────────────────────
df = df_all[df_all["topic_id"] == selected_id].copy()

if df.empty:
    st.warning("No data for the selected topic and filters.")
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
        engagement_sum=("engagement_sum", "sum"),
        engagement_normalized=("engagement_normalized", "mean"),
    )
).sort_values(time_col)

# ── KPI Row ─────────────────────────────────────────────────────
render_kpi_row([
    {"label": "🏆 Peak Trend Score",  "value": f"{df_topic_ts['trend_score'].max():.1f}"},
    {"label": "⚡ Avg Velocity",       "value": f"{df_topic_ts['velocity'].mean():.1f} /h"},
    {"label": "📈 Acceleration",       "value": f"{df_topic_ts['acceleration'].mean():+.1f}"},
    {"label": "💬 Total Mentions",     "value": f"{int(df_topic_ts['mention_count'].sum()):,}"},
])

st.divider()

# ── Chart 1: Velocity Over Time (Line Chart) ───────────────────
st.markdown("#### ⚡ Velocity Over Time")

fig_vel = go.Figure()
fig_vel.add_trace(go.Scatter(
    x=df_topic_ts[time_col],
    y=df_topic_ts["velocity"],
    mode="lines+markers",
    name="Velocity",
    line=dict(color=ACCENT_BLUE, width=2.5),
    marker=dict(size=4, color=ACCENT_BLUE),
    hovertemplate="<b>%{x}</b><br>Velocity: %{y:,}<extra></extra>",
))
fig_vel.add_trace(go.Scatter(
    x=df_topic_ts[time_col],
    y=df_topic_ts["acceleration"],
    mode="lines",
    name="Acceleration",
    line=dict(color=ACCENT_PURPLE, width=2, dash="dash"),
    hovertemplate="<b>%{x}</b><br>Acceleration: %{y:+,}<extra></extra>",
))
fig_vel.add_hline(y=0, line_dash="dot", line_color="#4A5568", line_width=1)
fig_vel.update_layout(
    xaxis_title="",
    yaxis_title="Mentions / hour",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)
st.plotly_chart(apply_chart_style(fig_vel, height=400), use_container_width=True)

st.divider()

# ── Chart 2: Sentiment Stacked Area ────────────────────────────
col_sent, col_cloud = st.columns([3, 2], gap="large")

with col_sent:
    st.markdown("#### 💬 Sentiment Distribution Over Time")

    fig_sent = go.Figure()
    for name, col, color in [
        ("Positive", "pos_count", POSITIVE),
        ("Negative", "neg_count", NEGATIVE),
        ("Neutral",  "neu_count", NEUTRAL),
    ]:
        fig_sent.add_trace(go.Scatter(
            x=df_topic_ts[time_col],
            y=df_topic_ts[col],
            mode="lines",
            name=name,
            stackgroup="sentiment",
            line=dict(width=0.5, color=color),
            fillcolor=color.replace(")", ",0.55)").replace("rgb", "rgba")
                if color.startswith("rgb") else f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.55)",
            hovertemplate=f"<b>{name}</b>: %{{y:,}}<extra></extra>",
        ))
    fig_sent.update_layout(
        yaxis_title="Post Count",
        xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(apply_chart_style(fig_sent, height=400), use_container_width=True)

# ── Word Cloud from top_keywords ────────────────────────────────
with col_cloud:
    st.markdown("#### 🏷️ Topic Word Cloud")

    topic_info = dim_topics[dim_topics["topic_id"] == selected_id]
    if not topic_info.empty:
        keywords = topic_info.iloc[0]["top_keywords"]

        # Parse keywords if stored as string representation of a list
        if isinstance(keywords, str):
            try:
                keywords = ast.literal_eval(keywords)
            except (ValueError, SyntaxError):
                keywords = [kw.strip() for kw in keywords.split(",")]

        if isinstance(keywords, (list, tuple)) and len(keywords) > 0:
            # Build frequency dict — assign decreasing weights by position
            freq_dict = {}
            for i, kw in enumerate(keywords):
                kw_str = str(kw).strip()
                if kw_str:
                    freq_dict[kw_str] = max(1, len(keywords) - i) * 10

            try:
                from wordcloud import WordCloud

                wc = WordCloud(
                    width=800,
                    height=500,
                    background_color="#0F0F23",
                    colormap="cool",
                    max_words=30,
                    prefer_horizontal=0.8,
                    margin=10,
                    font_path=None,
                ).generate_from_frequencies(freq_dict)

                # Render to bytes — NO matplotlib dependency
                img_buffer = io.BytesIO()
                wc.to_image().save(img_buffer, format="PNG")
                img_buffer.seek(0)
                st.image(img_buffer, use_container_width=True)

            except ImportError:
                # Fallback: styled tag chips
                _render_keyword_tags(keywords)
        else:
            st.info("No keywords available for this topic.")
    else:
        st.info("Topic not found in dim_topics.")

    # ── Topic Metadata Table ────────────────────────────────────
    if not topic_info.empty:
        row = topic_info.iloc[0]
        st.markdown(f"""
| Metric | Value |
|---|---|
| Coherence Score | `{row.get('coherence_score', 'N/A')}` |
| Total Mentions | `{int(row.get('total_mentions', 0)):,}` |
| First Seen | `{row.get('first_seen', 'N/A')}` |
| Last Seen | `{row.get('last_seen', 'N/A')}` |
| Model | `{row.get('model_version', 'N/A')}` |
        """)

st.divider()

# ── Row 3: Trend Score + Volume (Dual Axis) ─────────────────────
st.markdown("#### 📊 Trend Score & Mention Volume")

fig_dual = make_subplots(specs=[[{"secondary_y": True}]])

fig_dual.add_trace(
    go.Scatter(
        x=df_topic_ts[time_col], y=df_topic_ts["trend_score"],
        name="Trend Score",
        mode="lines",
        line=dict(color=PRIMARY, width=2.5),
    ),
    secondary_y=False,
)
fig_dual.add_trace(
    go.Bar(
        x=df_topic_ts[time_col], y=df_topic_ts["mention_count"],
        name="Mentions",
        opacity=0.25,
        marker_color=ACCENT_BLUE,
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
st.plotly_chart(apply_chart_style(fig_dual, height=380), use_container_width=True)

st.divider()

# ── Row 4: Sentiment by Source (Stacked horizontal bar) ─────────
st.markdown("#### 📡 Sentiment by Source")
src_sent = (
    df.groupby("source", as_index=False)
    .agg(pos=("pos_count", "sum"), neg=("neg_count", "sum"), neu=("neu_count", "sum"))
)

if not src_sent.empty:
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
    st.plotly_chart(apply_chart_style(fig_src_sent, height=320), use_container_width=True)


# ── Helper ──────────────────────────────────────────────────────
def _render_keyword_tags(keywords: list) -> None:
    """Render keywords as styled tag chips (fallback when wordcloud is unavailable)."""
    tags_html = "".join(
        f"<span style='"
        f"background:linear-gradient(135deg, {PRIMARY}33, {ACCENT_PURPLE}33);"
        f"padding:8px 18px;border-radius:20px;margin:6px;display:inline-block;"
        f"border:1px solid {PRIMARY}88;font-size:1rem;'>"
        f"{kw}</span>"
        for kw in keywords
    )
    st.markdown(f"<div style='padding:12px 0'>{tags_html}</div>", unsafe_allow_html=True)
