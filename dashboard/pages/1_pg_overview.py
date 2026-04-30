"""
Page 1 — Overview (Top Trending)
10-second situation awareness: "What's hot right now?"

Visuals:
  • KPI banner (mentions, authors, sentiment split)
  • Crisis alert banner (conditional)
  • Top 10 Leaderboard dataframe
  • Velocity vs Engagement grouped bar chart
  • Sentiment donut
  • Volume timeline by source
  • Source breakdown bar
"""
from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from components.sidebar import render_sidebar
from components.kpi_row import render_kpi_row
from components.chart_theme import (
    POSITIVE, NEGATIVE, NEUTRAL, SOURCE_COLORS,
    SENTIMENT_COLORS, PRIMARY, ACCENT_BLUE, ACCENT_PURPLE,
    apply_chart_style,
)
from data.queries import get_topic_activity, get_crisis_events

# ── Sidebar ─────────────────────────────────────────────────────
filters = render_sidebar()
df = get_topic_activity(filters.start_date, filters.end_date, filters.sources)

# ── Crisis Banner (conditional) ─────────────────────────────────
crisis_df = get_crisis_events(window_hours=24, severities=["HIGH"])
if len(crisis_df) > 0:
    st.error(
        f"🚨 **{len(crisis_df)} HIGH-severity event(s)** detected in the last 24 hours. "
        "Switch to the **Crisis Monitor** page for details.",
        icon="🚨",
    )

# ── Page Header ─────────────────────────────────────────────────
st.markdown("## 📊 Overview")
st.caption("Real-time snapshot of Vietnamese tech community trends & sentiment")

# ── KPI Row ─────────────────────────────────────────────────────
if not df.empty:
    today = df[df["bucket_date"] == df["bucket_date"].max()]
    total_mentions = int(today["mention_count"].sum())
    total_authors = int(today["unique_authors"].sum())
    total_posts = int(today["mention_count"].sum()) or 1
    pos_pct = today["pos_count"].sum() / total_posts * 100
    neg_pct = today["neg_count"].sum() / total_posts * 100
    neu_pct = today["neu_count"].sum() / total_posts * 100
else:
    total_mentions = total_authors = 0
    pos_pct = neg_pct = neu_pct = 0

render_kpi_row([
    {"label": "📝 Total Mentions", "value": f"{total_mentions:,}"},
    {"label": "👥 Unique Authors", "value": f"{total_authors:,}"},
    {"label": "🟢 Positive",       "value": f"{pos_pct:.1f}%"},
    {"label": "🔴 Negative",       "value": f"{neg_pct:.1f}%", "delta_color": "inverse"},
    {"label": "⚪ Neutral",        "value": f"{neu_pct:.1f}%", "delta_color": "off"},
])

st.divider()

# ── Row 1: Top 10 Leaderboard + Velocity vs Engagement ─────────
col_board, col_compare = st.columns([2, 3], gap="large")

with col_board:
    st.markdown("#### 🏆 Top 10 Trending Topics")
    if not df.empty:
        latest_hour = df["hour_bucket"].max()
        top_df = (
            df[df["hour_bucket"] == latest_hour]
            .groupby(["topic_id", "topic_label"], as_index=False)
            .agg(
                trend_score=("trend_score", "max"),
                velocity=("velocity", "sum"),
                engagement_sum=("engagement_sum", "sum"),
                mentions=("mention_count", "sum"),
            )
            .nlargest(10, "trend_score")
            .sort_values("trend_score", ascending=False)
            .reset_index(drop=True)
        )
        top_df.index = top_df.index + 1
        top_df.index.name = "Rank"

        st.dataframe(
            top_df[["topic_label", "trend_score", "velocity", "engagement_sum", "mentions"]],
            use_container_width=True,
            column_config={
                "topic_label": st.column_config.TextColumn("Topic", width="medium"),
                "trend_score": st.column_config.ProgressColumn(
                    "Trend Score",
                    format="%.1f",
                    min_value=0,
                    max_value=float(top_df["trend_score"].max()) * 1.1 if not top_df.empty else 100,
                ),
                "velocity": st.column_config.NumberColumn("Velocity", format="%d /h"),
                "engagement_sum": st.column_config.NumberColumn("Engagement", format="%,d"),
                "mentions": st.column_config.NumberColumn("Mentions", format="%,d"),
            },
        )
    else:
        st.info("No data available for the selected filters.")

with col_compare:
    st.markdown("#### ⚡ Velocity vs Engagement — Top 10")
    if not df.empty and not top_df.empty:
        # Prepare data for grouped bar chart
        chart_df = top_df.sort_values("trend_score", ascending=True).copy()

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            y=chart_df["topic_label"],
            x=chart_df["velocity"],
            name="Velocity (mentions/h)",
            orientation="h",
            marker_color=ACCENT_BLUE,
            text=chart_df["velocity"],
            texttemplate="%{text:,}",
            textposition="outside",
        ))
        fig_compare.add_trace(go.Bar(
            y=chart_df["topic_label"],
            x=chart_df["engagement_sum"],
            name="Engagement",
            orientation="h",
            marker_color=ACCENT_PURPLE,
            text=chart_df["engagement_sum"],
            texttemplate="%{text:,}",
            textposition="outside",
        ))
        fig_compare.update_layout(
            barmode="group",
            yaxis_title="",
            xaxis_title="Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(apply_chart_style(fig_compare, height=460), use_container_width=True)
    else:
        st.info("Not enough data for comparison chart.")

st.divider()

# ── Row 2: Trend Score Bar + Sentiment Donut ────────────────────
col_trend, col_donut = st.columns([3, 2], gap="large")

with col_trend:
    st.markdown("#### 🔥 Trend Score — Top 10")
    if not df.empty:
        bar_df = (
            df[df["hour_bucket"] == df["hour_bucket"].max()]
            .groupby(["topic_id", "topic_label"], as_index=False)
            .agg(trend_score=("trend_score", "max"))
            .nlargest(10, "trend_score")
            .sort_values("trend_score", ascending=True)
        )
        fig_bar = px.bar(
            bar_df,
            x="trend_score",
            y="topic_label",
            orientation="h",
            color="trend_score",
            color_continuous_scale=[[0, ACCENT_PURPLE], [1, PRIMARY]],
            text="trend_score",
        )
        fig_bar.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside",
        )
        fig_bar.update_layout(
            yaxis_title="",
            xaxis_title="Trend Score",
            coloraxis_showscale=False,
            showlegend=False,
        )
        st.plotly_chart(apply_chart_style(fig_bar, height=420), use_container_width=True)
    else:
        st.info("No data available.")

with col_donut:
    st.markdown("#### 💬 Sentiment Overview")
    if not df.empty:
        sentiment_totals = {
            "Positive": int(df["pos_count"].sum()),
            "Negative": int(df["neg_count"].sum()),
            "Neutral":  int(df["neu_count"].sum()),
        }
        fig_donut = go.Figure(go.Pie(
            labels=list(sentiment_totals.keys()),
            values=list(sentiment_totals.values()),
            hole=0.55,
            marker=dict(colors=SENTIMENT_COLORS),
            textinfo="label+percent",
            textfont=dict(size=13),
            hovertemplate="%{label}: %{value:,} posts (%{percent})<extra></extra>",
        ))
        fig_donut.update_layout(
            showlegend=False,
            annotations=[dict(
                text=f"{sum(sentiment_totals.values()):,}<br><span style='font-size:12px'>posts</span>",
                x=0.5, y=0.5, font_size=18, showarrow=False,
            )],
        )
        st.plotly_chart(apply_chart_style(fig_donut, height=420), use_container_width=True)
    else:
        st.info("No sentiment data.")

st.divider()

# ── Row 3: Volume Timeline + Source Breakdown ───────────────────
col_vol, col_src = st.columns([3, 2], gap="large")

with col_vol:
    st.markdown("#### 📈 Mention Volume Over Time")
    if not df.empty:
        vol_df = (
            df.groupby(["bucket_date", "source"], as_index=False)
            .agg(mentions=("mention_count", "sum"))
        )
        fig_area = px.area(
            vol_df,
            x="bucket_date",
            y="mentions",
            color="source",
            color_discrete_map=SOURCE_COLORS,
        )
        fig_area.update_layout(
            xaxis_title="",
            yaxis_title="Mentions",
            legend_title_text="Source",
        )
        st.plotly_chart(apply_chart_style(fig_area, height=380), use_container_width=True)

with col_src:
    st.markdown("#### 📡 Source Breakdown")
    if not df.empty:
        src_df = (
            df.groupby("source", as_index=False)
            .agg(mentions=("mention_count", "sum"))
            .sort_values("mentions", ascending=True)
        )
        fig_src = px.bar(
            src_df,
            x="mentions",
            y="source",
            orientation="h",
            color="source",
            color_discrete_map=SOURCE_COLORS,
            text="mentions",
        )
        fig_src.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_src.update_layout(showlegend=False, xaxis_title="Total Mentions", yaxis_title="")
        st.plotly_chart(apply_chart_style(fig_src, height=380), use_container_width=True)
