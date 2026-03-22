"""
Page 3 — Crisis Monitor
Anomaly detection dashboard: "Is something blowing up right now?"
"""
from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from components.sidebar import render_sidebar
from components.kpi_row import render_kpi_row
from components.chart_theme import (
    CRISIS_HIGH, CRISIS_MED, CRISIS_LOW, SEVERITY_COLORS,
    NEGATIVE, NEUTRAL, ACCENT_BLUE, PRIMARY,
    apply_chart_style,
)
from data.queries import get_crisis_events, get_topic_activity, get_dim_topics

# ── Sidebar ─────────────────────────────────────────────────────
filters = render_sidebar()

# ── Page Header ─────────────────────────────────────────────────
st.markdown("## 🚨 Crisis Monitor")
st.caption("Anomaly detection & controversy tracking — powered by Isolation Forest + Rolling Mean")

# ── Page-level Filters ──────────────────────────────────────────
fcol1, fcol2 = st.columns([2, 2], gap="large")

with fcol1:
    severity_filter = st.multiselect(
        "Severity",
        options=["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"],
    )

with fcol2:
    window_opt = st.radio(
        "Time Window",
        options=["Last 24h", "Last 7d", "Last 30d"],
        horizontal=True,
        index=1,
    )
    window_hours = {"Last 24h": 24, "Last 7d": 168, "Last 30d": 720}[window_opt]

# ── Load data ───────────────────────────────────────────────────
crisis_df = get_crisis_events(window_hours=window_hours, severities=severity_filter)
activity_df = get_topic_activity(filters.start_date, filters.end_date, filters.sources)

# ── KPI Row ─────────────────────────────────────────────────────
high_count  = len(crisis_df[crisis_df["severity"] == "HIGH"])  if not crisis_df.empty else 0
med_count   = len(crisis_df[crisis_df["severity"] == "MEDIUM"]) if not crisis_df.empty else 0
low_count   = len(crisis_df[crisis_df["severity"] == "LOW"])   if not crisis_df.empty else 0
avg_anomaly = crisis_df["anomaly_score"].mean() if not crisis_df.empty else 0

render_kpi_row([
    {"label": "🔴 HIGH",    "value": str(high_count)},
    {"label": "🟡 MEDIUM",  "value": str(med_count)},
    {"label": "🟢 LOW",     "value": str(low_count)},
    {"label": "📉 Avg Anomaly Score", "value": f"{avg_anomaly:.2f}"},
])

st.divider()

# ── Chart 1: Crisis Event Timeline (Scatter) ───────────────────
st.markdown("#### ⏱️ Crisis Event Timeline")

if not crisis_df.empty:
    crisis_df["severity_y"] = crisis_df["severity"].map({"HIGH": 3, "MEDIUM": 2, "LOW": 1})
    crisis_df["topics_str"] = crisis_df["affected_topic_labels"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )
    crisis_df["conditions_str"] = crisis_df["trigger_conditions"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )

    fig_timeline = px.scatter(
        crisis_df,
        x="detected_at",
        y="severity_y",
        size="anomaly_score",
        color="severity",
        color_discrete_map=SEVERITY_COLORS,
        hover_data={
            "severity_y": False,
            "anomaly_score": ":.2f",
            "neg_ratio": ":.0%",
            "mention_velocity": ":.0f",
            "topics_str": True,
            "conditions_str": True,
        },
        labels={
            "detected_at": "Time",
            "anomaly_score": "Anomaly Score",
            "neg_ratio": "Neg Ratio",
            "mention_velocity": "Velocity",
            "topics_str": "Topics",
            "conditions_str": "Triggers",
        },
        size_max=28,
    )
    fig_timeline.update_layout(
        yaxis=dict(
            tickvals=[1, 2, 3],
            ticktext=["LOW", "MEDIUM", "HIGH"],
            title="Severity",
        ),
        xaxis_title="",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(apply_chart_style(fig_timeline, height=380), use_container_width=True)
else:
    st.success("✅ No crisis events detected in the selected window. All clear!")

st.divider()

# ── Row 2: Z-Score Heatmap + Neg Ratio Threshold ──────────────
col_heat, col_neg = st.columns(2, gap="large")

with col_heat:
    st.markdown("#### 🌡️ Volume Z-Score Heatmap")
    if not activity_df.empty:
        # Pick top 12 topics by total mentions
        top_topics = (
            activity_df.groupby("topic_label", as_index=False)["mention_count"]
            .sum()
            .nlargest(12, "mention_count")["topic_label"]
            .tolist()
        )
        heat_data = activity_df[activity_df["topic_label"].isin(top_topics)].copy()

        # Aggregate to daily for cleaner heatmap
        heat_agg = (
            heat_data.groupby(["topic_label", "bucket_date"], as_index=False)
            .agg(zscore=("volume_zscore", "max"))
        )
        heat_pivot = heat_agg.pivot(index="topic_label", columns="bucket_date", values="zscore")
        heat_pivot = heat_pivot.fillna(0)

        fig_heat = px.imshow(
            heat_pivot,
            color_continuous_scale=[
                [0.0, ACCENT_BLUE],
                [0.4, "#1A1A2E"],
                [0.6, "#1A1A2E"],
                [0.8, CRISIS_MED],
                [1.0, CRISIS_HIGH],
            ],
            labels=dict(x="Date", y="Topic", color="Z-Score"),
            aspect="auto",
        )
        fig_heat.update_layout(
            xaxis=dict(dtick=86400000 * 3, tickformat="%d/%m"),
            coloraxis_colorbar=dict(title="Z-Score"),
        )
        st.plotly_chart(apply_chart_style(fig_heat, height=420), use_container_width=True)
    else:
        st.info("No activity data for heatmap.")

with col_neg:
    st.markdown("#### 📊 Negative Ratio vs Baseline")
    if not activity_df.empty:
        # Aggregate across topics/sources to system-level
        neg_ts = (
            activity_df.groupby("bucket_date", as_index=False)
            .agg(
                neg_ratio=("neg_ratio", "mean"),
                neg_ratio_avg=("neg_ratio_24h_avg", "mean"),
            )
            .sort_values("bucket_date")
        )
        # Threshold = avg + 2*std
        threshold = neg_ts["neg_ratio"].mean() + 2 * neg_ts["neg_ratio"].std()

        fig_neg = go.Figure()
        fig_neg.add_trace(go.Scatter(
            x=neg_ts["bucket_date"], y=neg_ts["neg_ratio"],
            name="Actual Neg Ratio", mode="lines",
            line=dict(color=NEGATIVE, width=2),
        ))
        fig_neg.add_trace(go.Scatter(
            x=neg_ts["bucket_date"], y=neg_ts["neg_ratio_avg"],
            name="24h Rolling Avg", mode="lines",
            line=dict(color=NEUTRAL, width=1.5, dash="dash"),
        ))
        fig_neg.add_hline(
            y=threshold,
            line_dash="dot", line_color=CRISIS_HIGH, line_width=1.5,
            annotation_text=f"2σ threshold ({threshold:.2%})",
            annotation_position="top right",
            annotation_font_color=CRISIS_HIGH,
        )
        fig_neg.update_layout(
            xaxis_title="", yaxis_title="Negative Ratio",
            yaxis_tickformat=".0%",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(apply_chart_style(fig_neg, height=420), use_container_width=True)
    else:
        st.info("No data for neg ratio chart.")

st.divider()

# ── Crisis Event Log (Table) ───────────────────────────────────
st.markdown("#### 📋 Crisis Event Log")

if not crisis_df.empty:
    display_df = crisis_df.copy()
    severity_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    display_df["sev_display"] = display_df["severity"].map(
        lambda s: f"{severity_emoji.get(s, '')} {s}"
    )
    display_df["topics"] = display_df["affected_topic_labels"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )
    display_df["triggers"] = display_df["trigger_conditions"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
    )
    display_df["time"] = pd.to_datetime(display_df["detected_at"]).dt.strftime("%Y-%m-%d %H:%M")

    st.dataframe(
        display_df[["sev_display", "time", "anomaly_score", "neg_ratio",
                     "mention_velocity", "topics", "triggers"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "sev_display": st.column_config.TextColumn("Severity", width="small"),
            "time": st.column_config.TextColumn("Detected At", width="medium"),
            "anomaly_score": st.column_config.NumberColumn("Score", format="%.2f", width="small"),
            "neg_ratio": st.column_config.ProgressColumn(
                "Neg Ratio", format="%.0f%%", min_value=0, max_value=1,
            ),
            "mention_velocity": st.column_config.NumberColumn("Velocity", format="%.0f /h"),
            "topics": st.column_config.TextColumn("Affected Topics", width="large"),
            "triggers": st.column_config.TextColumn("Trigger Conditions", width="medium"),
        },
    )

    # Expandable details for each event
    st.markdown("##### 🔍 Event Details")
    for _, row in display_df.head(5).iterrows():
        with st.expander(
            f"{severity_emoji.get(row['severity'], '')} "
            f"Event {row['event_id']} — {row['time']} — Score: {row['anomaly_score']:.2f}"
        ):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Trigger Conditions:**")
                for cond in (row["trigger_conditions"] if isinstance(row["trigger_conditions"], list)
                             else [row["trigger_conditions"]]):
                    cond_color = {
                        "neg_spike": NEGATIVE,
                        "volume_anomaly": CRISIS_MED,
                        "isolation_forest": PRIMARY,
                    }.get(cond, NEUTRAL)
                    st.markdown(
                        f"<span style='background:{cond_color}33;color:{cond_color};"
                        f"padding:4px 12px;border-radius:12px;margin:2px;'>"
                        f"⚡ {cond}</span>",
                        unsafe_allow_html=True,
                    )
            with c2:
                st.markdown("**Affected Topics:**")
                topics = row["affected_topic_labels"] if isinstance(row["affected_topic_labels"], list) else []
                for t in topics:
                    st.markdown(f"- {t}")

                evidence = row.get("evidence_doc_ids", [])
                if evidence and isinstance(evidence, list):
                    st.markdown("**Evidence Posts:**")
                    st.code(", ".join(evidence[:5]))
else:
    st.success("✅ No crisis events in the selected window.")
