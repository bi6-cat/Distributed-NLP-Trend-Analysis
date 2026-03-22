"""
Reusable KPI metric row component.
"""
from __future__ import annotations
import streamlit as st


def render_kpi_row(metrics: list[dict]) -> None:
    """
    Render a row of st.metric cards.

    Each dict in *metrics* should have:
        label  – metric name
        value  – current value (str or number)
        delta  – optional delta string (e.g. "+12%")
        delta_color – optional: "normal" | "inverse" | "off"
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=m.get("delta"),
                delta_color=m.get("delta_color", "normal"),
            )
