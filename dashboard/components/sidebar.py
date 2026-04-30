"""
Shared sidebar rendered on every page.
Returns the current filter state as a dict.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import streamlit as st


@dataclass
class Filters:
    start_date: dt.date
    end_date: dt.date
    sources: list[str]


def render_sidebar() -> Filters:
    """Draw the sidebar and return the active filter values."""
    with st.sidebar:
        st.markdown(
            "<h2 style='margin-bottom:0'>🇻🇳 Tech Trend Radar</h2>"
            "<p style='color:#718096;margin-top:0;font-size:0.85rem'>"
            "Vietnamese Tech Trend &amp; Controversy Radar</p>",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Global filters ──────────────────────────────────
        st.markdown("##### 🎛️ Filters")

        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input(
                "From",
                value=dt.date.today() - dt.timedelta(days=30),
                key="filter_start",
            )
        with col2:
            end = st.date_input(
                "To",
                value=dt.date.today(),
                key="filter_end",
            )

        sources = st.multiselect(
            "Sources",
            options=["voz", "vnexpress", "youtube", "tinhte"],
            default=["voz", "vnexpress", "youtube", "tinhte"],
            key="filter_sources",
        )

        st.divider()

        # ── Data status ─────────────────────────────────────
        st.caption("📡 **Data Status**")
        st.caption(f"Dashboard loaded: {dt.datetime.now():%H:%M %d/%m/%Y}")

    return Filters(start_date=start, end_date=end, sources=sources)
