"""
Vietnamese Tech Trend & Controversy Radar — Dashboard Entry Point.

Run with:
    streamlit run dashboard/app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Tech Trend Radar",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Define pages ────────────────────────────────────────────────
overview_page = st.Page("pages/1_pg_overview.py",          title="Overview",        icon="📊", default=True)
trends_page   = st.Page("pages/2_pg_trend_explorer.py",    title="Trend Explorer",  icon="🔥")
crisis_page   = st.Page("pages/3_pg_crisis_monitor.py",    title="Crisis Monitor",  icon="🚨")

pg = st.navigation([overview_page, trends_page, crisis_page])
pg.run()
