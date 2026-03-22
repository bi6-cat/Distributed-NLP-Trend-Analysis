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
overview_page = st.Page("pages/1_📊_Overview.py",       title="Overview",        icon="📊", default=True)
trends_page   = st.Page("pages/2_🔥_Trend_Explorer.py", title="Trend Explorer",  icon="🔥")
crisis_page   = st.Page("pages/3_🚨_Crisis_Monitor.py", title="Crisis Monitor",  icon="🚨")

pg = st.navigation([overview_page, trends_page, crisis_page])
pg.run()
