"""
dashboard/app.py
════════════════════════════════════════════════════════════════════════════════
MLOps Predictive Maintenance Dashboard — Entry Point

Run with:
    streamlit run dashboard/app.py

Assumes FastAPI backend is running at http://localhost:8000
"""

import streamlit as st

st.set_page_config(
    page_title="PHM MLOps Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    /* Dark industrial theme */
    .stApp {
        background-color: #0d0f14;
        color: #e2e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111318;
        border-right: 1px solid #1e2530;
    }

    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #f97316;
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        letter-spacing: 0.05em;
        font-size: 1.1rem;
        text-transform: uppercase;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #151820;
        border: 1px solid #1e2530;
        border-radius: 8px;
        padding: 1rem;
    }

    [data-testid="metric-container"] label {
        color: #64748b !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
    }

    /* Buttons */
    .stButton > button {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        background: #f97316;
        color: #0d0f14;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: #fb923c;
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(249,115,22,0.4);
    }

    /* Cards */
    .card {
        background: #151820;
        border: 1px solid #1e2530;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }

    .card-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #64748b;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #1e2530;
        padding-bottom: 0.5rem;
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.75rem;
        border-radius: 100px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .badge-healthy  { background: #14532d; color: #4ade80; }
    .badge-warning  { background: #713f12; color: #fbbf24; }
    .badge-critical { background: #7f1d1d; color: #f87171; }
    .badge-info     { background: #1e3a5f; color: #60a5fa; }
    .badge-drift    { background: #4c1d95; color: #c084fc; }

    /* Section headers */
    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        color: #f1f5f9 !important;
    }

    h1 { font-size: 1.8rem !important; letter-spacing: -0.02em; }
    h2 { font-size: 1.2rem !important; color: #94a3b8 !important; }
    h3 { font-size: 1rem !important; }

    /* Dataframe */
    .stDataFrame {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
    }

    /* Selectbox, text input */
    .stSelectbox, .stTextInput {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Divider */
    hr { border-color: #1e2530; margin: 1.5rem 0; }

    /* Alert / info boxes */
    .stAlert {
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #f97316 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ PHM MLOps")
    st.markdown("---")
    st.caption("NAVIGATION")

    pages = {
        "🏠  Overview":          "pages/1_overview.py",
        "🚀  Pipeline Control":  "pages/2_pipeline.py",
        "📊  RUL Monitor":       "pages/3_rul.py",
        "🔍  Data Quality":      "pages/4_data.py",
        "✅  Fault Review":       "pages/5_fault_review.py",
        "📜  Audit / History":   "pages/6_audit_history.py",
    }

    st.markdown("---")
    st.caption("BACKEND")
    api_url = st.text_input("API Base URL", value="http://localhost:8000", label_visibility="collapsed")
    st.session_state["api_url"] = api_url

    status_placeholder = st.empty()
    if st.button("Test Connection", use_container_width=True):
        import requests
        try:
            r = requests.get(f"{api_url}/docs", timeout=3)
            if r.status_code == 200:
                status_placeholder.success("✅ Connected")
            else:
                status_placeholder.error(f"❌ HTTP {r.status_code}")
        except Exception as e:
            status_placeholder.error(f"❌ {e}")

    st.markdown("---")
    st.caption(f"v1.0.0 · RUL Predictive Maintenance")

# ── Landing page ──────────────────────────────────────────────────────────────
st.markdown("# ⚙️ PHM MLOps Dashboard")
st.markdown("### Predictive Maintenance · Remaining Useful Life · SCADA Pipeline")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-header">Pipeline Control</div>
        Trigger the full workflow orchestrator, monitor step-by-step progress, and manage live serving runs.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-header">RUL & Monitoring</div>
        Live RUL predictions per bearing with health status, drift detection, and data quality flags.
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-header">Fault Review & Audit</div>
        Maintenance worker fault confirm/deny interface and full serving history audit trail.
    </div>
    """, unsafe_allow_html=True)

st.info("👈 Use the sidebar to navigate between sections. Set your API URL above and test the connection first.")