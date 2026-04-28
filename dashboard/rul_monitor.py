"""
gui_rul_monitor.py
═══════════════════════════════════════════════════════════════════════════════
GUI 2 — RUL Predictions Monitor

Fix #4: This is one of exactly two GUIs in the system.
        Runs as a standalone Streamlit app on a separate port from the fault GUI.

Purpose : Live view of Remaining Useful Life predictions per bearing.
          Shows the RUL time-series, PM status, drift/anomaly flags, and
          a real-time alert log. Auto-refreshes when Live mode is on.

Run     : streamlit run gui_rul_monitor.py --server.port 8502
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import requests

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RUL Monitor",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-critical { color: #f87171 !important; font-weight: 700; }
    .metric-warning  { color: #fbbf24 !important; font-weight: 700; }
    .metric-healthy  { color: #4ade80 !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 RUL Monitor")
    st.markdown("**Live Predictions**")
    st.markdown("---")

    api_url = st.text_input(
        "API Base URL",
        value=st.session_state.get("api_url", "http://localhost:8000"),
        label_visibility="visible",
    )
    st.session_state["api_url"] = api_url

    st.markdown("---")
    bearing = st.selectbox(
        "Bearing",
        [
            "Bearing1_1","Bearing1_2","Bearing1_3","Bearing1_4","Bearing1_5",
            "Bearing1_6","Bearing1_7","Bearing2_1","Bearing2_2","Bearing2_3",
        ],
        index=0,
    )
    n_records = st.slider("Records to display", 20, 500, 100, step=10)
    live_mode = st.toggle("🔴 Live (auto-refresh 8 s)", value=False)

    st.markdown("---")
    if st.button("🔌 Test Connection", use_container_width=True):
        try:
            r = requests.get(f"{api_url}/docs", timeout=3)
            st.success("✅ Connected") if r.status_code == 200 else st.error(f"❌ HTTP {r.status_code}")
        except Exception as e:
            st.error(f"❌ {e}")

    st.markdown("---")
    st.caption("PHM MLOps · RUL Monitor GUI")


# ── Fetch data ─────────────────────────────────────────────────────────────────
def _get_serving_history(bearing_name: str, limit: int) -> list:
    try:
        r = requests.get(
            f"{api_url}/serving-history",
            params={"bearing_name": bearing_name, "limit": limit},
            timeout=5,
        )
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


# ── Main page ──────────────────────────────────────────────────────────────────
st.markdown("# 📊 RUL Monitor")
st.markdown(f"Remaining Useful Life predictions — `{bearing}`")
st.markdown("---")

records = _get_serving_history(bearing, limit=n_records)

if not records:
    st.warning(
        f"No serving history for **{bearing}**. "
        "Start the serving pipeline and SCADA simulator first."
    )
    if live_mode:
        time.sleep(8)
        st.rerun()
    st.stop()

# ── Build DataFrame ────────────────────────────────────────────────────────────
rows = []
for r in records:
    pm  = r.get("pm",         {}) or {}
    inf = r.get("inference",  {}) or {}
    mn  = r.get("monitoring", {}) or {}
    rows.append({
        "burst_idx":    r.get("burst_idx"),
        "timestamp":    r.get("timestamp", ""),
        "rul_s":        r.get("rul_s")  or inf.get("rul_s"),
        "rul_min":      r.get("rul_min") or inf.get("rul_min"),
        "pm_status":    pm.get("status", "—"),
        "alert":        pm.get("alert", False),
        "action":       pm.get("recommended_action", "—"),
        "data_quality": r.get("data_quality", "clean"),
        "drift":        mn.get("drift_detected", False),
        "anomaly":      mn.get("anomaly_flag", False),
        "model_ver":    r.get("model_version", "—"),
    })

df = pd.DataFrame(rows).dropna(subset=["rul_min"]).sort_values("burst_idx")

# ── Key metrics ────────────────────────────────────────────────────────────────
if not df.empty:
    latest = df.iloc[-1]
    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "Latest RUL",
        f"{latest['rul_min']:.1f} min" if pd.notna(latest["rul_min"]) else "—",
        delta=(
            f"{df['rul_min'].iloc[-1] - df['rul_min'].iloc[-2]:.1f} min"
            if len(df) > 1 else None
        ),
    )
    m2.metric("PM Status",    latest["pm_status"].upper())
    m3.metric("Data Quality", latest["data_quality"])
    m4.metric("Alerts",       str(df["alert"].sum()))
    m5.metric(
        "Model",
        latest["model_ver"][:12] if latest["model_ver"] != "—" else "—",
    )

    # Critical banner — always visible, never stops predictions
    if latest["pm_status"] == "critical":
        st.error(
            f"🚨 **CRITICAL** — RUL is {latest['rul_min']:.1f} min. "
            "Predictions continue. Maintenance worker must act via the Fault Review GUI."
        )
    elif latest["pm_status"] == "warning":
        st.warning(
            f"⚠️ **WARNING** — RUL is {latest['rul_min']:.1f} min. "
            "Monitor closely."
        )

st.markdown("---")

# ── RUL Time-Series Chart ──────────────────────────────────────────────────────
st.markdown("#### RUL Over Time")

fig = go.Figure()

status_colors = {
    "healthy":  "#4ade80",
    "warning":  "#fbbf24",
    "critical": "#f87171",
}

for status, color in status_colors.items():
    mask = df["pm_status"] == status
    if mask.any():
        sub = df[mask]
        fig.add_trace(go.Scatter(
            x     = sub["burst_idx"],
            y     = sub["rul_min"],
            mode  = "markers",
            name  = status.capitalize(),
            marker= {"color": color, "size": 4, "opacity": 0.8},
        ))

alerts = df[df["alert"] == True]
if not alerts.empty:
    fig.add_trace(go.Scatter(
        x     = alerts["burst_idx"],
        y     = alerts["rul_min"],
        mode  = "markers",
        name  = "Alert",
        marker= {
            "color": "#f97316", "size": 12, "symbol": "triangle-up",
            "line": {"color": "#fff", "width": 1},
        },
    ))

fig.add_hline(
    y=60,  line_dash="dot", line_color="#f87171",
    annotation_text="Critical (1 h)",
    annotation_position="right",
    annotation_font={"size": 10, "color": "#f87171"},
)
fig.add_hline(
    y=240, line_dash="dot", line_color="#fbbf24",
    annotation_text="Warning (4 h)",
    annotation_position="right",
    annotation_font={"size": 10, "color": "#fbbf24"},
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    height=380,
    margin=dict(l=60, r=80, t=20, b=40),
    font={"color": "#e2e8f0"},
    xaxis={"title": "Burst Index", "gridcolor": "#1e2530",
           "showgrid": True, "color": "#64748b"},
    yaxis={"title": "RUL (minutes)", "gridcolor": "#1e2530",
           "showgrid": True, "color": "#64748b"},
    legend={"bgcolor": "#151820", "bordercolor": "#1e2530", "borderwidth": 1},
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ── Alert Log ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Alert Log")

alert_df = df[df["alert"] == True][
    ["burst_idx", "rul_min", "pm_status", "action", "data_quality"]
].copy()
alert_df.columns = ["Burst", "RUL (min)", "Status", "Recommended Action", "Data Quality"]

if alert_df.empty:
    st.success("✅ No alerts in the current window.")
else:
    st.error(f"⚠ {len(alert_df)} alert(s) in the last {n_records} bursts.")
    st.dataframe(alert_df, use_container_width=True, hide_index=True)

# ── Drift & Anomaly summary ────────────────────────────────────────────────────
st.markdown("---")
col_d1, col_d2 = st.columns(2)

with col_d1:
    st.markdown("#### Drift Flags")
    drift_df = df[df["drift"] == True][["burst_idx", "rul_min", "pm_status"]]
    if drift_df.empty:
        st.success("No drift detected.")
    else:
        st.warning(f"{len(drift_df)} burst(s) with drift.")
        st.dataframe(drift_df.rename(columns={
            "burst_idx": "Burst", "rul_min": "RUL (min)", "pm_status": "Status",
        }), use_container_width=True, hide_index=True)

with col_d2:
    st.markdown("#### Anomaly Flags")
    anom_df = df[df["anomaly"] == True][["burst_idx", "rul_min", "pm_status"]]
    if anom_df.empty:
        st.success("No anomalies detected.")
    else:
        st.warning(f"{len(anom_df)} burst(s) with anomaly flag.")
        st.dataframe(anom_df.rename(columns={
            "burst_idx": "Burst", "rul_min": "RUL (min)", "pm_status": "Status",
        }), use_container_width=True, hide_index=True)

# ── RUL Distribution ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### RUL Distribution")

fig2 = go.Figure()
fig2.add_trace(go.Histogram(
    x=df["rul_min"], nbinsx=40,
    marker_color="#f97316", opacity=0.75, name="RUL (min)",
))
fig2.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    height=250,
    margin=dict(l=50, r=20, t=20, b=40),
    font={"color": "#e2e8f0"},
    xaxis={"title": "RUL (min)", "gridcolor": "#1e2530", "color": "#64748b"},
    yaxis={"title": "Count",     "gridcolor": "#1e2530", "color": "#64748b"},
    bargap=0.05,
)
st.plotly_chart(fig2, use_container_width=True)

# ── Live refresh ───────────────────────────────────────────────────────────────
if live_mode:
    time.sleep(8)
    st.rerun()