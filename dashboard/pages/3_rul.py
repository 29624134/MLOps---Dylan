"""
dashboard/pages/3_📊_RUL_Monitor.py
════════════════════════════════════════════════════════════════════════════════
Live RUL predictions per bearing — time-series chart, PM status, alert log.
"""

import time
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dashboard.api_client import get_latest_bearing_records, get_serving_history

st.markdown("# 📊 RUL Monitor")
st.markdown("Remaining Useful Life predictions and predictive maintenance alerts.")
st.markdown("---")

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([3, 2, 1])
with col1:
    bearing = st.selectbox("Bearing", ["Bearing1_2","Bearing1_3","Bearing1_4","Bearing1_5","Bearing1_6","Bearing1_7", "Bearing2_3"], index=0)
with col2:
    n_records = st.slider("Records to display", 20, 500, 100, step=10)
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    live_mode = st.toggle("Live", value=False)

# ── Fetch data ────────────────────────────────────────────────────────────────
records = get_serving_history(bearing, limit=n_records)

if not records:
    st.warning(f"No serving history found for **{bearing}**. Run the pipeline first.")
    st.stop()

# Build DataFrame
rows = []
for r in records:
    pm  = r.get("pm",         {}) or {}
    inf = r.get("inference",  {}) or {}
    mn  = r.get("monitoring", {}) or {}
    fe  = r.get("fe",         {}) or {}
    rows.append({
        "burst_idx":   r.get("burst_idx"),
        "timestamp":   r.get("timestamp", ""),
        "rul_s":       r.get("rul_s")  or inf.get("rul_s"),
        "rul_min":     r.get("rul_min") or inf.get("rul_min"),
        "pm_status":   pm.get("status", "—"),
        "alert":       pm.get("alert", False),
        "action":      pm.get("recommended_action", "—"),
        "data_quality":r.get("data_quality", "clean"),
        "drift":       mn.get("drift_detected", False),
        "anomaly":     mn.get("anomaly_flag", False),
        "model_ver":   r.get("model_version", "—"),
    })

df = pd.DataFrame(rows).dropna(subset=["rul_min"]).sort_values("burst_idx")

# ── Key metrics row ───────────────────────────────────────────────────────────
if not df.empty:
    latest = df.iloc[-1]
    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Latest RUL",
              f"{latest['rul_min']:.1f} min" if pd.notna(latest['rul_min']) else "—",
              delta=f"{df['rul_min'].iloc[-1] - df['rul_min'].iloc[-2]:.1f} min"
                   if len(df) > 1 else None)

    m2.metric("PM Status",   latest["pm_status"].upper())
    m3.metric("Data Quality", latest["data_quality"])
    m4.metric("Alerts (shown)", str(df["alert"].sum()))
    m5.metric("Model Version", latest["model_ver"][:12] if latest["model_ver"] != "—" else "—")

st.markdown("---")

# ── RUL Time-Series Chart ─────────────────────────────────────────────────────
st.markdown("#### RUL Over Time")

fig = go.Figure()

# Colour code by PM status
status_colors = {"healthy": "#4ade80", "warning": "#fbbf24", "critical": "#f87171"}

for status, color in status_colors.items():
    mask = df["pm_status"] == status
    if mask.any():
        sub = df[mask]
        fig.add_trace(go.Scatter(
            x    = sub["burst_idx"],
            y    = sub["rul_min"],
            mode = "markers",
            name = status.capitalize(),
            marker={"color": color, "size": 4, "opacity": 0.8},
        ))

# Alert markers
alerts = df[df["alert"] == True]
if not alerts.empty:
    fig.add_trace(go.Scatter(
        x    = alerts["burst_idx"],
        y    = alerts["rul_min"],
        mode = "markers",
        name = "Alert",
        marker={"color": "#f97316", "size": 12, "symbol": "triangle-up",
                "line": {"color": "#fff", "width": 1}},
    ))

# Warning & critical threshold lines
fig.add_hline(y=60,  line_dash="dot", line_color="#f87171",
              annotation_text="Critical (1h)", annotation_position="right",
              annotation_font={"size": 10, "color": "#f87171"})
fig.add_hline(y=240, line_dash="dot", line_color="#fbbf24",
              annotation_text="Warning (4h)", annotation_position="right",
              annotation_font={"size": 10, "color": "#fbbf24"})

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    height=380,
    margin=dict(l=60, r=60, t=20, b=40),
    font={"color": "#e2e8f0", "family": "JetBrains Mono"},
    xaxis={
        "title":      "Burst Index",
        "gridcolor":  "#1e2530",
        "showgrid":   True,
        "color":      "#64748b",
    },
    yaxis={
        "title":      "RUL (minutes)",
        "gridcolor":  "#1e2530",
        "showgrid":   True,
        "color":      "#64748b",
    },
    legend={"bgcolor": "#151820", "bordercolor": "#1e2530", "borderwidth": 1},
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ── Alert Log ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Alert Log")

alert_df = df[df["alert"] == True][["burst_idx", "rul_min", "pm_status", "action", "data_quality"]].copy()
alert_df.columns = ["Burst", "RUL (min)", "Status", "Recommended Action", "Data Quality"]

if alert_df.empty:
    st.success("✅ No alerts in the current window.")
else:
    st.error(f"⚠ {len(alert_df)} alerts detected in the last {n_records} bursts.")
    st.dataframe(alert_df.style.applymap(
        lambda v: "color: #f87171; font-weight:600;" if v == "critical"
                  else "color: #fbbf24; font-weight:600;" if v == "warning"
                  else "",
        subset=["Status"]
    ), use_container_width=True, hide_index=True)

# ── RUL Distribution histogram ────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### RUL Distribution")

fig2 = go.Figure()
fig2.add_trace(go.Histogram(
    x=df["rul_min"],
    nbinsx=40,
    marker_color="#f97316",
    opacity=0.7,
    name="RUL (min)",
))
fig2.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    height=250,
    margin=dict(l=50, r=20, t=20, b=40),
    font={"color": "#e2e8f0", "family": "JetBrains Mono"},
    xaxis={"title": "RUL (min)", "gridcolor": "#1e2530", "color": "#64748b"},
    yaxis={"title": "Count",     "gridcolor": "#1e2530", "color": "#64748b"},
    bargap=0.05,
)
st.plotly_chart(fig2, use_container_width=True)

# ── Live refresh ──────────────────────────────────────────────────────────────
if live_mode:
    time.sleep(8)
    st.rerun()