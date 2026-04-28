"""
dashboard/pages/6_📜_Audit_History.py
════════════════════════════════════════════════════════════════════════════════
Serving History audit trail — step 10 (ServHistory → AuditSvc).
Full burst-level records, run summaries, and raw JSON inspector.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dashboard.api_client import get_serving_history, get_run_summary

st.markdown("# 📜 Audit / Serving History")
st.markdown("Full audit trail from the Serving History store (Step 9 → Step 10).")
st.markdown("---")

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])
with col1:
    bearing = st.selectbox("Bearing", ["Bearing1_5", "Bearing1_4", "Bearing1_6",
                                        "Bearing1_7", "Bearing2_3"])
with col2:
    n_records = st.slider("Records to fetch", 50, 2000, 200, step=50)

records = get_serving_history(bearing, limit=n_records)

if not records:
    st.warning(f"No serving history for **{bearing}**.")
    st.stop()

# ── Parse into flat DataFrame ──────────────────────────────────────────────────
rows = []
for r in records:
    pm  = r.get("pm", {})         or {}
    inf = r.get("inference", {})  or {}
    mn  = r.get("monitoring", {}) or {}
    fe  = r.get("fe", {})         or {}
    rows.append({
        "record_id":   r.get("record_id", r.get("_id", "—")),
        "run_id":      r.get("run_id", "—"),
        "burst_idx":   r.get("burst_idx"),
        "timestamp":   r.get("timestamp", "")[:19],
        "ok":          r.get("pipeline_ok", r.get("ok", True)),
        "rul_s":       r.get("rul_s")  or inf.get("rul_s"),
        "rul_min":     r.get("rul_min") or inf.get("rul_min"),
        "pm_status":   pm.get("status", "—"),
        "alert":       pm.get("alert", False),
        "action":      pm.get("recommended_action", "—"),
        "data_quality":r.get("data_quality", "clean"),
        "model_ver":   r.get("model_version") or inf.get("model_version", "—"),
        "drift":       mn.get("drift_detected", False),
        "anomaly":     mn.get("anomaly_flag", False),
        "fe_ready":    fe.get("ready", r.get("ready", False)),
        "error":       r.get("error", ""),
    })

df = pd.DataFrame(rows).sort_values("burst_idx", ascending=False)

# ── Summary metrics ───────────────────────────────────────────────────────────
total   = len(df)
ok_cnt  = df["ok"].sum()
err_cnt = total - ok_cnt
alerts  = df["alert"].sum()
drift   = df["drift"].sum()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Records", total)
m2.metric("✅ OK",          ok_cnt)
m3.metric("❌ Errors",      err_cnt)
m4.metric("🔔 Alerts",     alerts)
m5.metric("〰 Drift",       drift)

st.markdown("---")

# ── Run summary section ────────────────────────────────────────────────────────
st.markdown("### Run Summary")
run_id_input = st.text_input("Run ID (optional — leave blank to skip)",
                              value=st.session_state.get("active_run_id", ""),
                              placeholder="run_20260413_abc12345")

if run_id_input:
    summary = get_run_summary(run_id_input)
    if summary:
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Total Bursts",    summary.get("total_bursts", "—"))
        s2.metric("OK Bursts",       summary.get("ok_count", "—"))
        s3.metric("Errors",          summary.get("error_count", "—"))
        s4.metric("Critical",        summary.get("critical_count", "—"))
        s5.metric("Warning",         summary.get("warning_count", "—"))
        s6.metric("Healthy",         summary.get("healthy_count", "—"))

        bearings_str = ", ".join(summary.get("bearings", []))
        st.caption(f"Bearings: {bearings_str} · "
                   f"From: {summary.get('first_ts','—')[:19]} → "
                   f"To: {summary.get('last_ts','—')[:19]}")
    else:
        st.warning(f"No summary found for run `{run_id_input}`.")

st.markdown("---")

# ── PM Status chart ───────────────────────────────────────────────────────────
st.markdown("### PM Status Over Time")
status_counts = df["pm_status"].value_counts()
fig_status = go.Figure(go.Bar(
    x=status_counts.index,
    y=status_counts.values,
    marker_color=[
        {"healthy":  "#4ade80", "warning": "#fbbf24",
         "critical": "#f87171"}.get(s, "#64748b")
        for s in status_counts.index
    ],
    text=status_counts.values,
    textposition="outside",
    textfont={"family": "JetBrains Mono", "size": 11},
))
fig_status.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    height=250,
    margin=dict(l=40, r=20, t=20, b=40),
    font={"color": "#e2e8f0", "family": "JetBrains Mono"},
    xaxis={"color": "#64748b", "gridcolor": "#1e2530"},
    yaxis={"color": "#64748b", "gridcolor": "#1e2530", "title": "Count"},
)
st.plotly_chart(fig_status, use_container_width=True)

st.markdown("---")

# ── Full record table ──────────────────────────────────────────────────────────
st.markdown("### Serving History Records")

display_df = df[[
    "burst_idx", "timestamp", "run_id", "ok", "rul_min",
    "pm_status", "alert", "data_quality", "drift", "anomaly", "error"
]].copy()
display_df.columns = [
    "Burst", "Timestamp", "Run ID", "OK", "RUL (min)",
    "PM Status", "Alert", "Quality", "Drift", "Anomaly", "Error"
]

if "RUL (min)" in display_df.columns:
    display_df["RUL (min)"] = display_df["RUL (min)"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "—"
    )

styled = display_df.style.applymap(
    lambda v: "color: #4ade80; font-weight:600;" if v is True
              else "color: #f87171; font-weight:600;" if v is False and isinstance(v, bool)
              else "",
    subset=["OK", "Alert", "Drift", "Anomaly"]
).applymap(
    lambda v: "color: #f87171;" if v == "critical"
              else "color: #fbbf24;" if v == "warning"
              else "color: #4ade80;" if v == "healthy"
              else "",
    subset=["PM Status"]
)

st.dataframe(styled, use_container_width=True, hide_index=True)

# ── CSV download ───────────────────────────────────────────────────────────────
csv = display_df.to_csv(index=False)
st.download_button(
    "⬇ Download as CSV",
    data=csv,
    file_name=f"serving_history_{bearing}.csv",
    mime="text/csv",
)

# ── Raw JSON inspector ─────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔎 Raw JSON Record Inspector", expanded=False):
    burst_options = df["burst_idx"].dropna().astype(int).tolist()
    if burst_options:
        selected_burst = st.selectbox("Select burst to inspect", burst_options)
        record = next((r for r in records if r.get("burst_idx") == selected_burst), None)
        if record:
            st.json(record)