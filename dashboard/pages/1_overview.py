"""
dashboard/pages/1_🏠_Overview.py
════════════════════════════════════════════════════════════════════════════════
System overview: active run status, model info, bearing health at-a-glance.
"""

import time
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import get_workflow_status, get_latest_bearing_records, get_deployed_model

st.markdown("# 🏠 System Overview")
st.markdown("Real-time health snapshot of the PHM MLOps pipeline.")
st.markdown("---")

# ── Active run ID input ───────────────────────────────────────────────────────
col_run, col_refresh = st.columns([4, 1])
with col_run:
    run_id = st.text_input("Active Run ID (paste from Pipeline Control)",
                            value=st.session_state.get("active_run_id", ""),
                            placeholder="run_20260413_abc12345")
with col_refresh:
    st.markdown("<br>", unsafe_allow_html=True)
    auto_refresh = st.toggle("Auto-refresh", value=False)

if run_id:
    st.session_state["active_run_id"] = run_id

# ── Workflow status summary ───────────────────────────────────────────────────
if run_id:
    status_data = get_workflow_status(run_id)
    if status_data:
        overall = status_data.get("status", "UNKNOWN")
        steps   = status_data.get("steps", {})

        badge_color = {
            "RUNNING":   "#f97316",
            "COMPLETED": "#4ade80",
            "FAILED":    "#f87171",
            "QUEUED":    "#60a5fa",
        }.get(overall, "#94a3b8")

        st.markdown(f"""
        <div class="card">
            <div class="card-header">Workflow Run · {run_id}</div>
            <span style="font-size:1.5rem; font-weight:800; color:{badge_color}">
                {overall}
            </span>
            &nbsp;&nbsp;
            <span style="color:#64748b; font-family:'JetBrains Mono',monospace; font-size:0.8rem;">
                Started: {status_data.get('start_time','—')}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Step progress table
        if steps:
            st.markdown("#### Pipeline Steps")
            rows = []
            for step_name, step_info in steps.items():
                s = step_info.get("status", "PENDING")
                rows.append({
                    "Step":    step_name,
                    "Status":  s,
                    "Started": step_info.get("start_time", "—"),
                    "Ended":   step_info.get("end_time", "—"),
                    "Error":   step_info.get("error", ""),
                })
            df = pd.DataFrame(rows)

            def colour_status(val):
                c = {"COMPLETED": "#4ade80", "FAILED": "#f87171",
                     "RUNNING": "#f97316", "PENDING": "#64748b"}.get(val, "#94a3b8")
                return f"color: {c}; font-weight: 600; font-family: 'JetBrains Mono', monospace;"

            styled = df.style.applymap(colour_status, subset=["Status"])
            st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.info("Enter a Run ID above, or trigger a new run from **🚀 Pipeline Control**.")

st.markdown("---")

# ── Bearing health cards ──────────────────────────────────────────────────────
st.markdown("#### Bearing Health at a Glance")

BEARINGS = ["Bearing1_5"]  # live bearings — extend as needed

cols = st.columns(len(BEARINGS) if BEARINGS else 1)

for i, bearing in enumerate(BEARINGS):
    records = get_latest_bearing_records(bearing, n=1)
    with cols[i]:
        if records and len(records) > 0:
            r  = records[0]
            pm = r.get("pm", {})
            mn = r.get("monitoring", {})
            rul_min  = r.get("rul_min", None)
            pm_status = pm.get("status", "unknown")

            badge_cls = {
                "healthy":  "badge-healthy",
                "warning":  "badge-warning",
                "critical": "badge-critical",
            }.get(pm_status, "badge-info")

            rul_display = f"{rul_min:.1f} min" if rul_min is not None else "—"
            drift_flag  = "⚠ DRIFT" if mn.get("drift_detected") else "OK"
            drift_col   = "#c084fc" if mn.get("drift_detected") else "#4ade80"

            st.markdown(f"""
            <div class="card">
                <div class="card-header">{bearing}</div>
                <div style="font-size:2rem; font-family:'JetBrains Mono',monospace;
                            font-weight:700; color:#f1f5f9;">
                    {rul_display}
                </div>
                <div style="margin-top:0.5rem;">
                    <span class="badge {badge_cls}">{pm_status.upper()}</span>
                    &nbsp;
                    <span style="font-family:'JetBrains Mono',monospace;
                                 font-size:0.75rem; color:{drift_col};">
                        {drift_flag}
                    </span>
                </div>
                <div style="margin-top:0.75rem; color:#64748b;
                            font-family:'JetBrains Mono',monospace; font-size:0.7rem;">
                    Burst #{r.get('burst_idx','?')} ·
                    Quality: {r.get('data_quality','—')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Mini RUL gauge
            if rul_min is not None:
                max_rul = 240
                pct     = min(rul_min / max_rul, 1.0)
                color   = "#f87171" if pct < 0.2 else "#fbbf24" if pct < 0.5 else "#4ade80"

                fig = go.Figure(go.Indicator(
                    mode  = "gauge+number",
                    value = rul_min,
                    number= {"suffix": " min", "font": {"size": 20, "family": "JetBrains Mono"}},
                    gauge = {
                        "axis":  {"range": [0, max_rul], "tickcolor": "#64748b",
                                  "tickfont": {"size": 9, "family": "JetBrains Mono"}},
                        "bar":   {"color": color},
                        "bgcolor": "#151820",
                        "bordercolor": "#1e2530",
                        "steps": [
                            {"range": [0, max_rul * 0.2],  "color": "#2d1515"},
                            {"range": [max_rul * 0.2, max_rul * 0.5], "color": "#2d2515"},
                            {"range": [max_rul * 0.5, max_rul],       "color": "#152d15"},
                        ],
                    },
                    title={"text": "RUL", "font": {"size": 12, "color": "#64748b",
                                                    "family": "JetBrains Mono"}},
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor ="rgba(0,0,0,0)",
                    height=220,
                    margin=dict(l=20, r=20, t=30, b=10),
                    font={"color": "#e2e8f0"},
                )
                st.plotly_chart(fig, use_container_width=True, key=f"gauge_{bearing}")
        else:
            st.markdown(f"""
            <div class="card">
                <div class="card-header">{bearing}</div>
                <span style="color:#64748b; font-family:'JetBrains Mono',monospace;
                             font-size:0.85rem;">No data yet — run pipeline first.</span>
            </div>
            """, unsafe_allow_html=True)

# ── Deployed model info ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Deployed Model")
model_info = get_deployed_model()
if model_info:
    c1, c2, c3 = st.columns(3)
    c1.metric("Model ID",  model_info.get("model_id", "—"))
    c2.metric("Version",   model_info.get("version",  "—"))
    c3.metric("Deployed",  model_info.get("deployed_at", "—")[:10] if model_info.get("deployed_at") else "—")
else:
    st.warning("No deployed model found. Train and deploy a model first.")

if auto_refresh:
    time.sleep(10)
    st.rerun()