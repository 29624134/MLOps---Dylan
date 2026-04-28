"""
dashboard/pages/2_🚀_Pipeline_Control.py
════════════════════════════════════════════════════════════════════════════════
Trigger the workflow orchestrator, monitor step-by-step progress, and manage
live serving runs.
"""

import time
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dashboard.api_client import trigger_workflow, get_workflow_status, run_bearing_pipeline

st.markdown("# 🚀 Pipeline Control")
st.markdown("Trigger and monitor the MLOps workflow orchestrator.")
st.markdown("---")

# ── Trigger full workflow ─────────────────────────────────────────────────────
st.markdown("### Full Workflow Trigger")
st.markdown("Runs all orchestrator phases: ingest → feature extraction → training → serving pipeline.")

with st.expander("⚙️ Config Overrides (optional)", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        window_size   = st.number_input("Window Size",   value=40,    min_value=5,   max_value=200)
        burst_period  = st.number_input("Burst Period (s)", value=10.0, min_value=0.1, step=0.5)
    with col2:
        critical_thresh = st.number_input("Critical Threshold (s)", value=3600,  step=600)
        warning_thresh  = st.number_input("Warning Threshold (s)",  value=14400, step=600)
    realtime = st.checkbox("Realtime mode (sleep between bursts)", value=False)

config_overrides = {
    "window_size":          int(window_size),
    "burst_period":         float(burst_period),
    "critical_threshold_s": int(critical_thresh),
    "warning_threshold_s":  int(warning_thresh),
    "realtime":             realtime,
}

col_trig, col_info = st.columns([2, 5])
with col_trig:
    trigger_clicked = st.button("▶ Trigger Workflow", use_container_width=True)

if trigger_clicked:
    with st.spinner("Submitting workflow..."):
        result = trigger_workflow("rul_prediction", config_overrides)
    if result:
        run_id = result.get("run_id", "")
        st.session_state["active_run_id"] = run_id
        st.success(f"✅ Workflow queued! Run ID: `{run_id}`")
        st.info("Monitor progress below — paste the run ID into the status section.")

st.markdown("---")

# ── Status monitor ────────────────────────────────────────────────────────────
st.markdown("### Workflow Status Monitor")

col_id, col_poll = st.columns([4, 1])
with col_id:
    run_id_input = st.text_input(
        "Run ID",
        value=st.session_state.get("active_run_id", ""),
        placeholder="run_20260413_abc12345",
        label_visibility="collapsed",
    )
with col_poll:
    poll = st.button("🔄 Refresh", use_container_width=True)

if run_id_input:
    st.session_state["active_run_id"] = run_id_input
    status_data = get_workflow_status(run_id_input)

    if status_data:
        overall   = status_data.get("status", "UNKNOWN")
        steps     = status_data.get("steps", {})
        start     = status_data.get("start_time", "—")
        end       = status_data.get("end_time", "—")

        # Header
        badge_color = {
            "RUNNING":   "#f97316",
            "COMPLETED": "#4ade80",
            "FAILED":    "#f87171",
            "QUEUED":    "#60a5fa",
        }.get(overall, "#94a3b8")

        c1, c2, c3 = st.columns(3)
        c1.metric("Status",  overall)
        c2.metric("Started", start[:19] if start != "—" else "—")
        c3.metric("Ended",   end[:19]   if end   != "—" else "—")

        # Step progress bars
        st.markdown("#### Step-by-Step Progress")

        STATUS_ORDER = ["PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED"]
        STEP_ICONS = {
            "COMPLETED": "✅",
            "RUNNING":   "🔄",
            "FAILED":    "❌",
            "PENDING":   "⏳",
            "SKIPPED":   "⏭",
        }

        for step_name, step_info in steps.items():
            s     = step_info.get("status", "PENDING")
            icon  = STEP_ICONS.get(s, "•")
            err   = step_info.get("error", "")
            s_col = {
                "COMPLETED": "#4ade80",
                "RUNNING":   "#f97316",
                "FAILED":    "#f87171",
                "PENDING":   "#64748b",
                "SKIPPED":   "#94a3b8",
            }.get(s, "#94a3b8")

            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:1rem; padding:0.5rem 0;
                        border-bottom:1px solid #1e2530;">
                <span style="font-size:1.1rem;">{icon}</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;
                             flex:1; color:#e2e8f0;">{step_name}</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem;
                             color:{s_col}; font-weight:600;">{s}</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.7rem;
                             color:#64748b;">{step_info.get('start_time','')[:19]}</span>
            </div>
            """, unsafe_allow_html=True)

            if err:
                st.error(f"Error in `{step_name}`: {err}")

        # Auto-poll if running
        if overall == "RUNNING":
            time.sleep(5)
            st.rerun()

st.markdown("---")

# ── Live bearing serving ──────────────────────────────────────────────────────
st.markdown("### Live Bearing Serving")
st.markdown("Run the 4-stage Serving Pipeline on a live bearing (Bearing1_5 by default).")

col_b, col_m, col_rt = st.columns(3)
with col_b:
    bearing_name = st.text_input("Bearing Name", value="Bearing1_5")
with col_m:
    max_bursts = st.number_input("Max Bursts (0 = all)", value=0, min_value=0, step=10)
with col_rt:
    st.markdown("<br>", unsafe_allow_html=True)
    realtime_mode = st.checkbox("Realtime", value=False)

if st.button("▶ Start Live Serving", use_container_width=False):
    with st.spinner(f"Launching serving pipeline for {bearing_name}..."):
        result = run_bearing_pipeline(
            bearing_name,
            realtime=realtime_mode,
            max_bursts=max_bursts if max_bursts > 0 else None,
        )
    if result:
        rid = result.get("run_id", "")
        st.session_state["active_run_id"] = rid
        st.success(f"✅ Started! Run ID: `{rid}`")
        st.info("Check **📊 RUL Monitor** for live predictions.")