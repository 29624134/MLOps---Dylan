"""
dashboard/pages/7_retrain.py
════════════════════════════════════════════════════════════════════════════════
Dashboard → AutoTrain Retraining Trigger
(Diagram: Dashboard -.Retraining Triggered.-> AutoTrain)

This page implements the dashed arrow from the Dashboard to AutoTrain in the
V9 diagram.  It lets a Maintenance Worker or engineer manually trigger a
Pre-Production retraining run directly from the dashboard — separate from the
automatic retraining that fires on fault confirmation.

Sections
────────
1. Manual Retraining Trigger  — fires POST /preprod/trigger via API
2. Active Preprod Runs        — shows status of all recent preprod run_ids
3. Export Flush               — flush Serving History → Audit CSV (AuditSvc)
4. Export Paths               — display where export files are written to
"""

import time
import requests
import streamlit as st
import pandas as pd
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dashboard.api_client import get_workflow_status  # reuse existing helper


# ── API helper ────────────────────────────────────────────────────────────────

def _api_url() -> str:
    return st.session_state.get("api_url", "http://localhost:8000")


def _post(path: str, payload: dict = None, timeout: int = 30) -> dict:
    try:
        r = requests.post(f"{_api_url()}{path}", json=payload or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}


def _get(path: str, timeout: int = 10) -> dict:
    try:
        r = requests.get(f"{_api_url()}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown("# 🔁 Retraining & Export Control")
st.markdown(
    "Manually trigger Pre-Production retraining and manage the "
    "Export Service output. "
    "_(Dashboard -.Retraining Triggered.-> AutoTrain in the V9 diagram)_"
)
st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# Section 1: Manual Retraining Trigger
# ════════════════════════════════════════════════════════════════════════════

st.markdown("### 🚀 Trigger Pre-Production Retraining")
st.markdown(
    "Starts a background retraining run using all train-role bearings plus "
    "any confirmed fault data already in the Feature Store (Mirrored). "
    "The Serving Pipeline continues uninterrupted — the new model will "
    "hot-swap automatically if it outperforms the current champion."
)

col_btn, col_info = st.columns([2, 5])
with col_btn:
    trigger_clicked = st.button(
        "▶ Trigger Retraining",
        use_container_width=True,
        type="primary",
    )

if trigger_clicked:
    with st.spinner("Submitting retraining request..."):
        result = _post("/preprod/trigger")

    if "error" in result:
        st.error(f"❌ Failed to trigger retraining: {result['error']}")
    else:
        preprod_run_id = result.get("preprod_run_id", "")
        st.success(
            f"✅ Retraining started!  \n"
            f"**Run ID:** `{preprod_run_id}`  \n"
            f"PID: `{result.get('preprod_pid', '—')}`"
        )
        # Store so the status monitor below pre-fills it
        st.session_state["preprod_run_id"] = preprod_run_id
        st.info(
            "The serving pipeline will continue making predictions. "
            "When retraining completes, the new model is evaluated — "
            "if it's better, it replaces the current champion automatically."
        )

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# Section 2: Retraining Run Status Monitor
# ════════════════════════════════════════════════════════════════════════════

st.markdown("### 📊 Retraining Run Status")

col_rid, col_refresh = st.columns([4, 1])
with col_rid:
    preprod_run_id = st.text_input(
        "Preprod Run ID",
        value=st.session_state.get("preprod_run_id", ""),
        placeholder="preprod_20260423_123456_abc123",
        label_visibility="collapsed",
    )
with col_refresh:
    refresh = st.button("🔄 Refresh", use_container_width=True)

if preprod_run_id:
    st.session_state["preprod_run_id"] = preprod_run_id
    status_data = get_workflow_status(preprod_run_id)

    if status_data and "error" not in status_data:
        overall = status_data.get("status", "UNKNOWN")
        steps   = status_data.get("steps", {})
        start   = status_data.get("start_time", "—")
        end     = status_data.get("end_time",   "—")

        STATUS_COLOUR = {
            "RUNNING":   "#f97316",
            "COMPLETED": "#4ade80",
            "COMPLETE":  "#4ade80",
            "FAILED":    "#f87171",
            "QUEUED":    "#60a5fa",
        }
        STEP_ICONS = {
            "COMPLETED": "✅", "COMPLETE": "✅",
            "RUNNING":   "🔄",
            "FAILED":    "❌",
            "PENDING":   "⏳",
            "SKIPPED":   "⏭",
        }

        c1, c2, c3 = st.columns(3)
        c1.metric("Status",  overall)
        c2.metric("Started", start[:19] if start != "—" else "—")
        c3.metric("Ended",   end[:19]   if end   != "—" else "—")

        if steps:
            st.markdown("#### Steps")
            for step_name, step_info in steps.items():
                s    = step_info.get("status", "PENDING")
                icon = STEP_ICONS.get(s, "•")
                err  = step_info.get("error", "")
                col  = STATUS_COLOUR.get(s, "#94a3b8")

                st.markdown(
                    f"""<div style="display:flex; align-items:center; gap:1rem;
                        padding:0.4rem 0; border-bottom:1px solid #1e2530;">
                        <span style="font-size:1.2rem;">{icon}</span>
                        <span style="font-family:JetBrains Mono,monospace;
                              color:{col}; min-width:100px;">{s}</span>
                        <span style="color:#e2e8f0;">{step_name}</span>
                        {'<span style="color:#f87171; font-size:0.85rem;">— ' + err[:120] + '</span>' if err else ''}
                    </div>""",
                    unsafe_allow_html=True,
                )
    elif status_data and "error" in status_data:
        st.warning(f"Run `{preprod_run_id}` not found or API unreachable.")
    else:
        st.info("Enter a run ID above to check status.")

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# Section 3: Export Service — Audit Flush
# (AuditSvc → External Data Destination)
# ════════════════════════════════════════════════════════════════════════════

st.markdown("### 📤 Export Service — Audit Flush")
st.markdown(
    "Flush Serving History records to the external CSV destination "
    "_(ServHistory → AuditSvc → External)_.  Useful if the pipeline was "
    "running without the Export Service enabled, or to re-export a specific bearing."
)

col_b, col_lim, col_flush = st.columns([3, 1, 1])
with col_b:
    flush_bearing = st.selectbox(
        "Bearing",
        ["Bearing1_5", "Bearing1_4", "Bearing1_6", "Bearing1_7", "Bearing2_3"],
        label_visibility="collapsed",
    )
with col_lim:
    flush_limit = st.number_input("Records", value=500, min_value=10, max_value=5000, step=50)
with col_flush:
    flush_clicked = st.button("⬇ Flush", use_container_width=True)

if flush_clicked:
    with st.spinner(f"Flushing {flush_bearing} audit records..."):
        result = _post(
            "/audit/flush",
            {"bearing_name": flush_bearing, "limit": int(flush_limit)},
        )
    if "error" in result:
        st.error(f"❌ Flush failed: {result['error']}")
    else:
        st.success(
            f"✅ Exported **{result.get('exported', 0)}** records "
            f"for {flush_bearing} to the external CSV destination."
        )

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# Section 4: Export Paths
# ════════════════════════════════════════════════════════════════════════════

st.markdown("### 📁 Export Destination Paths")

paths = _get("/export/paths")
if "error" in paths:
    st.warning(f"Could not load export paths: {paths['error']}")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("RUL CSV",   paths.get("rul_csv")   or "disabled")
        st.metric("Audit CSV", paths.get("audit_csv") or "disabled")
    with col2:
        st.metric("JSON Snapshots", paths.get("json_snapshots") or "disabled")
        st.metric("Output Directory", paths.get("output_dir", "—"))