"""
dashboard/pages/5_✅_Fault_Review.py
════════════════════════════════════════════════════════════════════════════════
Maintenance Worker Interface — Confirm or Deny faults, then advance to the
next bearing in the live queue.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dashboard.api_client import get_serving_history


# ── API helpers ───────────────────────────────────────────────────────────────

def _api_url() -> str:
    return st.session_state.get("api_url", "http://localhost:8000")


def _get(path: str) -> dict:
    try:
        r = requests.get(f"{_api_url()}{path}", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _post(path: str, payload: dict = None, timeout: int = 30) -> dict:
    try:
        r = requests.post(f"{_api_url()}{path}", json=payload or {}, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _confirm_fault(bearing_name, run_id, rul_at_failure, worker_name, note):
    return _post("/bearing/confirm-fault", {
        "bearing_name":   bearing_name,
        "run_id":         run_id,
        "rul_at_failure": rul_at_failure,
        "worker_name":    worker_name,
        "note":           note,
    })


def _deny_fault(bearing_name, worker_name, note):
    return _post("/bearing/deny-fault", {
        "bearing_name": bearing_name,
        "worker_name":  worker_name,
        "note":         note,
    })


def _continue_to_next(worker_name):
    return _post("/bearing/continue", {
        "worker_name":     worker_name,
        "trigger_new_run": True,
    })


def _reset_queue():
    return _post("/bearing/reset-queue")


# ── Page ──────────────────────────────────────────────────────────────────────

st.markdown("# ✅ Fault Review")
st.markdown("Review pipeline alerts and confirm or deny faults.")
st.markdown("---")

# ── Current bearing + queue status ───────────────────────────────────────────
current_info = _get("/bearing/current")
queue_info   = _get("/bearing/queue")

if current_info.get("queue_exhausted"):
    st.success("✅ All live bearings in the queue have been processed.")

    st.markdown("---")
    st.markdown("### Reset Queue")
    st.markdown("Reset the queue to start again from the first bearing.")
    if st.button("🔄 Reset Queue → Start from Beginning", type="primary",
                 use_container_width=True):
        result = _reset_queue()
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.success(
                f"✅ Queue reset. First bearing: **{result.get('first_bearing')}**. "
                "Trigger a new workflow from Pipeline Control."
            )
            st.rerun()
    st.stop()

current_b = current_info.get("current_bearing", {})
if not current_b:
    st.warning("No live bearing found. Check the bearing queue configuration.")
    st.stop()

current_name = current_b.get("name", "Unknown")
remaining    = current_info.get("remaining", "?")
queue_len    = current_info.get("queue_length", "?")

# ── Queue header ──────────────────────────────────────────────────────────────
col_status, col_queue, col_reset = st.columns([2, 4, 1])

with col_status:
    st.markdown(f"""
    <div style="background:#1e2530; border-radius:8px; padding:14px 18px;
                border-left: 4px solid #f97316;">
        <div style="color:#64748b; font-size:0.7rem;
                    font-family:'JetBrains Mono',monospace;">
            CURRENT LIVE BEARING
        </div>
        <div style="color:#f97316; font-size:1.2rem; font-weight:700;
                    font-family:'JetBrains Mono',monospace; margin-top:4px;">
            {current_name}
        </div>
        <div style="color:#94a3b8; font-size:0.72rem; margin-top:4px;">
            {remaining} of {queue_len} remaining
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_queue:
    q_items = queue_info.get("queue", [])
    progress_html = "<div style='display:flex; gap:6px; flex-wrap:wrap; padding-top:6px;'>"
    for item in q_items:
        color = (
            "#f97316" if item["is_current"] else
            "#4ade80" if item["status"] in ("confirmed", "denied") else
            "#475569"
        )
        label = "▶ " if item["is_current"] else ""
        progress_html += (
            f"<span style='background:{color}22; border:1px solid {color}; "
            f"color:{color}; border-radius:4px; padding:2px 8px; "
            f"font-size:0.68rem; font-family:monospace;'>"
            f"{label}{item['name']}</span>"
        )
    progress_html += "</div>"
    st.markdown("**Bearing Queue**")
    st.markdown(progress_html, unsafe_allow_html=True)

with col_reset:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Reset", use_container_width=True,
                 help="Reset queue back to the first bearing"):
        if "confirm_reset" not in st.session_state:
            st.session_state["confirm_reset"] = True
            st.rerun()

# Confirmation dialog for reset
if st.session_state.get("confirm_reset"):
    st.warning(
        "⚠️ This will reset the queue to **Bearing 1** and mark all live bearings "
        "as available. Any unconfirmed fault data will not be affected. Continue?"
    )
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("✅ Yes, Reset Queue", type="primary", use_container_width=True):
            result = _reset_queue()
            st.session_state.pop("confirm_reset", None)
            st.session_state["bearing_decision"]      = None
            st.session_state["bearing_decision_name"] = None
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success(
                    f"Queue reset. Now starting from "
                    f"**{result.get('first_bearing', '?')}**. "
                    "Trigger a new workflow from Pipeline Control."
                )
            st.rerun()
    with col_no:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.pop("confirm_reset", None)
            st.rerun()

st.markdown("---")

# ── Worker name & run_id ──────────────────────────────────────────────────────
col_w, col_r = st.columns(2)
with col_w:
    worker_name = st.text_input("Maintenance Worker Name", value="Worker",
                                placeholder="e.g. Jan van der Berg")
with col_r:
    run_id_input = st.text_input(
        "Workflow Run ID",
        value=st.session_state.get("active_run_id", ""),
        placeholder="e.g. run_20240101_120000",
        help="The run_id that produced the predictions (from Pipeline Control page).",
    )

# ── Fetch serving history ─────────────────────────────────────────────────────
n_records        = st.slider("Max records to show", 20, 500, 100, step=10)
show_alerts_only = st.checkbox("Show alerts only", value=True)

records = get_serving_history(current_name, limit=n_records)

if not records:
    st.warning(
        f"No serving history for **{current_name}**. "
        "Run the serving pipeline first via Pipeline Control."
    )
    st.stop()

rows = []
for r in records:
    pm  = r.get("pm", {}) or {}
    inf = r.get("inference", {}) or {}
    rows.append({
        "burst_idx":   r.get("burst_idx"),
        "rul_s":       r.get("rul_s") or inf.get("rul_s", 0.0),
        "rul_min":     r.get("rul_min") or inf.get("rul_min", 0.0),
        "pm_status":   pm.get("status", "—"),
        "alert":       pm.get("alert", False),
        "action":      pm.get("recommended_action", "—"),
        "data_quality":r.get("data_quality", "clean"),
    })

df = pd.DataFrame(rows).sort_values("burst_idx", ascending=False)
if show_alerts_only:
    df = df[df["alert"] == True]

# ── Summary metrics ───────────────────────────────────────────────────────────
total_alerts = (pd.DataFrame(rows)["alert"] == True).sum()
latest_rul   = pd.DataFrame(rows).dropna(subset=["rul_min"])
latest_rul_v = latest_rul["rul_min"].iloc[-1] if not latest_rul.empty else None

m1, m2, m3 = st.columns(3)
m1.metric("Total Alerts",  total_alerts)
m2.metric("Latest RUL",    f"{latest_rul_v:.1f} min" if latest_rul_v is not None else "—")
m3.metric("Records shown", len(df))

st.markdown("---")

# ── Per-burst alert table ─────────────────────────────────────────────────────
if total_alerts == 0:
    st.success(f"✅ No alerts for **{current_name}**. Bearing appears healthy.")
    st.info("Click **Continue** below to keep monitoring or move to the next bearing.")
else:
    st.markdown(f"### Alerts for **{current_name}** ({len(df)} shown)")
    for _, row in df.iterrows():
        burst   = row["burst_idx"]
        rul_str = f"{row['rul_min']:.1f} min" if pd.notna(row.get("rul_min")) else "—"
        with st.expander(
            f"Burst #{burst} · RUL: {rul_str} · {row['pm_status'].upper()}",
            expanded=False,
        ):
            st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;
                        color:#94a3b8; line-height:1.8;">
                <b style="color:#e2e8f0;">Burst:</b> {burst}<br>
                <b style="color:#e2e8f0;">RUL:</b> {rul_str}<br>
                <b style="color:#e2e8f0;">PM Status:</b> {row['pm_status']}<br>
                <b style="color:#e2e8f0;">Recommended Action:</b> {row['action']}<br>
                <b style="color:#e2e8f0;">Data Quality:</b> {row['data_quality']}
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# ── Bearing-level fault decision ──────────────────────────────────────────────
st.markdown("### Bearing Fault Decision")
st.markdown(
    f"Confirm or deny the fault for **{current_name}**. "
    "Confirming sends the re-labelled features to the Feature Store and "
    "triggers a model retrain when Continue is clicked."
)

col_note, col_rul = st.columns(2)
with col_note:
    fault_note = st.text_area(
        "Note (optional)",
        placeholder="Describe the fault or reason for denial...",
        height=80,
    )
with col_rul:
    rul_at_failure = st.number_input(
        "Confirmed RUL at failure (seconds)",
        min_value=0.0,
        value=float(latest_rul_v * 60) if latest_rul_v is not None else 0.0,
        step=10.0,
        help="Actual remaining life when fault was declared. Used to re-label features.",
    )

if "bearing_decision" not in st.session_state:
    st.session_state["bearing_decision"] = None
if "bearing_decision_name" not in st.session_state:
    st.session_state["bearing_decision_name"] = None

btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    if st.button("✅ Confirm Fault", use_container_width=True, type="primary"):
        if not run_id_input.strip():
            st.error("Please enter the Workflow Run ID.")
        else:
            with st.spinner("Confirming fault and pushing to Feature Store..."):
                result = _confirm_fault(
                    bearing_name   = current_name,
                    run_id         = run_id_input.strip(),
                    rul_at_failure = rul_at_failure,
                    worker_name    = worker_name,
                    note           = fault_note,
                )
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.session_state["bearing_decision"]      = "confirmed"
                st.session_state["bearing_decision_name"] = current_name
                st.success(
                    f"✅ Fault confirmed for **{current_name}**. "
                    "Features pushed to Feature Store."
                )

with btn_col2:
    if st.button("❌ Deny Fault (False Positive)", use_container_width=True):
        with st.spinner("Denying fault..."):
            result = _deny_fault(
                bearing_name = current_name,
                worker_name  = worker_name,
                note         = fault_note,
            )
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.session_state["bearing_decision"]      = "denied"
            st.session_state["bearing_decision_name"] = current_name
            st.warning(
                f"❌ Fault denied for **{current_name}**. "
                "Features will NOT be sent to Feature Store."
            )

# ── Continue button ───────────────────────────────────────────────────────────
st.markdown("---")

decision      = st.session_state.get("bearing_decision")
decision_name = st.session_state.get("bearing_decision_name")

if decision and decision_name == current_name:
    label_str = "confirmed ✅" if decision == "confirmed" else "denied ❌"
    st.info(
        f"**{current_name}** has been **{label_str}**. "
        "Click **Continue** to place a new bearing and start monitoring it."
    )

    if st.button("▶ Continue → New Bearing", use_container_width=True, type="primary"):
        with st.spinner("Advancing to next bearing and starting ingestion..."):
            result = _continue_to_next(worker_name)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        elif result.get("status") == "queue_exhausted":
            st.success("🏁 All bearings in the queue have been processed!")
            st.session_state["bearing_decision"] = None
            st.rerun()
        else:
            next_b   = result.get("next_bearing", {})
            next_rid = result.get("run_id", "")
            st.success(
                f"▶ Now monitoring **{next_b.get('name', '?')}**. "
                + (f"Background run `{next_rid}` started — "
                   "check Pipeline Control for progress." if next_rid else "")
            )
            st.session_state["bearing_decision"]      = None
            st.session_state["bearing_decision_name"] = None
            st.session_state["active_run_id"]         = next_rid
            st.rerun()
else:
    st.markdown(
        "_Make a fault decision above (Confirm or Deny) before continuing._"
    )
    st.button("▶ Continue → New Bearing", use_container_width=True,
              disabled=True, help="Confirm or Deny the fault first.")