"""
gui_fault_review.py
═══════════════════════════════════════════════════════════════════════════════
GUI 1 — Maintenance Worker: Confirm or Deny Faults

Fix #4: This is one of exactly two GUIs in the system.
        Runs as a standalone Streamlit app on a separate port from the RUL GUI.

Purpose : Maintenance worker reviews bearing alerts and confirms or denies
          faults. Confirmed faults trigger pre-production retraining.

Run     : streamlit run gui_fault_review.py --server.port 8501
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fault Review — Maintenance Worker",
    page_icon="🔧",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .status-critical { color: #f87171; font-weight: 700; }
    .status-warning  { color: #fbbf24; font-weight: 700; }
    .status-healthy  { color: #4ade80; font-weight: 700; }
    .card {
        background: #1e2530;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid #2d3748;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar — API URL ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Fault Review")
    st.markdown("**Maintenance Worker Interface**")
    st.markdown("---")
    api_url = st.text_input(
        "API Base URL",
        value=st.session_state.get("api_url", "http://localhost:8000"),
        label_visibility="visible",
    )
    st.session_state["api_url"] = api_url

    st.markdown("---")
    if st.button("🔌 Test Connection", use_container_width=True):
        try:
            r = requests.get(f"{api_url}/docs", timeout=3)
            if r.status_code == 200:
                st.success("✅ Connected")
            else:
                st.error(f"❌ HTTP {r.status_code}")
        except Exception as e:
            st.error(f"❌ {e}")

    st.markdown("---")
    st.caption("PHM MLOps · Fault Review GUI")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get(path: str) -> dict:
    try:
        r = requests.get(f"{api_url}{path}", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _post(path: str, payload: dict = None, timeout: int = 60) -> dict:
    try:
        r = requests.post(f"{api_url}{path}", json=payload or {}, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _get_serving_history(bearing_name: str, limit: int = 200) -> list:
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
st.markdown("# 🔧 Fault Review")
st.markdown("Review bearing alerts and confirm or deny faults.")
st.markdown("---")

# ── Queue status ───────────────────────────────────────────────────────────────
current_info = _get("/bearing/current")
queue_info   = _get("/bearing/queue")

if current_info.get("queue_exhausted"):
    st.success("✅ All live bearings in the queue have been processed.")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset Queue", type="primary", use_container_width=True):
            res = _post("/bearing/reset-queue")
            if "error" in res:
                st.error(res["error"])
            else:
                st.success(f"Queue reset → first bearing: **{res.get('first_bearing')}**")
                st.rerun()
    st.stop()

current_b = current_info.get("current_bearing", {})
if not current_b:
    st.warning("No live bearing currently queued. Trigger a workflow from the pipeline first.")
    st.stop()

current_name = current_b.get("name", "")

# ── Queue overview ─────────────────────────────────────────────────────────────
q   = queue_info.get("queue", [])
idx = queue_info.get("current_index", 0)

col_q1, col_q2, col_q3 = st.columns(3)
col_q1.metric("Current Bearing",  current_name)
col_q2.metric("Queue Position",   f"{idx + 1} / {len(q)}")
col_q3.metric("Bearing Status",   current_b.get("status", "—").upper())

# ── Serving history for current bearing ───────────────────────────────────────
st.markdown("---")
st.markdown(f"### 📈 Serving History — `{current_name}`")

col_c1, col_c2 = st.columns([3, 1])
with col_c1:
    n_records = st.slider("Records to display", 20, 500, 150, step=10)
with col_c2:
    alerts_only = st.checkbox("Alerts only", value=True)

records = _get_serving_history(current_name, limit=n_records)

if not records:
    st.warning(
        f"No serving history for **{current_name}**. "
        "Run the serving pipeline first."
    )
else:
    rows = []
    for r in records:
        pm  = r.get("pm",         {}) or {}
        inf = r.get("inference",  {}) or {}
        rows.append({
            "burst_idx":  r.get("burst_idx"),
            "timestamp":  r.get("timestamp", ""),
            "rul_min":    r.get("rul_min") or inf.get("rul_min"),
            "pm_status":  pm.get("status", "—"),
            "alert":      pm.get("alert", False),
            "action":     pm.get("recommended_action", "—"),
        })

    df = pd.DataFrame(rows).dropna(subset=["rul_min"]).sort_values("burst_idx")
    if alerts_only:
        df = df[df["alert"] == True]

    if df.empty:
        st.success("✅ No alerts in the selected window.")
    else:
        st.error(f"⚠ {len(df)} alert(s) detected.")
        st.dataframe(
            df.rename(columns={
                "burst_idx": "Burst", "timestamp": "Time",
                "rul_min": "RUL (min)", "pm_status": "Status",
                "alert": "Alert", "action": "Recommended Action",
            }),
            use_container_width=True,
            hide_index=True,
        )

# ── Fault decision ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### ✅ Fault Decision")

# Worker identity
col_w, col_r = st.columns(2)
with col_w:
    worker_name = st.text_input(
        "Maintenance Worker Name",
        value=st.session_state.get("worker_name", ""),
        placeholder="e.g. Jan van der Berg",
    )
    if worker_name:
        st.session_state["worker_name"] = worker_name

with col_r:
    run_id_input = st.text_input(
        "Workflow Run ID",
        value=st.session_state.get("active_run_id", ""),
        placeholder="e.g. run_20240101_120000",
        help="The run_id that produced the predictions.",
    )
    if run_id_input:
        st.session_state["active_run_id"] = run_id_input

note = st.text_area("Notes (optional)", placeholder="Describe observations...")

if not worker_name:
    st.info("Enter your name before confirming or denying a fault.")
    st.stop()

# ── Decision has not been made yet ────────────────────────────────────────────
decision = st.session_state.get("bearing_decision")

if decision is None:
    col_conf, col_deny = st.columns(2)

    with col_conf:
        st.markdown(
            "<div class='card'>"
            "<b>✅ Confirm Fault</b><br>"
            "Confirms the bearing has failed. Features are pushed to the "
            "Feature Store Mirrored and retraining begins automatically."
            "</div>",
            unsafe_allow_html=True,
        )
        rul_at_failure = st.number_input(
            "RUL at failure (seconds)", min_value=0.0, value=0.0, step=1.0
        )
        if st.button("✅ Confirm Fault", type="primary", use_container_width=True):
            if not run_id_input:
                st.error("Please enter the Workflow Run ID.")
            else:
                with st.spinner("Confirming fault and starting retraining..."):
                    res = _post("/bearing/confirm-fault", {
                        "bearing_name":   current_name,
                        "run_id":         run_id_input,
                        "rul_at_failure": rul_at_failure,
                        "worker_name":    worker_name,
                        "note":           note,
                    }, timeout=90)
                if "error" in res:
                    st.error(f"Error: {res['error']}")
                else:
                    st.session_state["bearing_decision"]      = "confirmed"
                    st.session_state["bearing_decision_name"] = current_name
                    st.rerun()

    with col_deny:
        st.markdown(
            "<div class='card'>"
            "<b>❌ Deny Fault</b><br>"
            "Marks the alert as a false positive. Features are NOT pushed "
            "to the Feature Store Mirrored. No retraining triggered."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("❌ Deny Fault", use_container_width=True):
            with st.spinner("Denying fault..."):
                res = _post("/bearing/deny-fault", {
                    "bearing_name": current_name,
                    "worker_name":  worker_name,
                    "note":         note,
                })
            if "error" in res:
                st.error(f"Error: {res['error']}")
            else:
                st.session_state["bearing_decision"]      = "denied"
                st.session_state["bearing_decision_name"] = current_name
                st.rerun()

# ── Decision confirmed — advance queue ────────────────────────────────────────
else:
    decided_name = st.session_state.get("bearing_decision_name", current_name)

    if decision == "confirmed":
        st.success(
            f"✅ Fault **confirmed** for `{decided_name}`. "
            "Retraining started in background — serving continues."
        )
    else:
        st.info(f"❌ Fault **denied** for `{decided_name}` (false positive).")

    st.markdown("---")
    if st.button("➡️ Advance to Next Bearing", type="primary",
                 use_container_width=True):
        with st.spinner("Advancing queue..."):
            res = _post("/bearing/continue", {
                "worker_name":     worker_name,
                "trigger_new_run": True,
            })
        if "error" in res:
            st.error(f"Error: {res['error']}")
        else:
            st.session_state.pop("bearing_decision",      None)
            st.session_state.pop("bearing_decision_name", None)
            next_b = res.get("next_bearing")
            if next_b:
                st.success(f"Queue advanced → next bearing: **{next_b}**")
            else:
                st.success("Queue exhausted — all bearings processed.")
            st.rerun()

# ── Reset queue ────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("⚠️ Reset Queue"):
    st.warning(
        "Resets the live bearing queue back to the first bearing. "
        "All bearing statuses are set to 'available'. No files are deleted."
    )
    if st.button("🔄 Reset Queue", use_container_width=True):
        res = _post("/bearing/reset-queue")
        if "error" in res:
            st.error(res["error"])
        else:
            st.session_state.pop("bearing_decision",      None)
            st.session_state.pop("bearing_decision_name", None)
            st.success(
                f"Queue reset → first bearing: **{res.get('first_bearing')}**"
            )
            st.rerun()