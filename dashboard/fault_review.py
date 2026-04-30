"""
fault_review_app.py
════════════════════════════════════════════════════════════════════════════════
Standalone Fault Review — run with:
    streamlit run fault_review_app.py

Maintenance technician interface:
  Step 1 — Enter technician name
  Step 2 — Confirm or Deny the fault  (buttons disabled until name entered)
  Step 3 — Continue button appears after a decision is made
"""

import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fault Review",
    page_icon="✅",
    layout="centered",
)

API_BASE = "http://localhost:8000"

# ── Session state ─────────────────────────────────────────────────────────────
if "bearing_decision" not in st.session_state:
    st.session_state["bearing_decision"] = None
if "bearing_decision_name" not in st.session_state:
    st.session_state["bearing_decision_name"] = None

# ── API helpers ───────────────────────────────────────────────────────────────

def _get(path: str) -> dict:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot reach backend at `{API_BASE}`. Is the API running?")
        return {}
    except Exception as e:
        st.error(f"❌ API error: {e}")
        return {}


def _post(path: str, payload: dict = None, timeout: int = 15) -> dict:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot reach backend at `{API_BASE}`.")
        return {"error": "connection failed"}
    except Exception as e:
        st.error(f"❌ API error: {e}")
        return {"error": str(e)}


def _confirm_fault(bearing_name: str, worker_name: str) -> dict:
    return _post("/bearing/confirm-fault", {
        "bearing_name": bearing_name,
        "worker_name":  worker_name,
    })


def _deny_fault(bearing_name: str, worker_name: str) -> dict:
    return _post("/bearing/deny-fault", {
        "bearing_name": bearing_name,
        "worker_name":  worker_name,
    })


def _continue_to_next(worker_name: str) -> dict:
    return _post("/bearing/continue", {
        "worker_name":     worker_name,
        "trigger_new_run": True,
    })


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ✅ Fault Review")
st.markdown("Maintenance technician fault confirmation interface.")
st.markdown("---")

# ── Fetch active bearing ──────────────────────────────────────────────────────
current_info  = _get("/bearing/current")
current_name  = None

if current_info:
    # Handle both possible response shapes from the API
    if current_info.get("queue_exhausted"):
        st.success("🏁 All bearings in the queue have been processed.")
        st.stop()

    bearing_obj  = current_info.get("current_bearing") or current_info
    current_name = bearing_obj.get("name") if isinstance(bearing_obj, dict) else None

if not current_name:
    st.warning(
        "No active bearing is currently being monitored. "
        "Start the pipeline from the Pipeline Control page first."
    )
    st.stop()

st.markdown(f"### Active Bearing: `{current_name}`")
st.markdown("---")

# ── Step 1: Technician name ───────────────────────────────────────────────────
st.markdown("#### Step 1 — Enter Your Name")
worker_name = st.text_input(
    "Maintenance Technician Name",
    placeholder="e.g. John Smith",
    label_visibility="collapsed",
)
name_ok = bool(worker_name.strip())

# ── Step 2: Confirm or Deny ───────────────────────────────────────────────────
decision      = st.session_state["bearing_decision"]
decision_name = st.session_state["bearing_decision_name"]

# Only show the buttons if no decision has been made for this bearing yet
if not (decision and decision_name == current_name):
    st.markdown("#### Step 2 — Confirm or Deny the Fault")

    if not name_ok:
        st.info("Enter your name above to enable the fault buttons.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "✅ Confirm Fault",
            use_container_width=True,
            type="primary",
            disabled=not name_ok,
        ):
            with st.spinner("Confirming fault..."):
                result = _confirm_fault(current_name, worker_name.strip())
            if "error" not in result:
                st.session_state["bearing_decision"]      = "confirmed"
                st.session_state["bearing_decision_name"] = current_name
                st.rerun()

    with col2:
        if st.button(
            "❌ Deny Fault",
            use_container_width=True,
            disabled=not name_ok,
        ):
            with st.spinner("Denying fault..."):
                result = _deny_fault(current_name, worker_name.strip())
            if "error" not in result:
                st.session_state["bearing_decision"]      = "denied"
                st.session_state["bearing_decision_name"] = current_name
                st.rerun()

# ── Step 3: Continue (only after a decision) ──────────────────────────────────
# Re-read after potential rerun
decision      = st.session_state["bearing_decision"]
decision_name = st.session_state["bearing_decision_name"]

if decision and decision_name == current_name:
    label_str = "confirmed ✅" if decision == "confirmed" else "denied ❌"
    st.markdown("---")
    st.success(
        f"**{current_name}** has been **{label_str}** "
        f"by **{worker_name.strip() or 'technician'}**."
    )

    st.markdown("#### Step 3 — Continue")

    if st.button("▶ Continue → New Bearing", use_container_width=True, type="primary"):
        with st.spinner("Advancing to next bearing..."):
            result = _continue_to_next(worker_name.strip() or "unknown")

        if "error" in result:
            st.error(f"Error: {result['error']}")
        elif result.get("status") == "queue_exhausted":
            st.success("🏁 All bearings in the queue have been processed!")
            st.session_state["bearing_decision"]      = None
            st.session_state["bearing_decision_name"] = None
            st.rerun()
        else:
            next_b   = result.get("next_bearing", {})
            next_rid = result.get("run_id", "")
            st.success(
                f"▶ Now monitoring **{next_b.get('name', '?')}**. "
                + (f"Run `{next_rid}` started." if next_rid else "")
            )
            st.session_state["bearing_decision"]      = None
            st.session_state["bearing_decision_name"] = None
            st.rerun()