"""
fault_review_app.py
════════════════════════════════════════════════════════════════════════════════
Standalone Fault Review — run with:
    streamlit run fault_review_app.py

Maintenance technician interface:
  Step 0 — Select which bearing group to review (when multiple groups active)
  Step 1 — Enter technician name
  Step 2 — Confirm or Deny the fault  (buttons disabled until name entered)
  Step 3 — Continue button appears after a decision is made
════════════════════════════════════════════════════════════════════════════════
"""

import requests
import streamlit as st

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


def _continue_to_next(bearing_name: str, worker_name: str) -> dict:
    return _post("/bearing/continue", {
        "bearing_name":    bearing_name,
        "worker_name":     worker_name,
        "trigger_new_run": True,
    })


def _parse_current_bearings(info: dict) -> dict:
    """
    Parse /bearing/current response into {group: bearing_name} dict.

    Handles both response shapes:
      New (multi-group): {"group_1": "Bearing1_2", "group_2": "Bearing2_2", ...}
      Legacy (single):   {"current_bearing": {"name": "Bearing1_2"}, ...}
    """
    result = {}

    # New multi-group format: keys like "group_1", "group_2", "group_3"
    for key, value in info.items():
        if key.startswith("group_") and value:
            group_id = key.replace("group_", "")
            result[group_id] = value

    # Legacy single-bearing format fallback
    if not result:
        bearing_obj = info.get("current_bearing") or info
        if isinstance(bearing_obj, dict):
            name = bearing_obj.get("name")
        else:
            name = bearing_obj if isinstance(bearing_obj, str) else None
        if name:
            result["1"] = name

    return result  # e.g. {"1": "Bearing1_2", "2": "Bearing2_4", "3": "Bearing3_2"}


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# ✅ Fault Review")
st.markdown("Maintenance technician fault confirmation interface.")
st.markdown("---")

# ── Fetch active bearings ─────────────────────────────────────────────────────
current_info = _get("/bearing/current")

if not current_info:
    st.warning(
        "No active bearing is currently being monitored. "
        "Start the pipeline from the Pipeline Control page first."
    )
    st.stop()

if current_info.get("queue_exhausted"):
    st.success("🏁 All bearings in the queue have been processed.")
    st.stop()

active_bearings = _parse_current_bearings(current_info)

if not active_bearings:
    st.warning(
        "No active bearing is currently being monitored. "
        "Start the pipeline from the Pipeline Control page first."
    )
    st.stop()

# ── Group / bearing selection ─────────────────────────────────────────────────
if len(active_bearings) == 1:
    # Only one group active — no need to ask
    selected_group = list(active_bearings.keys())[0]
    current_name   = active_bearings[selected_group]
else:
    # Multiple groups active — let the tech pick which one to review
    st.markdown("#### Select Bearing Group to Review")
    options = {
        f"Group {g} — {name}": (g, name)
        for g, name in sorted(active_bearings.items())
    }
    chosen = st.selectbox(
        "Active bearing groups:",
        list(options.keys()),
        label_visibility="collapsed",
    )
    selected_group, current_name = options[chosen]

st.markdown(f"### Active Bearing: `{current_name}` &nbsp; (Group {selected_group})")
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
            else:
                st.error(f"Failed to confirm fault: {result.get('error')}")

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
            else:
                st.error(f"Failed to deny fault: {result.get('error')}")

# ── Step 3: Continue ──────────────────────────────────────────────────────────
decision      = st.session_state["bearing_decision"]
decision_name = st.session_state["bearing_decision_name"]

if decision and decision_name == current_name:
    label_str = "confirmed ✅" if decision == "confirmed" else "denied ❌"
    st.markdown("---")
    st.success(
        f"**{current_name}** has been **{label_str}** "
        f"by **{worker_name.strip() or 'technician'}**."
    )

    if decision == "confirmed":
        st.info(
            f"Group {selected_group} retraining has started automatically in the background. "
            f"Other groups continue serving uninterrupted."
        )

    st.markdown("#### Step 3 — Continue")

    if st.button("▶ Continue → Next Bearing", use_container_width=True, type="primary"):
        with st.spinner("Advancing to next bearing..."):
            result = _continue_to_next(current_name, worker_name.strip() or "unknown")

        if "error" in result:
            st.error(f"Error: {result['error']}")
        elif result.get("status") in ("queue_exhausted", "continued"):
            if result.get("status") == "queue_exhausted":
                st.success("🏁 All bearings in the queue have been processed!")
            else:
                run_id = result.get("run_id", "")
                st.success(
                    f"▶ Queue advanced. "
                    + (f"Run `{run_id}` started." if run_id else "")
                )
            st.session_state["bearing_decision"]      = None
            st.session_state["bearing_decision_name"] = None
            st.rerun()
        else:
            # Legacy response shape with next_bearing
            next_b  = result.get("next_bearing") or {}
            next_name = next_b.get("name") if isinstance(next_b, dict) else str(next_b)
            run_id  = result.get("run_id", "")
            st.success(
                f"▶ Now monitoring **{next_name or '?'}**. "
                + (f"Run `{run_id}` started." if run_id else "")
            )
            st.session_state["bearing_decision"]      = None
            st.session_state["bearing_decision_name"] = None
            st.rerun()