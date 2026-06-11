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

Decision state is persisted per-bearing in session_state AND is restored
from the bearing's actual status in bearings.json on every page load.
This means navigating away and back correctly shows the Continue button
without requiring the tech to re-confirm.
════════════════════════════════════════════════════════════════════════════════
"""

import requests
import streamlit as st

st.set_page_config(
    page_title="Fault Review",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
    }
    [data-testid="stHeader"] {
        background-color: #ffffff;
    }
    .stApp, .stMarkdown, p, label, h1, h2, h3, h4, h5, h6 {
        color: #000000;
    }
    /* Border around the whole page content */
    [data-testid="stMainBlockContainer"] {
        border: 2px solid #000000 !important;
        border-radius: 8px !important;
        padding: 2rem !important;
        margin-top: 1rem !important;
    }
    /* All headings and body text uniform size */
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown h5,
    .stMarkdown h6,
    .stMarkdown p,
    .stMarkdown li,
    label {
        font-size: 20px !important;
        font-weight: 600 !important;
        font-family: "Source Sans Pro", sans-serif !important;
    }
    .stMarkdown p {
        font-weight: 600 !important;
    }
    /* Black border on all buttons + font sizing */
    .stButton > button {
        border: 2px solid #000000 !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        font-family: "Source Sans Pro", sans-serif !important;
        color: #ffffff !important;
    }
    /* Confirm Fault (primary button) → green */
    .stButton > button[kind="primary"] {
        background-color: #2e7d32 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1b5e20 !important;
        border-color: #000000 !important;
    }
    /* Deny Fault (secondary button) → red */
    .stButton > button[kind="secondary"] {
        background-color: #c62828 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #8e0000 !important;
        border-color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_BASE = "http://localhost:8000"

# ── Session state ─────────────────────────────────────────────────────────────
# decisions: dict[bearing_name -> "confirmed" | "denied"]
# Stores decisions for ALL bearings so navigating between groups works.
if "decisions" not in st.session_state:
    st.session_state["decisions"] = {}

# ── API helpers ───────────────────────────────────────────────────────────────

def _get(path: str) -> dict:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach backend at `{API_BASE}`. Is the API running?")
        return {}
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


def _post(path: str, payload: dict = None, timeout: int = 15) -> dict:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach backend at `{API_BASE}`.")
        return {"error": "connection failed"}
    except Exception as e:
        st.error(f"API error: {e}")
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


def _get_bearing_status(bearing_name: str) -> str:
    """
    Fetch the real status of a bearing from the API (reads bearings.json).
    Returns one of: available / confirmed / denied / ingested / extracted / error
    Falls back to "available" if the API call fails.
    """
    queue_info = _get("/bearing/queue")
    if not queue_info:
        return "available"

    # /bearing/queue returns group_N keys, each with a queue list.
    # We also call /bearing/current to get the current bearing per group,
    # but the status lives in bearings.json which we read via the queue endpoint.
    # Simplest approach: call the dedicated status endpoint if it exists,
    # otherwise infer from the decisions dict already in session_state.
    # Since we don't have a /bearing/{name}/status endpoint, we use the
    # /bearing/queue response to detect which bearings have been actioned.
    # The real status is stored in session_state["decisions"] which we
    # now persist and restore below — so this function is a fallback.
    return st.session_state["decisions"].get(bearing_name, "available")


def _restore_decision_from_api(bearing_name: str) -> None:
    """
    On page load, check whether this bearing has already been confirmed or
    denied by reading its status from the API bearing queue.

    The API's /bearing/queue endpoint doesn't expose individual bearing
    statuses, but /bearing/current tells us the current head of each queue.
    The actual status is in bearings.json — we expose it via a lightweight
    call to the queue endpoint and infer: if the bearing is still the current
    head and is NOT in our decisions dict, it hasn't been actioned yet.

    For robustness we also accept status from a dedicated endpoint if added.
    """
    if bearing_name in st.session_state["decisions"]:
        # Already known — no API call needed
        return

    # Try the dedicated bearing info endpoint
    info = _get(f"/bearing/info/{bearing_name}")
    if info and info.get("status") in ("confirmed", "denied"):
        st.session_state["decisions"][bearing_name] = info["status"]


def _parse_current_bearings(info: dict) -> dict:
    """
    Parse /bearing/current response into {group: bearing_name} dict.
    Handles both response shapes:
      New (multi-group): {"group_1": "Bearing1_2", "group_2": "Bearing2_2", ...}
      Legacy (single):   {"current_bearing": {"name": "Bearing1_2"}, ...}
    """
    result = {}

    for key, value in info.items():
        if key.startswith("group_") and value:
            group_id = key.replace("group_", "")
            result[group_id] = value

    if not result:
        bearing_obj = info.get("current_bearing") or info
        if isinstance(bearing_obj, dict):
            name = bearing_obj.get("name")
        else:
            name = bearing_obj if isinstance(bearing_obj, str) else None
        if name:
            result["1"] = name

    return result


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("Fault Review")
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
    st.success("All bearings in the queue have been processed.")
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
    selected_group = list(active_bearings.keys())[0]
    current_name   = active_bearings[selected_group]
else:
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

# ── Restore decision state from bearings.json via API ────────────────────────
# This is the key fix: on every page load we check whether this bearing has
# already been actioned (confirmed/denied) — even if the tech navigated away.
# We do this by fetching the bearing's actual status from the backend.
# The status is written to bearings.json by the confirm/deny endpoints.
bearing_info    = _get(f"/bearing/info/{current_name}") if True else {}
actual_status   = bearing_info.get("status", "") if bearing_info else ""

# If the backend says confirmed or denied but we don't have it in session,
# restore it so the Continue button appears without re-confirming.
if actual_status in ("confirmed", "denied"):
    if current_name not in st.session_state["decisions"]:
        st.session_state["decisions"][current_name] = actual_status

# Get the current decision for this bearing
decision = st.session_state["decisions"].get(current_name)

# ── Step 1: Technician name ───────────────────────────────────────────────────
st.markdown("#### Step 1 — Enter Your Name")
worker_name = st.text_input(
    "Maintenance Technician Name",
    placeholder="e.g. John Smith",
    label_visibility="collapsed",
)
name_ok = bool(worker_name.strip())

# ── Step 2: Confirm or Deny ───────────────────────────────────────────────────
if not decision:
    st.markdown("#### Step 2 — Confirm or Deny the Fault")

    if not name_ok:
        st.info("Enter your name above to enable the fault buttons.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "Confirm Fault",
            use_container_width=True,
            type="primary",
            disabled=not name_ok,
        ):
            with st.spinner("Confirming fault..."):
                result = _confirm_fault(current_name, worker_name.strip())
            if "error" not in result:
                st.session_state["decisions"][current_name] = "confirmed"
                st.rerun()
            else:
                st.error(f"Failed to confirm fault: {result.get('error')}")

    with col2:
        if st.button(
            "Deny Fault",
            use_container_width=True,
            disabled=not name_ok,
        ):
            with st.spinner("Denying fault..."):
                result = _deny_fault(current_name, worker_name.strip())
            if "error" not in result:
                st.session_state["decisions"][current_name] = "denied"
                st.rerun()
            else:
                st.error(f"Failed to deny fault: {result.get('error')}")

# ── Step 3: Continue ──────────────────────────────────────────────────────────
if decision:
    label_str = "confirmed" if decision == "confirmed" else "denied"
    st.markdown("---")
    st.success(
        f"**{current_name}** has been **{label_str}**."
    )

    if decision == "confirmed":
        st.info(
            f"Group {selected_group} retraining has started automatically in the background. "
            f"Other groups continue serving uninterrupted."
        )

    st.markdown("#### Step 3 — Continue")

    if not name_ok:
        st.info("Enter your name above to enable the Continue button.")

    if st.button(
        "▶ Continue → Next Bearing",
        use_container_width=True,
        type="primary",
        disabled=not name_ok,
    ):
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
            # Clear decision for this bearing — it's been actioned and advanced
            st.session_state["decisions"].pop(current_name, None)
            st.rerun()
        else:
            next_b    = result.get("next_bearing") or {}
            next_name = next_b.get("name") if isinstance(next_b, dict) else str(next_b)
            run_id    = result.get("run_id", "")
            st.success(
                f"▶ Now monitoring **{next_name or '?'}**. "
                + (f"Run `{run_id}` started." if run_id else "")
            )
            st.session_state["decisions"].pop(current_name, None)
            st.rerun()