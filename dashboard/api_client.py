"""
dashboard/api_client.py
"""

import requests
import streamlit as st
from typing import Any, Dict, Optional, List


def _base() -> str:
    return st.session_state.get("api_url", "http://localhost:8000")


def _get(path: str, params: Optional[Dict] = None, timeout: int = 30) -> Any:
    try:
        r = requests.get(f"{_base()}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        # API busy — fall back to direct MongoDB read
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot reach backend at `{_base()}`.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None


def _post(path: str, payload: Dict, timeout: int = 15) -> Any:
    try:
        r = requests.post(f"{_base()}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot reach backend at `{_base()}`.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None


# ── Direct MongoDB access (used as fallback when API is busy) ─────────────────

MONGO_URI = "mongodb://localhost:27017"
DB_NAME   = "phm_mlops"

def _mongo_get_latest(bearing_name: str, n: int = 100) -> Optional[List[Dict]]:
    """Read directly from MongoDB — bypasses the API entirely."""
    try:
        from pymongo import MongoClient, DESCENDING
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        col    = client[DB_NAME]["serving_history"]
        cursor = (
            col.find({"bearing_name": bearing_name})
               .sort("burst_idx", DESCENDING)
               .limit(n)
        )
        docs = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            docs.append(doc)
        client.close()
        return docs if docs else None
    except Exception:
        return None


def _mongo_get_run_summary(run_id: str) -> Optional[Dict]:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        col    = client[DB_NAME]["serving_history"]
        total    = col.count_documents({"run_id": run_id})
        ok_count = col.count_documents({"run_id": run_id, "pipeline_ok": True})
        alerts   = col.count_documents({"run_id": run_id, "pm.alert": True})
        client.close()
        return {
            "run_id":       run_id,
            "total_bursts": total,
            "ok_count":     ok_count,
            "error_count":  total - ok_count,
            "total_alerts": alerts,
        }
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def trigger_workflow(workflow_name: str = "rul_prediction",
                     config_overrides: Optional[Dict] = None) -> Optional[Dict]:
    return _post("/workflow/trigger", {
        "workflow_name":    workflow_name,
        "config_overrides": config_overrides or {},
        "priority":         "normal",
    })


def get_workflow_status(run_id: str) -> Optional[Dict]:
    return _get(f"/workflow/{run_id}/status")


def run_bearing_pipeline(bearing_name: str, realtime: bool = False,
                          max_bursts: Optional[int] = None) -> Optional[Dict]:
    payload = {"bearing_name": bearing_name, "realtime": realtime}
    if max_bursts:
        payload["max_bursts"] = max_bursts
    return _post("/serve/pipeline/bearing", payload)


def get_serving_history(bearing_name: str, limit: int = 200) -> Optional[Any]:
    """Try API first, fall back to direct MongoDB read if API is busy/timing out."""
    result = _get("/serving-history",
                  params={"bearing_name": bearing_name, "limit": limit},
                  timeout=10)   # short timeout — fail fast to trigger fallback
    if result is None:
        # API is busy processing bursts — read MongoDB directly
        result = _mongo_get_latest(bearing_name, n=limit)
        if result:
            st.caption("⚡ Live data — reading MongoDB directly (API busy)")
    return result


def get_run_summary(run_id: str) -> Optional[Dict]:
    result = _get(f"/serving-history/run/{run_id}/summary", timeout=10)
    if result is None:
        result = _mongo_get_run_summary(run_id)
    return result


def get_latest_bearing_records(bearing_name: str, n: int = 20) -> Optional[Any]:
    result = _get(f"/serving-history/bearing/{bearing_name}/latest",
                  params={"n": n}, timeout=10)
    if result is None:
        result = _mongo_get_latest(bearing_name, n=n)
    return result


def get_deployed_model() -> Optional[Dict]:
    return _get("/model/deployed", timeout=10)


def get_workflow_registry() -> Optional[Any]:
    return _get("/workflow/registry/list", timeout=10)