"""
rul_app.py
════════════════════════════════════════════════════════════════════════════════
Standalone RUL Monitor — run with:
    streamlit run rul_app.py

Reads from the FastAPI backend (default: http://localhost:8000).
Plots the Remaining Useful Life time-series per bearing, colour-coded by
PM status (healthy / warning / critical).

Data structure expected from /serving-history:
    Each record has:
        burst_idx         : int
        bearing_name      : str
        pm_status         : str   (top-level shortcut written by ServingPipeline)
        inference         : dict  { rul_s, rul_min, ... }
        pm                : dict  { status, rul_s, alert, ... }
        monitoring        : dict  { drift_detected, ... }
"""

import time
import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RUL Monitor",
    page_icon="📊",
    layout="wide",
)

API_BASE = "http://localhost:8000"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(path: str, params: dict = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot reach backend at `{API_BASE}`. Is the API running?")
        return None
    except Exception as e:
        st.error(f"❌ API error: {e}")
        return None


def _fetch_records(bearing_name: str, limit: int) -> list:
    """Fetch serving history records for a bearing."""
    data = _get("/serving-history", params={"bearing_name": bearing_name, "limit": limit})
    return data if isinstance(data, list) else []


def _extract_rul(record: dict):
    """
    Extract RUL values from a record regardless of nesting depth.
    ServingHistory stores RUL nested under 'inference', but also writes
    top-level shortcuts. Try all known locations.
    """
    inf = record.get("inference") or {}
    pm  = record.get("pm")        or {}

    rul_s = (
        record.get("rul_s")
        or inf.get("rul_s")
        or pm.get("rul_s")
        or inf.get("predicted_rul_s")
    )
    if rul_s is None:
        return None, None

    rul_min = (
        record.get("rul_min")
        or inf.get("rul_min")
        or pm.get("rul_min")
        or (rul_s / 60.0)
    )
    return float(rul_s), float(rul_min)


def _extract_status(record: dict) -> str:
    """Extract PM status from wherever it lives in the record."""
    pm = record.get("pm") or {}
    return (
        record.get("pm_status")
        or pm.get("status")
        or pm.get("pm_status")
        or "unknown"
    ).lower()


STATUS_COLOURS = {
    "healthy":  "#4ade80",   # green
    "warning":  "#facc15",   # yellow
    "critical": "#f87171",   # red
    "unknown":  "#94a3b8",   # grey
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📊 RUL Monitor")
st.markdown("Remaining Useful Life predictions per bearing.")
st.markdown("---")

# ── Controls ──────────────────────────────────────────────────────────────────
col_b, col_n, col_r = st.columns([3, 2, 2])

with col_b:
    bearing = st.selectbox(
        "Bearing",
        ["Bearing1_5", "Bearing1_4", "Bearing1_6", "Bearing1_7", "Bearing2_3"],
    )
with col_n:
    n_records = st.slider("Records to show", min_value=50, max_value=2500, value=500, step=50)
with col_r:
    st.markdown("<br>", unsafe_allow_html=True)
    auto_refresh = st.checkbox("Auto-refresh (10 s)", value=False)

# ── Fetch ─────────────────────────────────────────────────────────────────────
records = _fetch_records(bearing, n_records)

if not records:
    st.warning(f"No serving history found for **{bearing}**. Run the pipeline first.")
    st.stop()

# ── Build DataFrame ───────────────────────────────────────────────────────────
rows = []
for r in records:
    rul_s, rul_min = _extract_rul(r)
    if rul_s is None:
        continue   # warmup burst — not ready yet
    rows.append({
        "burst_idx": r.get("burst_idx", 0),
        "rul_s":     rul_s,
        "rul_min":   rul_min,
        "rul_h":     rul_s / 3600.0,
        "status":    _extract_status(r),
        "drift":     (r.get("monitoring") or {}).get("drift_detected", False),
        "alert":     (r.get("pm") or {}).get("alert", False),
    })

if not rows:
    st.warning(
        f"Records were found for **{bearing}** but none contain RUL data. "
        "The pipeline warmup window (first 40 bursts) produces no predictions."
    )
    st.stop()

df = pd.DataFrame(rows).sort_values("burst_idx").reset_index(drop=True)

# ── Summary metrics ───────────────────────────────────────────────────────────
latest      = df.iloc[-1]
latest_rul  = latest["rul_min"]
latest_stat = latest["status"]
n_alerts    = int(df["alert"].sum())
n_drift     = int(df["drift"].sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest RUL", f"{latest_rul:.1f} min")
m2.metric(
    "PM Status",
    latest_stat.capitalize(),
    delta=None,
)
m3.metric("Alert Bursts", n_alerts)
m4.metric("Drift Events", n_drift)

st.markdown("---")

# ── RUL Time-Series Chart ─────────────────────────────────────────────────────
# Split df into status groups so each group gets its own colour trace.
fig = go.Figure()

for status, colour in STATUS_COLOURS.items():
    subset = df[df["status"] == status]
    if subset.empty:
        continue
    fig.add_trace(go.Scatter(
        x=subset["burst_idx"],
        y=subset["rul_min"],
        mode="lines+markers",
        name=status.capitalize(),
        line=dict(color=colour, width=1.5),
        marker=dict(size=3, color=colour),
        hovertemplate=(
            "<b>Burst %{x}</b><br>"
            "RUL: %{y:.1f} min<br>"
            f"Status: {status}<extra></extra>"
        ),
    ))

# Threshold lines
critical_min = 3600 / 60    # 1 h
warning_min  = 14400 / 60   # 4 h

fig.add_hline(
    y=critical_min,
    line_dash="dash",
    line_color="#f87171",
    annotation_text="Critical threshold (1 h)",
    annotation_position="top left",
)
fig.add_hline(
    y=warning_min,
    line_dash="dot",
    line_color="#facc15",
    annotation_text="Warning threshold (4 h)",
    annotation_position="top left",
)

fig.update_layout(
    title=f"RUL over time — {bearing}",
    xaxis_title="Burst index",
    yaxis_title="RUL (minutes)",
    height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=60, r=20, t=60, b=60),
    hovermode="x unified",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")

st.plotly_chart(fig, use_container_width=True)

# ── Alert log ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### PM Alert Bursts")

alerts_df = df[df["alert"] == True][["burst_idx", "rul_min", "rul_h", "status"]]
if alerts_df.empty:
    st.info("No PM alerts recorded for this bearing.")
else:
    st.dataframe(
        alerts_df.rename(columns={
            "burst_idx": "Burst",
            "rul_min":   "RUL (min)",
            "rul_h":     "RUL (h)",
            "status":    "Status",
        }),
        use_container_width=True,
        hide_index=True,
    )

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(10)
    st.rerun()