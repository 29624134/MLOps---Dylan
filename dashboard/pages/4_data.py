"""
dashboard/pages/4_🔍_Data_Quality.py
════════════════════════════════════════════════════════════════════════════════
Data-centric monitoring: drift flags, anomaly types, per-feature stats.
Covers MLOps Monitoring stage output (§3) and Feature Engineering labels (§4).
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import get_serving_history

st.markdown("# 🔍 Data Quality Monitor")
st.markdown("Distribution drift, anomaly detection, and per-feature statistics.")
st.markdown("---")

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])
with col1:
    bearing = st.selectbox("Bearing", ["Bearing1_5", "Bearing1_4", "Bearing1_6",
                                        "Bearing1_7", "Bearing2_3"])
with col2:
    n_records = st.slider("Records", 50, 1000, 200, step=50)

records = get_serving_history(bearing, limit=n_records)

if not records:
    st.warning(f"No data for **{bearing}**. Run the serving pipeline first.")
    st.stop()

# ── Parse data ────────────────────────────────────────────────────────────────
rows = []
for r in records:
    mn = r.get("monitoring", {}) or {}
    fe = r.get("fe", {})         or {}
    ql = fe.get("quality_labels", {}) or {}

    rows.append({
        "burst_idx":      r.get("burst_idx"),
        "data_quality":   r.get("data_quality", "clean"),
        "drift_detected": mn.get("drift_detected", False),
        "anomaly_flag":   mn.get("anomaly_flag", False),
        "drift_features": mn.get("drift_features", []),
        "baseline_ready": mn.get("baseline_ready", False),
        "outlier":        ql.get("outlier", False),
        "missing":        ql.get("missing", False),
        "anomaly_type":   ql.get("anomaly_type", "none"),
        "fe_ready":       fe.get("ready", False),
    })

df = pd.DataFrame(rows).sort_values("burst_idx")

# ── Summary metrics ───────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
total   = len(df)
drifted = df["drift_detected"].sum()
anomaly = df["anomaly_flag"].sum()
outlier = df["outlier"].sum()

m1.metric("Total Bursts",   total)
m2.metric("Drift Detected", f"{drifted} ({100*drifted/total:.1f}%)" if total else "—",
          delta=None if drifted == 0 else f"{drifted} flagged")
m3.metric("Anomaly Flags",  f"{anomaly}")
m4.metric("Outlier Bursts", f"{outlier}")

st.markdown("---")

# ── Drift timeline ────────────────────────────────────────────────────────────
st.markdown("#### Drift Detection Timeline")

fig = go.Figure()

# Background: all bursts
fig.add_trace(go.Scatter(
    x=df["burst_idx"],
    y=[0.2] * len(df),
    mode="markers",
    marker={"color": "#1e2530", "size": 6, "symbol": "line-ns"},
    name="No drift",
    showlegend=True,
))

# Drift bursts
drift_df = df[df["drift_detected"]]
if not drift_df.empty:
    fig.add_trace(go.Scatter(
        x=drift_df["burst_idx"],
        y=[0.5] * len(drift_df),
        mode="markers",
        marker={"color": "#c084fc", "size": 10, "symbol": "diamond"},
        name="Drift detected",
    ))

# Anomaly bursts
anom_df = df[df["anomaly_flag"]]
if not anom_df.empty:
    fig.add_trace(go.Scatter(
        x=anom_df["burst_idx"],
        y=[0.8] * len(anom_df),
        mode="markers",
        marker={"color": "#f87171", "size": 10, "symbol": "x"},
        name="Anomaly flag",
    ))

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    height=200,
    margin=dict(l=50, r=20, t=10, b=40),
    font={"color": "#e2e8f0", "family": "JetBrains Mono"},
    xaxis={"title": "Burst Index", "gridcolor": "#1e2530", "color": "#64748b"},
    yaxis={"visible": False},
    legend={"bgcolor": "#151820", "bordercolor": "#1e2530", "borderwidth": 1},
)
st.plotly_chart(fig, use_container_width=True)

# ── Anomaly type breakdown ─────────────────────────────────────────────────────
st.markdown("---")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Anomaly Type Breakdown")
    atype_counts = df["anomaly_type"].value_counts().reset_index()
    atype_counts.columns = ["Anomaly Type", "Count"]

    colors = {
        "none":    "#4ade80",
        "spike":   "#f87171",
        "dropout": "#fbbf24",
        "null":    "#c084fc",
    }
    fig_pie = go.Figure(go.Pie(
        labels=atype_counts["Anomaly Type"],
        values=atype_counts["Count"],
        marker={"colors": [colors.get(t, "#64748b") for t in atype_counts["Anomaly Type"]],
                "line": {"color": "#0d0f14", "width": 2}},
        textfont={"family": "JetBrains Mono", "size": 11},
        hole=0.4,
    ))
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        font={"color": "#e2e8f0", "family": "JetBrains Mono"},
        legend={"bgcolor": "rgba(0,0,0,0)", "font": {"size": 11}},
        showlegend=True,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.markdown("#### Data Quality Label Distribution")
    dq_counts = df["data_quality"].value_counts().reset_index()
    dq_counts.columns = ["Quality Label", "Count"]

    dq_colors = {
        "clean":            "#4ade80",
        "outlier_detected": "#fbbf24",
        "missing_detected": "#c084fc",
        "spike_detected":   "#f87171",
        "dropout_detected": "#fb923c",
        "no_model":         "#64748b",
        "prediction_failed":"#f87171",
    }
    fig_bar = go.Figure(go.Bar(
        x=dq_counts["Quality Label"],
        y=dq_counts["Count"],
        marker_color=[dq_colors.get(q, "#64748b") for q in dq_counts["Quality Label"]],
        text=dq_counts["Count"],
        textposition="outside",
        textfont={"family": "JetBrains Mono", "size": 10},
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=20, r=20, t=20, b=60),
        font={"color": "#e2e8f0", "family": "JetBrains Mono"},
        xaxis={"tickangle": -30, "color": "#64748b", "gridcolor": "#1e2530"},
        yaxis={"gridcolor": "#1e2530", "color": "#64748b"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Most drifted features ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Most Frequently Drifted Features")

all_drift_features = []
for feat_list in df["drift_features"]:
    if isinstance(feat_list, list):
        all_drift_features.extend(feat_list)

if all_drift_features:
    from collections import Counter
    feat_counts = Counter(all_drift_features)
    feat_df = pd.DataFrame(feat_counts.items(), columns=["Feature", "Drift Count"])
    feat_df = feat_df.sort_values("Drift Count", ascending=False).head(20)

    fig_feat = go.Figure(go.Bar(
        y=feat_df["Feature"],
        x=feat_df["Drift Count"],
        orientation="h",
        marker_color="#c084fc",
        opacity=0.85,
    ))
    fig_feat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=150, r=20, t=10, b=40),
        font={"color": "#e2e8f0", "family": "JetBrains Mono", "size": 10},
        xaxis={"title": "Drift Event Count", "gridcolor": "#1e2530", "color": "#64748b"},
        yaxis={"color": "#64748b", "tickfont": {"size": 9}},
    )
    st.plotly_chart(fig_feat, use_container_width=True)
else:
    st.info("No drift feature details available yet — baseline may still be warming up.")

# ── Raw monitoring records table ───────────────────────────────────────────────
st.markdown("---")
with st.expander("🔎 Raw Quality Records (last 50)", expanded=False):
    display_cols = ["burst_idx", "data_quality", "anomaly_type",
                    "outlier", "missing", "drift_detected", "anomaly_flag"]
    st.dataframe(df[display_cols].tail(50).style.applymap(
        lambda v: "color: #f87171; font-weight:600;" if v is True else
                  "color: #4ade80;" if v is False else "",
        subset=["drift_detected", "anomaly_flag", "outlier", "missing"]
    ), use_container_width=True, hide_index=True)