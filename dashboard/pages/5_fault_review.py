"""
dashboard/pages/5_✅_Fault_Review.py
════════════════════════════════════════════════════════════════════════════════
Maintenance Worker Interface — Confirm or Deny faults detected by the pipeline.
Step 8 in the diagram: MaintWorker → Dashboard (Confirm/Deny Faults).

Confirmed faults feed back as labelled data (step 12: Dashboard → FeatStoreMirror).
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api_client import get_serving_history

# ── Local JSON store for confirmed/denied faults ───────────────────────────────
FAULT_STORE = os.path.join(os.path.dirname(__file__), "..", "fault_labels.json")


def _load_fault_labels():
    if os.path.exists(FAULT_STORE):
        with open(FAULT_STORE) as f:
            return json.load(f)
    return {}


def _save_fault_labels(labels: dict):
    os.makedirs(os.path.dirname(FAULT_STORE), exist_ok=True)
    with open(FAULT_STORE, "w") as f:
        json.dump(labels, f, indent=2)


st.markdown("# ✅ Fault Review")
st.markdown("Review pipeline alerts and confirm or deny faults as a maintenance worker.")
st.markdown("---")

st.info("""
**Workflow:** When the Serving Pipeline raises an alert, it appears here for review.
- **Confirm** → labels this burst as a genuine fault (feeds step 12: new labelled data → Feature Store Mirror)
- **Deny**    → marks as false positive (preserves data, notes the non-fault)
""")

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([3, 2, 2])
with col1:
    bearing = st.selectbox("Bearing", ["Bearing1_5", "Bearing1_4", "Bearing1_6",
                                        "Bearing1_7", "Bearing2_3"])
with col2:
    show_only_alerts = st.checkbox("Show alerts only", value=True)
with col3:
    n_records = st.slider("Max records", 20, 500, 100)

# ── Fetch & filter ─────────────────────────────────────────────────────────────
records = get_serving_history(bearing, limit=n_records)

if not records:
    st.warning(f"No serving history for **{bearing}**. Run the pipeline first.")
    st.stop()

fault_labels = _load_fault_labels()

rows = []
for r in records:
    pm  = r.get("pm", {}) or {}
    inf = r.get("inference", {}) or {}
    burst = r.get("burst_idx")
    key   = f"{bearing}:{burst}"
    rows.append({
        "key":         key,
        "burst_idx":   burst,
        "rul_min":     r.get("rul_min") or inf.get("rul_min"),
        "pm_status":   pm.get("status", "—"),
        "alert":       pm.get("alert", False),
        "action":      pm.get("recommended_action", "—"),
        "data_quality":r.get("data_quality", "clean"),
        "label":       fault_labels.get(key, {}).get("label", "Pending"),
        "note":        fault_labels.get(key, {}).get("note", ""),
        "labelled_by": fault_labels.get(key, {}).get("labelled_by", ""),
        "labelled_at": fault_labels.get(key, {}).get("labelled_at", ""),
    })

df = pd.DataFrame(rows).sort_values("burst_idx", ascending=False)
if show_only_alerts:
    df = df[df["alert"] == True]

# ── Summary ───────────────────────────────────────────────────────────────────
total_alerts   = (pd.DataFrame(rows)["alert"] == True).sum()
confirmed      = sum(1 for v in fault_labels.values() if v.get("label") == "Confirmed")
denied         = sum(1 for v in fault_labels.values() if v.get("label") == "Denied")
pending        = total_alerts - confirmed - denied

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Alerts",  total_alerts)
c2.metric("✅ Confirmed",   confirmed)
c3.metric("❌ Denied",      denied)
c4.metric("⏳ Pending",     max(0, pending))

st.markdown("---")

# ── Review table ──────────────────────────────────────────────────────────────
if df.empty:
    st.success("✅ No alerts to review in the current window.")
    st.stop()

st.markdown(f"### Alerts for **{bearing}** ({len(df)} shown)")

worker_name = st.text_input("Your name (Maintenance Worker)", value="Worker",
                             placeholder="e.g. Jan van der Berg")

for _, row in df.iterrows():
    key        = row["key"]
    current_lbl= row["label"]
    burst      = row["burst_idx"]
    rul        = f"{row['rul_min']:.1f} min" if pd.notna(row.get("rul_min")) else "—"

    label_color = {
        "Confirmed": "#4ade80",
        "Denied":    "#f87171",
        "Pending":   "#fbbf24",
    }.get(current_lbl, "#64748b")

    with st.expander(
        f"Burst #{burst} · RUL: {rul} · Status: {row['pm_status'].upper()} · "
        f"Label: {current_lbl}",
        expanded=(current_lbl == "Pending")
    ):
        col_info, col_action = st.columns([3, 2])

        with col_info:
            st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;
                        color:#94a3b8; line-height:1.8;">
                <b style="color:#e2e8f0;">Burst:</b> {burst}<br>
                <b style="color:#e2e8f0;">RUL:</b> {rul}<br>
                <b style="color:#e2e8f0;">PM Status:</b> {row['pm_status']}<br>
                <b style="color:#e2e8f0;">Recommended Action:</b> {row['action']}<br>
                <b style="color:#e2e8f0;">Data Quality:</b> {row['data_quality']}<br>
                <b style="color:#e2e8f0;">Current Label:</b>
                    <span style="color:{label_color}; font-weight:600;">{current_lbl}</span>
            </div>
            """, unsafe_allow_html=True)

            if row["labelled_by"]:
                st.caption(f"Labelled by {row['labelled_by']} at {row['labelled_at']}")

        with col_action:
            note = st.text_area(f"Note (burst {burst})", value=row["note"],
                                placeholder="Describe the fault or reason for denial...",
                                key=f"note_{key}", height=80)

            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button("✅ Confirm", key=f"confirm_{key}"):
                    fault_labels[key] = {
                        "label":       "Confirmed",
                        "note":        note,
                        "labelled_by": worker_name,
                        "labelled_at": datetime.now().isoformat(),
                        "bearing":     bearing,
                        "burst_idx":   int(burst),
                    }
                    _save_fault_labels(fault_labels)
                    st.success("Confirmed!")
                    st.rerun()

            with btn_col2:
                if st.button("❌ Deny", key=f"deny_{key}"):
                    fault_labels[key] = {
                        "label":       "Denied",
                        "note":        note,
                        "labelled_by": worker_name,
                        "labelled_at": datetime.now().isoformat(),
                        "bearing":     bearing,
                        "burst_idx":   int(burst),
                    }
                    _save_fault_labels(fault_labels)
                    st.warning("Denied.")
                    st.rerun()

            with btn_col3:
                if st.button("🔄 Reset", key=f"reset_{key}"):
                    fault_labels.pop(key, None)
                    _save_fault_labels(fault_labels)
                    st.rerun()

# ── Export labelled data ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Export Labelled Fault Data")
st.markdown("Export confirmed/denied fault labels as CSV for downstream retraining (Step 12 → Feature Store Mirror).")

if fault_labels:
    export_df = pd.DataFrame(list(fault_labels.values()))
    export_df = export_df.sort_values("burst_idx") if "burst_idx" in export_df.columns else export_df
    st.dataframe(export_df, use_container_width=True, hide_index=True)

    csv = export_df.to_csv(index=False)
    st.download_button(
        "⬇ Download Fault Labels CSV",
        data=csv,
        file_name=f"fault_labels_{bearing}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
else:
    st.info("No labels saved yet. Review and confirm/deny alerts above.")