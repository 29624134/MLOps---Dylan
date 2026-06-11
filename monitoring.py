"""
monitor.py
═══════════════════════════════════════════════════════════════════════════════
Live MLOps System Monitor — run this in a separate PyCharm terminal while
scada_simulator.py and run_serving.py are running.

Three modes
───────────
Live mode (default):
    Polls MongoDB serving_history every N seconds and prints a refreshing
    dashboard showing latency (per-stage + e2e), CPU, memory, throughput,
    health, RUL trend.

Summary mode (--summary):
    One-shot full system report. Includes:
      • all collections, all bearings, all runs, model registry state
      • per-stage latency table (mean / median / p95 / p99)
      • end-to-end latency percentiles (sent_at → telemetry.record)
      • ingestion-lag distribution (queue wait between SCADA and serving)

Thesis-export mode (--export-csv PATH):
    Writes one row per burst from serving_history to a CSV file, including
    every latency field (pipeline, e2e, ingestion lag, all per-stage timings)
    plus the operational columns. This is the file you import into pandas /
    Excel / your thesis appendix.

Usage
─────
    python monitor.py
    python monitor.py --interval 3
    python monitor.py --run_id serve_xyz
    python monitor.py --bearing Bearing1_5
    python monitor.py --summary
    python monitor.py --summary --run_id serve_xyz
    python monitor.py --export-csv thesis_latency.csv
    python monitor.py --export-csv thesis_latency.csv --run_id serve_xyz

Stop live mode with Ctrl+C.
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from typing import Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MONGO_URI  = "mongodb://localhost:27017"
DEFAULT_DB_NAME    = "phm_mlops"
DEFAULT_INTERVAL_S = 5.0

# Stage keys we expect in stage_timings_ms. Kept here so we have a stable
# order in tables and CSVs even if the pipeline adds/removes fields.
_STAGE_KEYS = (
    "fe_ms",
    "inference_ms",
    "pm_ms",
    "monitoring_ms",
    "serving_history_ms",
    "export_ms",
    "audit_ms",
    "pipeline_total_ms",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _percentile(sorted_vals: Sequence[float], frac: float) -> Optional[float]:
    """
    Safe percentile from an already-sorted list. Clamps the index so it
    never exceeds len-1 (fixes the off-by-one in the original monitor).
    Returns None when the list is empty.
    """
    if not sorted_vals:
        return None
    n   = len(sorted_vals)
    idx = min(int(n * frac), n - 1)
    return float(sorted_vals[idx])


def _stats(values):
    """Return (n, mean, min, p50, p95, p99, max) for a list of numbers."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    return {
        "n":    n,
        "mean": sum(s) / n,
        "min":  s[0],
        "p50":  _percentile(s, 0.50),
        "p95":  _percentile(s, 0.95),
        "p99":  _percentile(s, 0.99),
        "max":  s[-1],
    }


def _bar(value: float, max_value: float, width: int = 30,
         warn: float = 0.6, crit: float = 0.85) -> str:
    if max_value <= 0:
        return "[" + "-" * width + "]"
    ratio  = min(value / max_value, 1.0)
    filled = int(ratio * width)
    bar    = "█" * filled + "░" * (width - filled)
    pct    = ratio * 100
    tag    = "CRIT" if ratio >= crit else ("WARN" if ratio >= warn else "OK  ")
    return f"[{bar}] {pct:5.1f}%  {tag}"


def _fmt_ms(val) -> str:
    if val is None:
        return "    —    "
    return f"{val:>8.2f} ms"


def _fmt_rul(val) -> str:
    if val is None:
        return "   —   "
    h = int(val // 60)
    m = int(val % 60)
    return f"{h}h {m:02d}m"


def _status_icon(status: str) -> str:
    return {"healthy": "✅", "warning": "⚠️ ", "critical": "🔴"}.get(status, "❓")


def _div(W=65):
    print("─" * W)


def _hdr(title, W=65):
    print(f"\n  ── {title} {'─' * max(0, W - len(title) - 6)}")


# ─────────────────────────────────────────────────────────────────────────────
# Monitor
# ─────────────────────────────────────────────────────────────────────────────

class LiveMonitor:

    def __init__(
        self,
        mongo_uri:    str,
        db_name:      str,
        interval:     float,
        run_id:       Optional[str],
        bearing_name: Optional[str],
    ):
        from pymongo import MongoClient
        from utils.db_collections import (
            COL_SERVING_HISTORY, COL_RUL_PREDICTIONS,
            COL_FACTORY_FEATURES, COL_FEATURE_STORE,
            COL_FEATURE_STORE_MIRRORED, COL_MODEL_REGISTRY,
            COL_WORKFLOW_REGISTRY, COL_PREPROD_RUNS,
        )

        self._interval = interval
        self._run_id   = run_id
        self._bearing  = bearing_name
        self._db_name  = db_name

        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        try:
            client.admin.command("ping")
        except Exception as e:
            print(f"\n✗  Cannot connect to MongoDB at {mongo_uri}")
            print(f"   Error: {e}")
            print("   Is MongoDB running? Is the API started?")
            sys.exit(1)

        db = client[db_name]
        self._db  = db
        self._sh  = db[COL_SERVING_HISTORY]
        self._rul = db[COL_RUL_PREDICTIONS]
        self._ff  = db[COL_FACTORY_FEATURES]
        self._fs  = db[COL_FEATURE_STORE]
        self._fsm = db[COL_FEATURE_STORE_MIRRORED]
        self._mr  = db[COL_MODEL_REGISTRY]
        self._wr  = db[COL_WORKFLOW_REGISTRY]
        self._pr  = db[COL_PREPROD_RUNS]

        self._col_names = {
            COL_SERVING_HISTORY:        "serving_history",
            COL_RUL_PREDICTIONS:        "RUL_predictions",
            COL_FACTORY_FEATURES:       "factory_features",
            COL_FEATURE_STORE:          "feature_store",
            COL_FEATURE_STORE_MIRRORED: "feature_store_mirrored",
            COL_MODEL_REGISTRY:         "model_registry",
            COL_WORKFLOW_REGISTRY:      "workflow_registry",
            COL_PREPROD_RUNS:           "preprod_runs",
        }

        print(f"✓  Connected to MongoDB → {db_name}")
        time.sleep(1)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _query(self, collection, extra_filter=None, limit=200):
        q = {}
        if self._run_id:
            q["run_id"] = self._run_id
        if self._bearing:
            q["bearing_name"] = self._bearing
        if extra_filter:
            q.update(extra_filter)
        return list(collection.find(q, {"_id": 0}).sort("timestamp", -1).limit(limit))

    def _latest_run_id(self) -> Optional[str]:
        doc = self._sh.find_one({}, sort=[("timestamp", -1)])
        return doc.get("run_id") if doc else None

    def _pipeline_ms_of(self, doc) -> Optional[float]:
        """Prefer the new pipeline_ms field; fall back to legacy latency_ms."""
        val = doc.get("pipeline_ms")
        if val is None:
            val = doc.get("latency_ms")
        return val

    # ── Live dashboard ────────────────────────────────────────────────────────

    def render(self):
        run_id = self._run_id or self._latest_run_id()
        if not run_id:
            print("  Waiting for serving pipeline to start...\n")
            return

        docs = self._query(self._sh, limit=200)
        if not docs:
            print(f"  No records yet for run_id={run_id}\n")
            print("  Waiting for bursts from scada_simulator.py...")
            return

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        W = 72

        # Pull values
        pipe_lats   = [self._pipeline_ms_of(d) for d in docs]
        pipe_lats   = [v for v in pipe_lats if v is not None]
        e2e_lats    = [d["e2e_ms"]           for d in docs if d.get("e2e_ms")           is not None]
        ingest_lags = [d["ingestion_lag_ms"] for d in docs if d.get("ingestion_lag_ms") is not None]
        cpu_vals    = [d["cpu_percent"]      for d in docs if d.get("cpu_percent")      is not None]
        mem_vals    = [d["memory_mb"]        for d in docs if d.get("memory_mb")        is not None]
        rul_mins    = [d["rul_min"]          for d in docs if d.get("rul_min")          is not None]
        statuses    = [d.get("pm_status", "unknown") for d in docs]
        models      = list({d["model_version"] for d in docs if d.get("model_version")})

        drift_count   = sum(1 for d in docs if d.get("drift_detected"))
        anomaly_count = sum(1 for d in docs if d.get("anomaly_flag"))
        ok_count      = sum(1 for d in docs if d.get("pipeline_ok"))
        err_count     = len(docs) - ok_count
        total_bursts  = self._sh.count_documents({"run_id": run_id})

        latest        = docs[0]
        latest_burst  = latest.get("burst_idx", "?")
        latest_bear   = latest.get("bearing_name", "?")
        latest_status = latest.get("pm_status", "unknown")
        latest_rul    = latest.get("rul_min")
        latest_pipe   = self._pipeline_ms_of(latest)
        latest_e2e    = latest.get("e2e_ms")
        latest_ingest = latest.get("ingestion_lag_ms")
        latest_cpu    = latest.get("cpu_percent")
        latest_mem    = latest.get("memory_mb")
        latest_stage  = latest.get("stage_timings_ms") or {}

        pipe_stats = _stats(pipe_lats)
        e2e_stats  = _stats(e2e_lats)
        ing_stats  = _stats(ingest_lags)

        avg_cpu = sum(cpu_vals) / len(cpu_vals) if cpu_vals else None
        max_cpu = max(cpu_vals)                 if cpu_vals else None
        avg_mem = sum(mem_vals) / len(mem_vals) if mem_vals else None

        cutoff = time.time() - 60
        recent = [d for d in docs if d.get("timestamp") and
                  d["timestamp"].timestamp() > cutoff]
        tput_60s = len(recent)

        print("=" * W)
        print(f"  PHM MLOps — Live Monitor              {now_str}")
        print("=" * W)
        print(f"  Run ID   : {run_id}")
        print(f"  Bearing  : {latest_bear}   Burst #{latest_burst}")
        print(f"  Models   : {', '.join(models) if models else '—'}")
        _div(W)

        _hdr("Latest Burst", W)
        print(f"  Status      : {_status_icon(latest_status)}  {latest_status.upper()}")
        print(f"  RUL         : {_fmt_rul(latest_rul)}")
        print(f"  Pipeline    : {_fmt_ms(latest_pipe)}")
        print(f"  Ingestion   : {_fmt_ms(latest_ingest)}   (SCADA → run_serving pickup)")
        print(f"  End-to-end  : {_fmt_ms(latest_e2e)}   (sent_at → telemetry written)")
        if latest_cpu is not None:
            print(f"  CPU         : {_bar(latest_cpu, 100, width=25)}")
        if latest_mem is not None:
            print(f"  Memory      : {latest_mem:.1f} MB")

        # Per-stage breakdown for the latest burst
        if latest_stage:
            _hdr("Latest Burst — Stage Breakdown", W)
            for k in _STAGE_KEYS:
                v = latest_stage.get(k)
                if v is None:
                    continue
                print(f"  {k:<22}: {_fmt_ms(v)}")

        _hdr(f"Pipeline Latency  (last {len(pipe_lats)} bursts)", W)
        if pipe_stats:
            print(f"  mean   : {_fmt_ms(pipe_stats['mean'])}")
            print(f"  min    : {_fmt_ms(pipe_stats['min'])}")
            print(f"  p50    : {_fmt_ms(pipe_stats['p50'])}")
            print(f"  p95    : {_fmt_ms(pipe_stats['p95'])}")
            print(f"  p99    : {_fmt_ms(pipe_stats['p99'])}")
            print(f"  max    : {_fmt_ms(pipe_stats['max'])}")
            print(f"  budget : {_bar(pipe_stats['mean'], 10000, width=25, warn=0.02, crit=0.1)}  (10s burst period)")
        else:
            print("  No pipeline-latency data yet.")

        _hdr(f"End-to-End Latency  (last {len(e2e_lats)} bursts)", W)
        if e2e_stats:
            print(f"  mean   : {_fmt_ms(e2e_stats['mean'])}")
            print(f"  p50    : {_fmt_ms(e2e_stats['p50'])}")
            print(f"  p95    : {_fmt_ms(e2e_stats['p95'])}")
            print(f"  p99    : {_fmt_ms(e2e_stats['p99'])}")
            print(f"  max    : {_fmt_ms(e2e_stats['max'])}")
        else:
            print("  No end-to-end data yet (SCADA simulator may not be writing sent_at).")

        _hdr(f"Ingestion Lag  (last {len(ingest_lags)} bursts)", W)
        if ing_stats:
            print(f"  mean   : {_fmt_ms(ing_stats['mean'])}   (sent_at → run_serving pickup)")
            print(f"  p95    : {_fmt_ms(ing_stats['p95'])}")
            print(f"  max    : {_fmt_ms(ing_stats['max'])}")
        else:
            print("  No ingestion-lag data yet.")

        _hdr("Resource Usage  (run_serving.py process only)", W)
        if avg_cpu is not None:
            print(f"  CPU avg : {_bar(avg_cpu, 100, width=25)}")
            print(f"  CPU max : {_bar(max_cpu, 100, width=25)}")
        else:
            print("  CPU     : — (psutil not installed or no data yet)")
        if avg_mem is not None:
            print(f"  Mem avg : {avg_mem:.1f} MB")

        _hdr("Throughput", W)
        print(f"  Total bursts   : {total_bursts:,}")
        print(f"  Last 60s       : {tput_60s} bursts  (note: capped at last 200 docs)")
        print(f"  Pipeline OK    : {ok_count}   Errors: {err_count}")

        _hdr(f"Health  (last {len(docs)} bursts)", W)
        print(f"  🔴 Critical : {statuses.count('critical')}")
        print(f"  ⚠️  Warning  : {statuses.count('warning')}")
        print(f"  ✅ Healthy  : {statuses.count('healthy')}")
        print(f"  🔀 Drift    : {drift_count}   🔺 Anomaly: {anomaly_count}")

        if len(rul_mins) >= 3:
            window     = rul_mins[:10]
            max_rul    = max(window) if max(window) > 0 else 1.0
            trend_vals = list(reversed(window))
            trend_str  = "  " + "".join(
                "▁▂▃▄▅▆▇█"[min(int((v / max_rul) * 8), 7)] for v in trend_vals
            )
            _hdr(f"RUL Trend  (oldest→latest, last {len(trend_vals)} bursts)", W)
            print(f"  {trend_str}   latest: {_fmt_rul(rul_mins[0])}")

        print("\n" + "=" * W)
        print(f"  Refresh: {self._interval}s | --summary for full report | "
              f"--export-csv to dump | Ctrl+C to stop")

    # ── Summary report ────────────────────────────────────────────────────────

    def _stage_table(self, docs):
        """Build per-stage latency stats across the supplied docs."""
        cols = {k: [] for k in _STAGE_KEYS}
        for d in docs:
            st = d.get("stage_timings_ms") or {}
            for k in _STAGE_KEYS:
                v = st.get(k)
                if v is not None:
                    cols[k].append(v)
        return {k: _stats(vs) for k, vs in cols.items()}

    def summary(self):
        W   = 78
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("\n" + "=" * W)
        print(f"  PHM MLOps — SYSTEM SUMMARY REPORT")
        print(f"  Generated : {now}")
        print(f"  Database  : {self._db_name}")
        if self._run_id:
            print(f"  Run filter: {self._run_id}")
        if self._bearing:
            print(f"  Bearing   : {self._bearing}")
        print("=" * W)

        # ── 1. MongoDB collections ────────────────────────────────────────────
        _hdr("MongoDB Collections", W)
        col_names_in_db = self._db.list_collection_names()
        print(f"  {'Collection':<32} {'Documents':>12}  {'Present'}")
        _div(W)
        for col_key, label in sorted(self._col_names.items(), key=lambda x: x[1]):
            n      = self._db[col_key].count_documents({})
            exists = "✅" if col_key in col_names_in_db else "❌  MISSING — not created yet"
            print(f"  {col_key:<32} {n:>12,}  {exists}")
        extra = [c for c in col_names_in_db
                 if c not in self._col_names and c != "metadata"]
        if extra:
            print(f"\n  Other collections present: {', '.join(sorted(extra))}")

        # ── 2. Model registry ─────────────────────────────────────────────────
        _hdr("Model Registry", W)
        models = list(self._mr.find({}, {"_id": 0}))
        if not models:
            print("  No models registered yet.")
        else:
            print(f"  {'':2} {'Model ID':<14} {'Status':<12} {'Target':<8} "
                  f"{'MAE_s':>10}  {'RMSE_s':>10}  {'CRA':>6}  {'Registered'}")
            _div(W)
            for m in sorted(models, key=lambda x: x.get("registered_at", "")):
                mid    = (m.get("model_id") or "?")[:12]
                status = m.get("status", "?")
                target = m.get("target_feature", "?")
                mets   = m.get("metrics", {})
                mae    = mets.get("mae_s")
                rmse   = mets.get("rmse_s")
                cra    = mets.get("mean_cra")
                mae_s  = f"{mae:>10.0f}" if mae  is not None else "         —"
                rms_s  = f"{rmse:>10.0f}" if rmse is not None else "         —"
                cra_s  = f"{cra:>6.3f}"   if cra  is not None else "     —"
                reg_at = (m.get("registered_at") or "")[:16]
                icon   = "🚀" if status == "deployed" else ("✅" if status == "approved" else "  ")
                print(f"  {icon} {mid:<14} {status:<12} {target:<8} "
                      f"{mae_s}  {rms_s}  {cra_s}  {reg_at}")

        # ── 3. Serving runs ───────────────────────────────────────────────────
        _hdr("Serving Runs  (serving_history)", W)
        run_query = {"run_id": self._run_id} if self._run_id else {}
        run_ids   = self._sh.distinct("run_id", run_query)
        if not run_ids:
            print("  No serving runs recorded yet.")
        else:
            print(f"  {'Run ID':<30} {'Bursts':>7} {'OK':>5} {'Err':>5} "
                  f"{'Avg pipe':>10}  {'Avg e2e':>10}  {'Bearing(s)'}")
            _div(W)
            for rid in sorted(run_ids, reverse=True):
                rdocs    = list(self._sh.find({"run_id": rid}, {"_id": 0}))
                n        = len(rdocs)
                ok       = sum(1 for d in rdocs if d.get("pipeline_ok"))
                err      = n - ok
                pipe     = [self._pipeline_ms_of(d) for d in rdocs]
                pipe     = [v for v in pipe if v is not None]
                e2es     = [d["e2e_ms"] for d in rdocs if d.get("e2e_ms") is not None]
                avg_p    = f"{sum(pipe)/len(pipe):>8.1f}ms" if pipe else "         —"
                avg_e    = f"{sum(e2es)/len(e2es):>8.1f}ms" if e2es else "         —"
                bearings = sorted({d.get("bearing_name", "?") for d in rdocs})
                bear_str = ", ".join(bearings)[:22]
                print(f"  {rid:<30} {n:>7,} {ok:>5} {err:>5} {avg_p}  {avg_e}  {bear_str}")

        # ── 4. Latency breakdown (the thesis numbers) ────────────────────────
        _hdr("Latency Breakdown — All-Run Combined  (or filtered)", W)
        match = {}
        if self._run_id:
            match["run_id"] = self._run_id
        if self._bearing:
            match["bearing_name"] = self._bearing
        all_docs = list(self._sh.find(match, {"_id": 0}))

        if not all_docs:
            print("  No latency data recorded yet.")
        else:
            pipe = [self._pipeline_ms_of(d) for d in all_docs]
            pipe = [v for v in pipe if v is not None]
            e2e  = [d["e2e_ms"]           for d in all_docs if d.get("e2e_ms")           is not None]
            ing  = [d["ingestion_lag_ms"] for d in all_docs if d.get("ingestion_lag_ms") is not None]

            print(f"  {'Metric':<22}{'n':>7}{'mean':>11}{'p50':>11}"
                  f"{'p95':>11}{'p99':>11}{'max':>11}")
            _div(W)
            for label, vals in (
                ("pipeline_ms",       pipe),
                ("e2e_ms",            e2e),
                ("ingestion_lag_ms",  ing),
            ):
                st = _stats(vals)
                if st:
                    print(f"  {label:<22}{st['n']:>7}{st['mean']:>10.2f}ms"
                          f"{st['p50']:>10.2f}ms{st['p95']:>10.2f}ms"
                          f"{st['p99']:>10.2f}ms{st['max']:>10.2f}ms")
                else:
                    print(f"  {label:<22}{'—':>7}{'—':>11}{'—':>11}"
                          f"{'—':>11}{'—':>11}{'—':>11}")

            # Per-stage
            _hdr("Per-Stage Pipeline Breakdown", W)
            stage_stats = self._stage_table(all_docs)
            print(f"  {'Stage':<22}{'n':>7}{'mean':>11}{'p50':>11}"
                  f"{'p95':>11}{'p99':>11}{'max':>11}")
            _div(W)
            for k in _STAGE_KEYS:
                st = stage_stats.get(k)
                if st is None:
                    print(f"  {k:<22}{'—':>7}{'—':>11}{'—':>11}"
                          f"{'—':>11}{'—':>11}{'—':>11}")
                else:
                    print(f"  {k:<22}{st['n']:>7}{st['mean']:>10.2f}ms"
                          f"{st['p50']:>10.2f}ms{st['p95']:>10.2f}ms"
                          f"{st['p99']:>10.2f}ms{st['max']:>10.2f}ms")

            # Compliance buckets vs the 10 s burst period
            n     = len(pipe)
            if n:
                under_50  = sum(1 for v in pipe if v <  50)
                under_100 = sum(1 for v in pipe if v < 100)
                under_500 = sum(1 for v in pipe if v < 500)
                over_1s   = sum(1 for v in pipe if v >= 1000)
                _hdr("Pipeline Latency Buckets (vs 10s burst period)", W)
                print(f"  < 50 ms  : {under_50:>6,} / {n:,}  ({under_50/n*100:5.1f}%)")
                print(f"  < 100 ms : {under_100:>6,} / {n:,}  ({under_100/n*100:5.1f}%)")
                print(f"  < 500 ms : {under_500:>6,} / {n:,}  ({under_500/n*100:5.1f}%)")
                if over_1s:
                    print(f"  > 1000ms : {over_1s:>6,} / {n:,}  ({over_1s/n*100:5.1f}%)  ⚠️")

        # ── 5. Health summary across all runs ─────────────────────────────────
        _hdr("Overall Health  (all RUL_predictions)", W)
        total_rul = self._rul.count_documents({})
        if total_rul:
            crit  = self._rul.count_documents({"pm_status": "critical"})
            warn  = self._rul.count_documents({"pm_status": "warning"})
            hlthy = self._rul.count_documents({"pm_status": "healthy"})
            drift = self._rul.count_documents({"monitoring.drift_detected": True})
            anom  = self._rul.count_documents({"monitoring.anomaly_flag": True})
            print(f"  Total predictions : {total_rul:,}")
            print(f"  🔴 Critical       : {crit:,}  ({crit/total_rul*100:.1f}%)")
            print(f"  ⚠️  Warning        : {warn:,}  ({warn/total_rul*100:.1f}%)")
            print(f"  ✅ Healthy        : {hlthy:,}  ({hlthy/total_rul*100:.1f}%)")
            print(f"  🔀 Drift events   : {drift:,}  ({drift/total_rul*100:.1f}%)")
            print(f"  🔺 Anomaly events : {anom:,}  ({anom/total_rul*100:.1f}%)")
        else:
            print("  No prediction data yet.")

        # ── 6. Workflow & preprod ─────────────────────────────────────────────
        _hdr("Workflow Registry", W)
        wf_docs = list(self._wr.find({}, {"_id": 0}))
        if not wf_docs:
            print("  No workflows registered yet.")
        else:
            for w in wf_docs:
                print(f"  {w.get('workflow_name','?'):<28} "
                      f"status={w.get('status','?'):<12} "
                      f"version={w.get('version','?')}")

        _hdr("Pre-Production Runs  (last 5)", W)
        pp_docs = list(self._pr.find({}, {"_id": 0}).sort("timestamp", -1).limit(5))
        if not pp_docs:
            print("  No pre-production runs recorded yet.")
        else:
            for p in pp_docs:
                print(f"  run={str(p.get('run_id','?'))[:22]:<24}  "
                      f"bearing={str(p.get('bearing_name','?')):<14}  "
                      f"status={p.get('status','?')}")

        print("\n" + "=" * W)
        print("  End of report.")
        print("=" * W + "\n")

    # ── Thesis CSV export ─────────────────────────────────────────────────────

    def export_csv(self, path: str):
        match = {}
        if self._run_id:
            match["run_id"] = self._run_id
        if self._bearing:
            match["bearing_name"] = self._bearing

        cursor = self._sh.find(match, {"_id": 0}).sort("timestamp", 1)

        fieldnames = [
            "timestamp", "run_id", "bearing_name", "burst_idx", "model_version",
            "pipeline_ok", "pm_status", "rul_s", "rul_min",
            "drift_detected", "anomaly_flag",
            "pipeline_ms", "latency_ms", "ingestion_lag_ms", "e2e_ms",
            "cpu_percent", "memory_mb", "bursts_this_session",
        ] + list(_STAGE_KEYS)

        n_written = 0
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for d in cursor:
                stage = d.get("stage_timings_ms") or {}
                row   = {k: d.get(k) for k in fieldnames if k not in _STAGE_KEYS}
                # ensure timestamp is iso8601
                ts = d.get("timestamp")
                if hasattr(ts, "isoformat"):
                    row["timestamp"] = ts.isoformat()
                for sk in _STAGE_KEYS:
                    row[sk] = stage.get(sk)
                w.writerow(row)
                n_written += 1

        print(f"\n  ✓  Wrote {n_written:,} rows → {os.path.abspath(path)}")
        if n_written == 0:
            print("    (No documents matched the filter — check --run_id / --bearing.)")
        print()

    # ── Run loop ──────────────────────────────────────────────────────────────

    def run(self):
        print("\n  Starting monitor — waiting for data...\n")
        while True:
            try:
                _clear()
                self.render()
            except Exception as e:
                print(f"\n  Monitor error: {e}")
            time.sleep(self._interval)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Live MLOps monitor — polls serving_history for real-time "
                    "stats and supports thesis-ready latency export.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python monitor.py
  python monitor.py --interval 3
  python monitor.py --bearing Bearing1_5
  python monitor.py --run_id serve_xyz
  python monitor.py --summary
  python monitor.py --summary --run_id serve_xyz
  python monitor.py --export-csv latency.csv
  python monitor.py --export-csv latency.csv --run_id serve_xyz
        """,
    )
    parser.add_argument("--mongo_uri",  type=str,   default=DEFAULT_MONGO_URI)
    parser.add_argument("--db_name",    type=str,   default=DEFAULT_DB_NAME)
    parser.add_argument("--interval",   type=float, default=DEFAULT_INTERVAL_S,
                        help=f"Live refresh interval in seconds "
                             f"(default: {DEFAULT_INTERVAL_S})")
    parser.add_argument("--run_id",     type=str,   default=None,
                        help="Filter to a specific run_id (default: latest)")
    parser.add_argument("--bearing",    type=str,   default=None,
                        help="Filter to a specific bearing name")
    parser.add_argument("--summary",    action="store_true", default=False,
                        help="Print a full one-shot system summary report and exit")
    parser.add_argument("--export-csv", type=str,   default=None,
                        dest="export_csv",
                        help="Export filtered serving_history to CSV and exit "
                             "(for thesis analysis)")
    return parser.parse_args()


if __name__ == "__main__":
    args    = _parse_args()
    monitor = LiveMonitor(
        mongo_uri    = args.mongo_uri,
        db_name      = args.db_name,
        interval     = args.interval,
        run_id       = args.run_id,
        bearing_name = args.bearing,
    )

    if args.export_csv:
        monitor.export_csv(args.export_csv)
    elif args.summary:
        monitor.summary()
    else:
        try:
            monitor.run()
        except KeyboardInterrupt:
            print("\n\n  Monitor stopped.")