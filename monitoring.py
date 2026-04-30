"""
monitor.py
═══════════════════════════════════════════════════════════════════════════════
Live MLOps System Monitor — run this in a separate PyCharm terminal while
scada_simulator.py and run_serving.py are running.

Two modes
─────────
Live mode (default):
    Polls MongoDB serving_history every N seconds and prints a refreshing
    dashboard showing latency, CPU, memory, throughput, health, RUL trend.

Summary mode (--summary):
    One-shot full system report — all collections, all bearings, all runs,
    model registry state, database sizes. Useful after a run completes.

Usage
─────
    python monitor.py                          # live dashboard, latest run
    python monitor.py --interval 3             # refresh every 3s
    python monitor.py --run_id serve_xyz       # filter to specific run
    python monitor.py --bearing Bearing1_5     # filter to specific bearing
    python monitor.py --summary                # full one-shot system report

Stop live mode with Ctrl+C.
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MONGO_URI  = "mongodb://localhost:27017"
DEFAULT_DB_NAME    = "phm_mlops"
DEFAULT_INTERVAL_S = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clear():
    os.system("cls" if os.name == "nt" else "clear")


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
        return "   —   "
    return f"{val:>7.1f} ms"


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
        W = 65

        latencies     = [d["latency_ms"]  for d in docs if d.get("latency_ms")  is not None]
        cpu_vals      = [d["cpu_percent"] for d in docs if d.get("cpu_percent") is not None]
        mem_vals      = [d["memory_mb"]   for d in docs if d.get("memory_mb")   is not None]
        rul_mins      = [d["rul_min"]     for d in docs if d.get("rul_min")     is not None]
        statuses      = [d.get("pm_status", "unknown") for d in docs]
        models        = list({d["model_version"] for d in docs if d.get("model_version")})
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
        latest_lat    = latest.get("latency_ms")
        latest_cpu    = latest.get("cpu_percent")
        latest_mem    = latest.get("memory_mb")

        avg_lat = sum(latencies) / len(latencies) if latencies else None
        max_lat = max(latencies)                  if latencies else None
        min_lat = min(latencies)                  if latencies else None
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else None
        avg_cpu = sum(cpu_vals) / len(cpu_vals)   if cpu_vals  else None
        max_cpu = max(cpu_vals)                   if cpu_vals  else None
        avg_mem = sum(mem_vals) / len(mem_vals)   if mem_vals  else None

        cutoff   = time.time() - 60
        recent   = [d for d in docs if d.get("timestamp") and
                    d["timestamp"].timestamp() > cutoff]
        tput_60s = len(recent)

        print("=" * W)
        print(f"  PHM MLOps — Live Monitor         {now_str}")
        print("=" * W)
        print(f"  Run ID   : {run_id}")
        print(f"  Bearing  : {latest_bear}   Burst #{latest_burst}")
        print(f"  Models   : {', '.join(models) if models else '—'}")
        _div(W)

        _hdr("Latest Burst", W)
        print(f"  Status   : {_status_icon(latest_status)}  {latest_status.upper()}")
        print(f"  RUL      : {_fmt_rul(latest_rul)}")
        print(f"  Latency  : {_fmt_ms(latest_lat)}")
        if latest_cpu is not None:
            print(f"  CPU      : {_bar(latest_cpu, 100, width=25)}")
        if latest_mem is not None:
            print(f"  Memory   : {latest_mem:.1f} MB")

        _hdr(f"Latency  (last {len(latencies)} bursts)", W)
        print(f"  Avg      : {_fmt_ms(avg_lat)}")
        print(f"  Min      : {_fmt_ms(min_lat)}")
        print(f"  Max      : {_fmt_ms(max_lat)}")
        if p95_lat is not None:
            print(f"  p95      : {_fmt_ms(p95_lat)}")
        if avg_lat is not None:
            print(f"  Budget   : {_bar(avg_lat, 10000, width=25, warn=0.02, crit=0.1)}  (10s period)")

        _hdr("Resource Usage", W)
        if avg_cpu is not None:
            print(f"  CPU avg  : {_bar(avg_cpu, 100, width=25)}")
            print(f"  CPU max  : {_bar(max_cpu, 100, width=25)}")
        else:
            print("  CPU      : — (psutil not installed or no data yet)")
        if avg_mem is not None:
            print(f"  Mem avg  : {avg_mem:.1f} MB")

        _hdr("Throughput", W)
        print(f"  Total bursts   : {total_bursts:,}")
        print(f"  Last 60s       : {tput_60s} bursts")
        print(f"  Pipeline OK    : {ok_count}   Errors: {err_count}")

        _hdr(f"Health  (last {len(docs)} bursts)", W)
        print(f"  🔴 Critical : {statuses.count('critical')}")
        print(f"  ⚠️  Warning  : {statuses.count('warning')}")
        print(f"  ✅ Healthy  : {statuses.count('healthy')}")
        print(f"  🔀 Drift    : {drift_count}   🔺 Anomaly: {anomaly_count}")

        if len(rul_mins) >= 3:
            trend_vals = list(reversed(rul_mins[:10]))
            trend_str  = "  " + "".join(
                "▁▂▃▄▅▆▇█"[min(int((v / max(rul_mins[:10])) * 8), 7)]
                if max(rul_mins[:10]) > 0 else "▁"
                for v in trend_vals
            )
            _hdr(f"RUL Trend  (oldest→latest, last {len(trend_vals)} bursts)", W)
            print(f"  {trend_str}   latest: {_fmt_rul(rul_mins[0])}")

        print("\n" + "=" * W)
        print(f"  Refreshing every {self._interval}s  |  --summary for full report  |  Ctrl+C to stop")

    # ── Summary report ────────────────────────────────────────────────────────

    def summary(self):
        W   = 70
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("\n" + "=" * W)
        print(f"  PHM MLOps — SYSTEM SUMMARY REPORT")
        print(f"  Generated : {now}")
        print(f"  Database  : {self._db_name}")
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
            print(f"  {'':2} {'Model ID':<14} {'Status':<12} {'Target':<8} {'MAE_s':>7}  {'RMSE_s':>7}  {'MAPE':>6}  {'Registered'}")
            _div(W)
            for m in sorted(models, key=lambda x: x.get("registered_at", "")):
                mid    = (m.get("model_id") or "?")[:12]
                status = m.get("status", "?")
                target = m.get("target_feature", "?")
                mets   = m.get("metrics", {})
                mae    = mets.get("mae_s")
                rmse   = mets.get("rmse_s")
                mape   = mets.get("mape")
                mae_s  = f"{mae:>7.0f}" if mae   is not None else "      —"
                rms_s  = f"{rmse:>7.0f}" if rmse  is not None else "      —"
                map_s  = f"{mape:>5.1f}%" if mape  is not None else "     —"
                reg_at = (m.get("registered_at") or "")[:16]
                icon   = "🚀" if status == "deployed" else ("✅" if status == "approved" else "  ")
                print(f"  {icon} {mid:<14} {status:<12} {target:<8} {mae_s}  {rms_s}  {map_s}  {reg_at}")

        # ── 3. Serving runs ───────────────────────────────────────────────────
        _hdr("Serving Runs  (serving_history)", W)
        run_ids = self._sh.distinct("run_id")
        if not run_ids:
            print("  No serving runs recorded yet.")
        else:
            print(f"  {'Run ID':<30} {'Bursts':>7} {'OK':>5} {'Err':>5} {'Avg lat':>10}  {'Bearing(s)'}")
            _div(W)
            for rid in sorted(run_ids, reverse=True):
                rdocs    = list(self._sh.find({"run_id": rid}, {"_id": 0}))
                n        = len(rdocs)
                ok       = sum(1 for d in rdocs if d.get("pipeline_ok"))
                err      = n - ok
                lats     = [d["latency_ms"] for d in rdocs if d.get("latency_ms") is not None]
                avg_l    = f"{sum(lats)/len(lats):>8.1f} ms" if lats else "         —"
                bearings = sorted({d.get("bearing_name", "?") for d in rdocs})
                bear_str = ", ".join(bearings)[:22]
                print(f"  {rid:<30} {n:>7,} {ok:>5} {err:>5} {avg_l}  {bear_str}")

        # ── 4. Per-bearing RUL summary ────────────────────────────────────────
        _hdr("Per-Bearing RUL Summary  (RUL_predictions)", W)
        bearing_ids = self._rul.distinct("bearing_name")
        if not bearing_ids:
            print("  No RUL predictions recorded yet.")
        else:
            print(f"  {'Bearing':<16} {'Bursts':>7} {'Last RUL':>9} {'Status':<10} {'Drift':>6} {'Anomaly':>8}")
            _div(W)
            for b in sorted(bearing_ids):
                bdocs  = list(self._rul.find({"bearing_name": b}, {"_id": 0})
                              .sort("timestamp", -1).limit(200))
                n      = self._rul.count_documents({"bearing_name": b})
                latest = bdocs[0] if bdocs else {}
                inf    = latest.get("inference", {}) or {}
                rul    = inf.get("rul_min") or latest.get("rul_min")
                status = latest.get("pm_status", "?")
                drift  = sum(1 for d in bdocs
                             if (d.get("monitoring") or {}).get("drift_detected"))
                anom   = sum(1 for d in bdocs
                             if (d.get("monitoring") or {}).get("anomaly_flag"))
                rul_s  = _fmt_rul(rul)
                icon   = _status_icon(status)
                print(f"  {b:<16} {n:>7,} {rul_s:>9} {icon} {status:<8} {drift:>6} {anom:>8}")

        # ── 5. Feature stores ─────────────────────────────────────────────────
        _hdr("Feature Stores", W)
        ff_bearings  = self._ff.distinct("dataset_id")
        fs_bearings  = self._fs.distinct("bearing_name")
        fsm_bearings = self._fsm.distinct("dataset_id")

        ff_list  = ", ".join(sorted(ff_bearings)[:5])  + ("..." if len(ff_bearings)  > 5 else "")
        fsm_list = ", ".join(sorted(fsm_bearings)[:5]) + ("..." if len(fsm_bearings) > 5 else "")

        print(f"  factory_features       : {self._ff.count_documents({}):>8,} docs  "
              f"{len(ff_bearings)} bearings")
        if ff_list:
            print(f"    Bearings : {ff_list}")
        print(f"  feature_store (live)   : {self._fs.count_documents({}):>8,} docs  "
              f"{len(fs_bearings)} bearing(s) active")
        print(f"  feature_store_mirrored : {self._fsm.count_documents({}):>8,} docs  "
              f"{len(fsm_bearings)} confirmed fault bearing(s)")
        if fsm_list:
            print(f"    Bearings : {fsm_list}")

        # ── 6. Latency breakdown across all runs ──────────────────────────────
        _hdr("Latency Breakdown  (all runs combined)", W)
        all_lats = [d["latency_ms"] for d in
                    self._sh.find({}, {"latency_ms": 1, "_id": 0})
                    if d.get("latency_ms") is not None]
        if all_lats:
            s     = sorted(all_lats)
            n     = len(s)
            mean  = sum(s) / n
            under_50  = sum(1 for l in s if l <  50)
            under_100 = sum(1 for l in s if l < 100)
            under_500 = sum(1 for l in s if l < 500)
            print(f"  Samples  : {n:,}")
            print(f"  Mean     : {_fmt_ms(mean)}")
            print(f"  Median   : {_fmt_ms(s[n//2])}")
            print(f"  p95      : {_fmt_ms(s[int(n*0.95)])}")
            print(f"  p99      : {_fmt_ms(s[int(n*0.99)])}")
            print(f"  Max      : {_fmt_ms(max(s))}")
            print(f"  < 50ms   : {under_50:,} / {n:,}  ({under_50/n*100:.1f}%)  ← Excellent")
            print(f"  < 100ms  : {under_100:,} / {n:,}  ({under_100/n*100:.1f}%)  ← Good")
            print(f"  < 500ms  : {under_500:,} / {n:,}  ({under_500/n*100:.1f}%)  ← Acceptable")
            over_1s = sum(1 for l in s if l >= 1000)
            if over_1s:
                print(f"  > 1000ms : {over_1s:,} / {n:,}  ({over_1s/n*100:.1f}%)  ← ⚠️  Investigate")
        else:
            print("  No latency data recorded yet.")

        # ── 7. Health summary across all runs ─────────────────────────────────
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

        # ── 8. Workflow & preprod ─────────────────────────────────────────────
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
        description="Live MLOps monitor — polls serving_history for real-time stats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python monitor.py                        # live dashboard, auto-latest run
  python monitor.py --interval 3           # refresh every 3s
  python monitor.py --bearing Bearing1_5   # filter to one bearing
  python monitor.py --run_id serve_xyz     # filter to specific run
  python monitor.py --summary              # full one-shot system report
        """,
    )
    parser.add_argument("--mongo_uri", type=str,   default=DEFAULT_MONGO_URI)
    parser.add_argument("--db_name",   type=str,   default=DEFAULT_DB_NAME)
    parser.add_argument("--interval",  type=float, default=DEFAULT_INTERVAL_S,
                        help=f"Live refresh interval in seconds (default: {DEFAULT_INTERVAL_S})")
    parser.add_argument("--run_id",    type=str,   default=None,
                        help="Filter to a specific run_id (default: latest)")
    parser.add_argument("--bearing",   type=str,   default=None,
                        help="Filter to a specific bearing name")
    parser.add_argument("--summary",   action="store_true", default=False,
                        help="Print a full one-shot system summary report and exit")
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
    if args.summary:
        monitor.summary()
    else:
        try:
            monitor.run()
        except KeyboardInterrupt:
            print("\n\n  Monitor stopped.")