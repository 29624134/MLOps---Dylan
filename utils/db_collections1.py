"""
utils/db_collections.py
═══════════════════════════════════════════════════════════════════════════════
Central definition of all MongoDB collection names.

Import from here everywhere — never hardcode collection names in other files.

Layout
──────
phm_mlops (database)
├── factory_features        ← Feature Store: historical train/val features
│                             seeded by seed_historical_data.py
├── feature_store           ← Live Feature Store: SCADA bursts (live_features)
│                             + monitoring metrics written back by run_serving.py
├── feature_store_mirrored  ← Confirmed fault features for retraining
│                             written by confirm_fault_and_push_to_store()
├── serving_history         ← Full prediction audit log, one doc per burst
├── model_registry          ← Model lifecycle: pending→approved→deployed→archived
├── workflow_registry       ← Workflow DAG versions (CI/CD)
├── serving_lock            ← Burst-boundary lock (prevents mid-burst hot-swap)
└── preprod_runs            ← Tracks which fault data has been used in retraining
═══════════════════════════════════════════════════════════════════════════════
"""

# ── Feature stores ─────────────────────────────────────────────────────────────

# Historical run-to-failure features for train/val bearings.
# Written once by seed_historical_data.py; read by orchestrator for training.
COL_FACTORY_FEATURES = "factory_features"

# Live SCADA burst data written by scada_simulator.py.
# Also receives monitoring metrics written back by run_serving.py.
# Replaces old: 'live_features' + 'monitoring_metrics'
COL_FEATURE_STORE = "feature_store"

# Confirmed fault features for pre-production retraining.
# Written by confirm_fault_and_push_to_store(); read by run_preprod.py.
# Replaces old: 'confirmed_faults'
COL_FEATURE_STORE_MIRRORED = "feature_store_mirrored"

# ── Serving & audit ────────────────────────────────────────────────────────────

# Full pipeline output per burst — inference, PM status, monitoring flags.
# Written by run_serving.py; read by both GUIs and the API audit endpoint.
COL_SERVING_HISTORY = "serving_history"

# ── Registries ─────────────────────────────────────────────────────────────────

# Model lifecycle tracking (MongoDB-backed ModelRegistry).
COL_MODEL_REGISTRY = "model_registry"

# Workflow DAG version tracking (MongoDB-backed WorkflowRegistry).
COL_WORKFLOW_REGISTRY = "workflow_registry"

# ── Internal / operational ────────────────────────────────────────────────────

# Burst-boundary lock written by run_serving.py before each burst.
# model_registry.write_champion_pointer() waits for this to clear.
COL_SERVING_LOCK = "serving_lock"

# Records which confirmed fault bearings have already been used in retraining.
# Prevents double-training on the same fault data across multiple preprod runs.
COL_PREPROD_RUNS = "preprod_runs"