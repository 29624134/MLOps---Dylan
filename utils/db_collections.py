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
├── RUL_predictions         ← Full prediction audit log, one doc per burst
├── serving_history         ← logs serving latency, throughput (e.g., number
│                             of API calls), resource usage (e.g., CPU/database
│                             usage), model/correction metrics, model versions,
│                             and feedback metadata
├── model_registry          ← Model lifecycle: pending→approved→deployed→archived
├── workflow_registry       ← Workflow DAG versions (CI/CD)
└── preprod_runs            ← Tracks which fault data has been used in retraining
═══════════════════════════════════════════════════════════════════════════════
"""

# ── Feature stores ─────────────────────────────────────────────────────────────

# Historical run-to-failure features for train/val bearings.
# Written once by seed_historical_data.py; read by orchestrator for training.
# Never read from CSV after seeding — all training data comes from here.
COL_FACTORY_FEATURES = "factory_features"

# Live SCADA burst data written by scada_simulator.py.
# Also receives monitoring metrics written back by run_serving.py.
# Replaces old: 'live_features' + 'monitoring_metrics'
COL_FEATURE_STORE = "feature_store"

# Confirmed fault features for pre-production retraining.
# Written by confirm_fault_and_push_to_store(); read by run_preprod.py.
# Replaces old: 'confirmed_faults'
COL_FEATURE_STORE_MIRRORED = "feature_store_mirrored"

# ── Predictions ────────────────────────────────────────────────────────────────

# Full prediction audit log — one document per burst.
# Written by run_serving.py; stores raw inference output, PM status,
# monitoring flags, feature snapshot, and model version for each burst.
COL_RUL_PREDICTIONS = "RUL_predictions"

# ── Serving & audit ────────────────────────────────────────────────────────────

# Serving telemetry and operational metadata per burst:
# latency, throughput (API call counts), resource usage (CPU/DB),
# model/correction metrics, model versions, and feedback metadata.
# Written by run_serving.py; read by the Dashboard and Audit Service.
COL_SERVING_HISTORY = "serving_history"

# ── Registries ─────────────────────────────────────────────────────────────────

# Model lifecycle tracking (MongoDB-backed ModelRegistry).
# lifecycle: pending → approved → deployed → archived
COL_MODEL_REGISTRY = "model_registry"

# Workflow DAG version tracking (MongoDB-backed WorkflowRegistry).
COL_WORKFLOW_REGISTRY = "workflow_registry"

# ── Internal / operational ────────────────────────────────────────────────────

# Burst-boundary lock written by run_serving.py before each burst.
# model_registry.write_champion_pointer() waits for this to clear.
# Internal use only — not part of the public collection layout.
COL_SERVING_LOCK = "serving_lock"

# Records which confirmed fault bearings have already been used in retraining.
# Prevents double-training on the same fault data across multiple preprod runs.
COL_PREPROD_RUNS = "preprod_runs"