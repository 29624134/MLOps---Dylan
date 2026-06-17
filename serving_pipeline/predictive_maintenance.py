"""
serving_pipeline/predictive_maintenance.py
═══════════════════════════════════════════════════════════════════════════════
Serving Pipeline — Stage 3: Predictive Maintenance (RUL)

Responsibilities
────────────────
1. Accept raw inference output (RUL in seconds, horizon predictions).
2. Apply configurable business-logic thresholds to produce a human-readable
   maintenance status: "healthy" | "warning" | "critical".
3. Attach alert conditions and recommended actions.
4. Pass low-confidence flags from Inference through to Serving History.

Default thresholds (tunable via config dict)
────────────────────────────────────────────
    critical : RUL_s ≤  1 hour  (3 600 s)
    warning  : RUL_s ≤  4 hours (14 400 s)
    healthy  : RUL_s >  4 hours

These are conservative defaults for industrial bearing maintenance and can
be overridden per deployment via the config dict passed to the constructor.

Usage
─────
    from serving_pipeline.predictive_maintenance import PredictiveMaintenance

    pm = PredictiveMaintenance()                          # default thresholds
    pm = PredictiveMaintenance(config={                   # custom thresholds
        "critical_threshold_s": 7200,
        "warning_threshold_s":  28800,
    })

    result = pm.run(inference_output)
    # result["status"]            — "healthy" | "warning" | "critical" | "unknown"
    # result["rul_s"]             — float
    # result["rul_min"]           — float
    # result["rul_hours"]         — float
    # result["alert"]             — bool
    # result["recommended_action"]— str
    # result["low_confidence"]    — bool (passed through from Inference)
    # result["horizon_preds"]     — list[float]
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Default threshold values (seconds)
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_CRITICAL_S = 1_440   # 1 hour
_DEFAULT_WARNING_S  = 1_800   # 4 hours

# Recommended actions by status
_RECOMMENDED_ACTIONS = {
    "critical": (
        "IMMEDIATE ACTION REQUIRED: Schedule bearing replacement within 1 hour. "
        "Reduce machine load and increase monitoring frequency."
    ),
    "warning": (
        "MAINTENANCE ADVISORY: Plan bearing replacement within 4 hours. "
        "Monitor vibration levels closely for further degradation."
    ),
    "healthy": (
        "No immediate action required. Continue standard monitoring schedule."
    ),
    "unknown": (
        "RUL estimate unavailable — check model deployment and sensor data."
    ),
}


class PredictiveMaintenance:
    """
    Stage 3 of the Serving Pipeline.

    Converts raw RUL seconds into an actionable maintenance status with
    threshold-based alerting and recommended actions.

    Parameters
    ──────────
    config : dict | None
        Optional overrides:
            critical_threshold_s : int   (default 3 600)
            warning_threshold_s  : int   (default 14 400)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self._critical_s = int(cfg.get("critical_threshold_s", _DEFAULT_CRITICAL_S))
        self._warning_s  = int(cfg.get("warning_threshold_s",  _DEFAULT_WARNING_S))

        if self._critical_s >= self._warning_s:
            raise ValueError(
                f"critical_threshold_s ({self._critical_s}) must be "
                f"< warning_threshold_s ({self._warning_s})"
            )

        logger.info(
            f"[PM] PredictiveMaintenance initialised — "
            f"critical≤{self._critical_s}s  warning≤{self._warning_s}s"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Derive maintenance status from Inference stage output.

        Parameters
        ──────────
        inference_output : dict produced by ServingInference.run()

        Returns
        ───────
        dict with keys:
            status              : str    — "healthy"|"warning"|"critical"|"unknown"
            rul_s               : float | None
            rul_min             : float | None
            rul_hours           : float | None
            alert               : bool
            recommended_action  : str
            low_confidence      : bool
            horizon_preds       : list[float]
            thresholds          : dict   — the thresholds used
            evaluated_at        : str    — ISO-8601 UTC timestamp
        """
        # ── Skipped / no prediction ───────────────────────────────────────────
        if inference_output.get("skipped") or inference_output.get("rul_s") is None:
            return self._unknown_result(inference_output)

        rul_s          = float(inference_output["rul_s"])
        rul_min        = rul_s / 60.0
        rul_hours      = rul_s / 3600.0
        low_confidence = inference_output.get("data_quality", "clean") != "clean"
        horizon_preds  = inference_output.get("horizon_preds", [])

        # ── Threshold classification ──────────────────────────────────────────
        status = self._classify(rul_s)
        alert  = status in ("critical", "warning")

        if alert:
            logger.warning(
                f"[PM] ALERT — status={status}  RUL={rul_s:.0f}s ({rul_hours:.2f}h)"
                + (f"  [data_quality={inference_output.get('data_quality', 'clean')}]" if low_confidence else "")
            )
        else:
            logger.info(
                f"[PM] status={status}  RUL={rul_s:.0f}s ({rul_hours:.2f}h)"
            )

        return {
            "status":             status,
            "rul_s":              rul_s,
            "rul_min":            round(rul_min, 3),
            "rul_hours":          round(rul_hours, 4),
            "alert":              alert,
            "recommended_action": _RECOMMENDED_ACTIONS[status],
            "data_quality":       inference_output.get("data_quality", "clean"),
            "low_confidence":     low_confidence,  # kept for backward compat
            "horizon_preds":      horizon_preds,
            "thresholds": {
                "critical_s": self._critical_s,
                "warning_s":  self._warning_s,
            },
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _classify(self, rul_s: float) -> str:
        if rul_s <= self._critical_s:
            return "critical"
        if rul_s <= self._warning_s:
            return "warning"
        return "healthy"

    def _unknown_result(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status":             "unknown",
            "rul_s":              None,
            "rul_min":            None,
            "rul_hours":          None,
            "alert":              False,
            "recommended_action": _RECOMMENDED_ACTIONS["unknown"],
            "low_confidence":     inference_output.get("low_confidence", True),
            "horizon_preds":      inference_output.get("horizon_preds", []),
            "thresholds": {
                "critical_s": self._critical_s,
                "warning_s":  self._warning_s,
            },
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "skipped":      inference_output.get("skipped", False),
        }