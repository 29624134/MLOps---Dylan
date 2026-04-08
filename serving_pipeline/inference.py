"""
serving_pipeline/inference.py
═══════════════════════════════════════════════════════════════════════════════
Serving Pipeline — Stage 2: Inference

Responsibilities
────────────────
1. Accept the quality-labelled feature dict from the Feature Engineering stage.
2. Load the approved RUL model from the ModelRegistry (versioned artifact).
3. Run prediction via LivePredictor — gracefully handles quality-flagged inputs
   (does not fail; logs what was detected when quality issues are present).
4. Return structured inference output for the Predictive Maintenance stage.

Quality label handling
──────────────────────
Per MLOps Project Instructions §4: inference must NOT fail when quality flags
are present. Instead of a boolean low_confidence flag, the result carries a
descriptive data_quality label explaining what was detected:

    "clean"             — no issues detected
    "outlier_detected"  — one or more features are statistical outliers
    "missing_detected"  — one or more features are NaN or inf
    "spike_detected"    — extreme outlier (z > 6sigma from window history)
    "dropout_detected"  — sensor dropout (RMS near zero)
    "no_model"          — no deployed model found
    "prediction_failed" — model raised an exception during inference

Usage
─────
    from serving_pipeline.inference import ServingInference

    infer = ServingInference()

    result = infer.run(feature_engineering_output)
    # result["rul_s"]          — float, predicted RUL in seconds
    # result["rul_min"]        — float, predicted RUL in minutes
    # result["horizon_preds"]  — list[float], multi-step horizon
    # result["model_version"]  — str
    # result["data_quality"]   — str, descriptive label (see above)
    # result["skipped"]        — bool (True if buffer not yet ready)
"""

import logging
import warnings
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Quality → label mapping
# ─────────────────────────────────────────────────────────────────────────────

def _derive_quality_label(quality: Dict[str, Any]) -> str:
    """
    Convert the Feature Engineering quality dict into a single descriptive
    label that explains what the system detected about this burst.

    Priority order (most severe first):
        missing   -> "missing_detected"
        dropout   -> "dropout_detected"
        spike     -> "spike_detected"
        outlier   -> "outlier_detected"
        none      -> "clean"
    """
    if not quality:
        return "clean"

    anomaly_type = quality.get("anomaly_type", "none")
    has_missing  = quality.get("missing", False)
    has_outlier  = quality.get("outlier", False)

    if has_missing or anomaly_type == "null":
        return "missing_detected"
    if anomaly_type == "dropout":
        return "dropout_detected"
    if anomaly_type == "spike":
        return "spike_detected"
    if has_outlier:
        return "outlier_detected"
    return "clean"


# ─────────────────────────────────────────────────────────────────────────────
# ServingInference
# ─────────────────────────────────────────────────────────────────────────────

class ServingInference:
    """
    Stage 2 of the Serving Pipeline.

    Lazy-loads the deployed model on first call (or when model version changes).

    Parameters
    ──────────
    model_registry_path : str | None
        Path to registry.json.  Defaults to ModelRegistry own default.
    target_feature : str
        The target the deployed model predicts (default "RUL_s").
    """

    def __init__(
        self,
        model_registry_path: Optional[str] = None,
        target_feature: str = "RUL_s",
    ):
        self._registry_path  = model_registry_path
        self._target_feature = target_feature
        self._predictor      = None
        self._model_version  = None
        logger.info("[Inference] ServingInference initialised.")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, fe_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on the output of the Feature Engineering stage.

        Parameters
        ──────────
        fe_output : dict produced by ServingFeatureEngineer.process_burst()

        Returns
        ───────
        dict with keys:
            skipped        : bool         — True if window not yet full
            rul_s          : float | None — predicted RUL in seconds
            rul_min        : float | None — predicted RUL in minutes
            horizon_preds  : list[float]  — multi-step horizon predictions
            model_version  : str | None   — model ID from ModelRegistry
            data_quality   : str          — descriptive label of what was detected
            quality_labels : dict | None  — full FE quality dict (passed through)
        """
        # ── Not ready yet (window still filling) ─────────────────────────────
        if not fe_output.get("ready", False):
            return {
                "skipped":        True,
                "rul_s":          None,
                "rul_min":        None,
                "horizon_preds":  [],
                "model_version":  None,
                "data_quality":   "clean",
                "quality_labels": fe_output.get("quality_labels"),
            }

        feature_vector = fe_output["feature_vector"]
        quality        = fe_output.get("quality_labels") or {}
        data_quality   = _derive_quality_label(quality)

        if data_quality != "clean":
            logger.warning(
                f"[Inference] Burst {fe_output.get('burst_idx')}: "
                f"data_quality='{data_quality}'  "
                f"outlier={quality.get('outlier')}  "
                f"missing={quality.get('missing')}  "
                f"anomaly_type={quality.get('anomaly_type')}  "
                f"— prediction will proceed"
            )

        # ── Ensure predictor is loaded ────────────────────────────────────────
        self._ensure_predictor()

        if self._predictor is None:
            logger.error("[Inference] No deployed model available — cannot predict.")
            return {
                "skipped":        False,
                "rul_s":          None,
                "rul_min":        None,
                "horizon_preds":  [],
                "model_version":  None,
                "data_quality":   "no_model",
                "quality_labels": quality,
                "error":          "No deployed model found in ModelRegistry.",
            }

        # ── Run prediction ────────────────────────────────────────────────────
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                rul_s         = self._predictor.predict(feature_vector)
                horizon_preds = self._predictor.predict_horizon(feature_vector).tolist()
        except Exception as exc:
            logger.error(f"[Inference] Prediction failed: {exc}")
            return {
                "skipped":        False,
                "rul_s":          None,
                "rul_min":        None,
                "horizon_preds":  [],
                "model_version":  self._model_version,
                "data_quality":   "prediction_failed",
                "quality_labels": quality,
                "error":          str(exc),
            }

        rul_min = rul_s / 60.0 if rul_s is not None else None

        logger.info(
            f"[Inference] Burst {fe_output.get('burst_idx')}: "
            f"RUL={rul_s:.1f}s ({rul_min:.2f} min)  "
            f"model={self._model_version}  "
            f"data_quality='{data_quality}'"
        )

        return {
            "skipped":        False,
            "rul_s":          rul_s,
            "rul_min":        rul_min,
            "horizon_preds":  horizon_preds,
            "model_version":  self._model_version,
            "data_quality":   data_quality,
            "quality_labels": quality,
        }

    def reload_model(self) -> None:
        """Force reload the deployed model from the registry."""
        self._predictor     = None
        self._model_version = None
        self._ensure_predictor()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _ensure_predictor(self) -> None:
        """Lazy-load the deployed model if not already loaded."""
        if self._predictor is not None:
            return

        try:
            from utils.model_registry import ModelRegistry
            from Live_implementation.live_predictor import LivePredictor

            kwargs = {}
            if self._registry_path:
                kwargs["registry_path"] = self._registry_path

            registry    = ModelRegistry(**kwargs)
            model_entry = registry.get_deployed_model(self._target_feature)

            if not model_entry:
                logger.warning(
                    f"[Inference] No deployed model for target '{self._target_feature}'."
                )
                return

            self._predictor     = LivePredictor.from_path(model_entry["model_path"])
            self._model_version = model_entry["model_id"]
            logger.info(
                f"[Inference] Loaded model '{self._model_version}' "
                f"from {model_entry['model_path']}"
            )

        except Exception as exc:
            logger.error(f"[Inference] Could not load model: {exc}")