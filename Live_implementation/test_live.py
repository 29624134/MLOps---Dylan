"""
tests/test_live_pipeline.py
===========================
Unit + integration tests for the live serving pipeline.

Test levels
-----------
1. Unit — LiveFeatureBuffer in isolation (no model, no files)
2. Unit — LivePredictor.from_path() with a toy model saved to a temp file
3. Integration — run_live_pipeline() replaying synthetic bursts end-to-end

Run with:
    pytest tests/test_live_pipeline.py -v

Or for a quick smoke test:
    python tests/test_live_pipeline.py
"""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Live_implementation.live_feature_buffer import (
    LiveFeatureBuffer,
    _burst_features,
    _compute_kurtosis,
    _compute_skewness,
)
from Live_implementation.live_predictor import LivePredictor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIGNAL_LEN = 2560   # typical PHM 2012 burst length
WINDOW_SIZE = 40
N_BASE_FEATURES = 18   # 9 per axis × 2 axes
N_MODEL_FEATURES = N_BASE_FEATURES * 3   # mean + std + slope = 54


def _random_signal(seed=0, length=SIGNAL_LEN) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(length).astype(np.float32)


def _make_toy_checkpoint(input_dim: int, horizon: int = 10, seed: int = 7) -> bytes:
    """Build a minimal valid .pt checkpoint and return its bytes.

    Uses nn.init.normal_ so weights are non-zero, and fits the scaler on
    random data so it doesn't collapse all variance to 0.
    """
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(input_dim, horizon))
    # Ensure non-trivial weights so inputs map to different outputs
    nn.init.normal_(net[0].weight, mean=0.0, std=0.1)
    nn.init.zeros_(net[0].bias)

    # Fit scaler on random data so mean/std are non-trivial
    rng = np.random.default_rng(seed)
    X_fit = rng.standard_normal((64, input_dim)).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(X_fit)

    buf = io.BytesIO()
    joblib.dump(scaler, buf)

    checkpoint = {
        "model_state_dict": net.state_dict(),
        "scaler_bytes":     buf.getvalue(),
        "hyperparameters":  {
            "horizon":      horizon,
            "hidden_units": [],
            "dropout":      0.0,
            "rul_scale":    30000.0,
            "epochs":       1,
            "lr":           1e-4,
            "batch_size":   32,
            "patience":     1,
            "seed":         seed,
        },
    }

    out = io.BytesIO()
    torch.save(checkpoint, out)
    return out.getvalue()


# ---------------------------------------------------------------------------
# 1. Unit tests — statistical helpers
# ---------------------------------------------------------------------------

class TestStatHelpers(unittest.TestCase):

    def test_skewness_symmetric_signal(self):
        """Symmetric signal should have skewness close to 0."""
        x = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
        self.assertAlmostEqual(_compute_skewness(x), 0.0, places=6)

    def test_kurtosis_normal_approx(self):
        """Long normal signal should have excess kurtosis ≈ 0."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100_000)
        self.assertAlmostEqual(_compute_kurtosis(x), 0.0, delta=0.05)

    def test_burst_features_keys(self):
        """_burst_features must return all 9 expected keys."""
        sig = _random_signal()
        feats = _burst_features(sig)
        expected = {"max", "min", "mean", "sd", "rms", "skew", "kurt", "crest", "form"}
        self.assertEqual(set(feats.keys()), expected)

    def test_burst_features_rms_positive(self):
        """RMS must be positive for any non-zero signal."""
        sig = _random_signal()
        feats = _burst_features(sig)
        self.assertGreater(feats["rms"], 0.0)


# ---------------------------------------------------------------------------
# 2. Unit tests — LiveFeatureBuffer
# ---------------------------------------------------------------------------

class TestLiveFeatureBuffer(unittest.TestCase):

    def test_returns_none_while_filling(self):
        buf = LiveFeatureBuffer(window_size=WINDOW_SIZE)
        for i in range(WINDOW_SIZE - 1):
            result = buf.push_burst(_random_signal(i), _random_signal(i + 1000))
            self.assertIsNone(result, f"Expected None at burst {i}")

    def test_returns_array_when_full(self):
        buf = LiveFeatureBuffer(window_size=WINDOW_SIZE)
        result = None
        for i in range(WINDOW_SIZE):
            result = buf.push_burst(_random_signal(i), _random_signal(i + 1000))
        self.assertIsNotNone(result)
        self.assertIsInstance(result, np.ndarray)

    def test_output_shape(self):
        """Output must be shape (54,) = 18 base features × 3 stats."""
        buf = LiveFeatureBuffer(window_size=WINDOW_SIZE)
        result = None
        for i in range(WINDOW_SIZE):
            result = buf.push_burst(_random_signal(i), _random_signal(i + 1000))
        self.assertEqual(result.shape, (N_MODEL_FEATURES,))

    def test_output_dtype_float32(self):
        buf = LiveFeatureBuffer(window_size=WINDOW_SIZE)
        result = None
        for i in range(WINDOW_SIZE):
            result = buf.push_burst(_random_signal(i), _random_signal(i + 1000))
        self.assertEqual(result.dtype, np.float32)

    def test_no_nan_in_output(self):
        buf = LiveFeatureBuffer(window_size=WINDOW_SIZE)
        result = None
        for i in range(WINDOW_SIZE):
            result = buf.push_burst(_random_signal(i), _random_signal(i + 1000))
        self.assertFalse(np.any(np.isnan(result)), "Feature vector contains NaN")

    def test_feature_names_length(self):
        buf = LiveFeatureBuffer(window_size=WINDOW_SIZE)
        self.assertEqual(len(buf.get_feature_names()), N_MODEL_FEATURES)

    def test_feature_names_suffixes(self):
        buf = LiveFeatureBuffer(window_size=WINDOW_SIZE)
        names = buf.get_feature_names()
        for n in names:
            self.assertTrue(
                n.endswith("_mean") or n.endswith("_std") or n.endswith("_slope"),
                f"Unexpected feature name: {n}"
            )

    def test_bursts_seen_counter(self):
        buf = LiveFeatureBuffer(window_size=WINDOW_SIZE)
        for i in range(15):
            buf.push_burst(_random_signal(i), _random_signal(i + 1000))
        self.assertEqual(buf.bursts_seen, 15)

    def test_is_ready_flag(self):
        buf = LiveFeatureBuffer(window_size=5)
        for i in range(4):
            buf.push_burst(_random_signal(i), _random_signal(i + 100))
            self.assertFalse(buf.is_ready)
        buf.push_burst(_random_signal(99), _random_signal(199))
        self.assertTrue(buf.is_ready)

    def test_rolling_window_slides(self):
        """After filling, each new push should produce a different output."""
        buf = LiveFeatureBuffer(window_size=WINDOW_SIZE)
        vec1 = None
        for i in range(WINDOW_SIZE):
            vec1 = buf.push_burst(_random_signal(i), _random_signal(i + 1000))

        # Push a drastically different burst (constant high-g signal)
        high_g = np.full(SIGNAL_LEN, 30.0, dtype=np.float32)
        vec2 = buf.push_burst(high_g, high_g)

        self.assertFalse(
            np.allclose(vec1, vec2),
            "Feature vector did not change after pushing a very different burst"
        )

    def test_parity_with_training_formula(self):
        """
        Manually compute rolling mean for h_max and verify it matches
        the buffer's output — ensuring train/serve parity for that stat.
        """
        buf = LiveFeatureBuffer(window_size=5)
        signals = [_random_signal(i) for i in range(6)]
        h_max_vals = [_burst_features(s)["max"] for s in signals]

        for i, s in enumerate(signals):
            result = buf.push_burst(s, _random_signal(i + 500))

        # After 6 pushes with window=5, window contains bursts 1..5
        expected_mean = float(np.mean(h_max_vals[1:]))   # last 5 values
        # Feature name: "h_max_mean" is index 0 in the output
        self.assertAlmostEqual(float(result[0]), expected_mean, places=5)


# ---------------------------------------------------------------------------
# 3. Unit tests — LivePredictor
# ---------------------------------------------------------------------------

class TestLivePredictor(unittest.TestCase):

    def setUp(self):
        """Write a toy checkpoint to a temp file before each test."""
        self.tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        self.tmp.write(_make_toy_checkpoint(input_dim=N_MODEL_FEATURES, horizon=10))
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_loads_from_path(self):
        p = LivePredictor.from_path(self.tmp.name)
        self.assertIsNotNone(p)

    def test_predict_returns_float(self):
        p = LivePredictor.from_path(self.tmp.name)
        vec = np.zeros(N_MODEL_FEATURES, dtype=np.float32)
        result = p.predict(vec)
        self.assertIsInstance(result, float)

    def test_predict_non_negative(self):
        """Prediction must always be >= 0 (clipped)."""
        p = LivePredictor.from_path(self.tmp.name)
        # Toy model with random weights — output might be negative before clip
        for seed in range(10):
            vec = np.random.default_rng(seed).standard_normal(N_MODEL_FEATURES).astype(np.float32)
            self.assertGreaterEqual(p.predict(vec), 0.0)

    def test_predict_horizon_shape(self):
        p = LivePredictor.from_path(self.tmp.name)
        vec = np.zeros(N_MODEL_FEATURES, dtype=np.float32)
        horizon_preds = p.predict_horizon(vec)
        self.assertEqual(horizon_preds.shape, (10,))

    def test_predict_accepts_1d_and_2d_input(self):
        p = LivePredictor.from_path(self.tmp.name)
        vec_1d = np.zeros(N_MODEL_FEATURES, dtype=np.float32)
        vec_2d = vec_1d.reshape(1, -1)
        r1 = p.predict(vec_1d)
        r2 = p.predict(vec_2d)
        self.assertAlmostEqual(r1, r2, places=5)


# ---------------------------------------------------------------------------
# 4. Integration test — buffer → predictor end-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        self.tmp.write(_make_toy_checkpoint(input_dim=N_MODEL_FEATURES, horizon=10))
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_pipeline_produces_predictions_after_warmup(self):
        """
        Push WINDOW_SIZE + 10 synthetic bursts through buffer → predictor.
        The first WINDOW_SIZE - 1 should yield None; the rest should
        produce valid float predictions.
        """
        buf = LiveFeatureBuffer(window_size=WINDOW_SIZE)
        pred = LivePredictor.from_path(self.tmp.name)

        none_count = 0
        pred_count = 0

        for i in range(WINDOW_SIZE + 10):
            vec = buf.push_burst(_random_signal(i), _random_signal(i + 1000))
            if vec is None:
                none_count += 1
            else:
                rul = pred.predict(vec)
                self.assertGreaterEqual(rul, 0.0)
                pred_count += 1

        self.assertEqual(none_count, WINDOW_SIZE - 1)
        self.assertEqual(pred_count, 11)

    def test_increasing_amplitude_changes_prediction(self):
        """
        Injecting progressively higher-amplitude bursts should change
        the RUL prediction — verifying the model actually sees new data.
        """
        buf = LiveFeatureBuffer(window_size=WINDOW_SIZE)
        pred = LivePredictor.from_path(self.tmp.name)

        # Warm up the buffer with normal signals
        for i in range(WINDOW_SIZE):
            buf.push_burst(_random_signal(i), _random_signal(i + 1000))

        # Baseline prediction
        normal_burst = _random_signal(999)
        vec1 = buf.push_burst(normal_burst, normal_burst)
        rul1 = pred.predict(vec1)

        # Push a very high-amplitude burst 40 more times to fill the window
        high_amp = np.full(SIGNAL_LEN, 25.0, dtype=np.float32)
        vec2 = None
        for _ in range(WINDOW_SIZE):
            vec2 = buf.push_burst(high_amp, high_amp)
        rul2 = pred.predict(vec2)

        # Predictions should differ (toy model, so just check they're not identical)
        self.assertNotAlmostEqual(rul1, rul2, places=1,
                                  msg="Predictions unchanged despite different input signals")


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)