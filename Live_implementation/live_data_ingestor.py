"""
Live Pipeline
=============
Wires DataIngestorPHM (streaming) → LiveFeatureBuffer → LivePredictor
into a single end-to-end serving loop.

Run modes
---------
1. Production (realtime=True):
       python live_pipeline.py --bearing_path /path/to/BearingX_Y
                               --model_path   workflow_data/.../rul_model.pt

2. Fast replay / smoke test (realtime=False):
       python live_pipeline.py --bearing_path /path/to/BearingX_Y
                               --model_path   workflow_data/.../rul_model.pt
                               --no_realtime

3. From registry (deployed model):
       python live_pipeline.py --bearing_path /path/to/BearingX_Y
                               --from_registry
                               --run_id __deployed__
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Allow running from repo root or from the Live implementation/ folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Live_implementation.live_feature_buffer import LiveFeatureBuffer
from Live_implementation.live_predictor import LivePredictor
from scripts import DataIngestorPHM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("live_pipeline")


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def run_live_pipeline(
    bearing_path: str,
    predictor: LivePredictor,
    window_size: int = 40,
    burst_period: float = 10.0,
    realtime: bool = False,
    max_bursts: Optional[int] = None,
) -> list:
    """
    Stream bursts from bearing_path through the buffer and predictor.

    Parameters
    ----------
    bearing_path : str  — folder containing acc_*.csv files
    predictor    : LivePredictor — loaded and ready
    window_size  : int  — rolling window (must match training, default 40)
    burst_period : float — seconds between bursts (default 10.0)
    realtime     : bool — if True, sleep burst_period between bursts
    max_bursts   : int  — stop after this many bursts (None = all)

    Returns
    -------
    list of dicts: [{burst_idx, time_s, rul_s, rul_min, h_max, v_max}, ...]
    """
    buffer = LiveFeatureBuffer(window_size=window_size)
    ingestor = DataIngestorPHM(config={
        "input_location":  bearing_path,
        "output_location": bearing_path,
    })

    results = []
    logger.info(
        f"Starting live pipeline | bearing={bearing_path} "
        f"| window={window_size} | realtime={realtime}"
    )

    for burst in ingestor.stream_bursts(
        bearing_path, burst_period=burst_period, realtime=realtime
    ):
        if max_bursts and burst["burst_idx"] >= max_bursts:
            logger.info(f"Reached max_bursts={max_bursts}, stopping.")
            break

        feature_vec = buffer.push_burst(burst["h_signal"], burst["v_signal"])

        if feature_vec is None:
            # Still filling the buffer — nothing to predict yet
            continue

        rul_s = predictor.predict(feature_vec)
        rul_min = rul_s / 60.0

        result = {
            "burst_idx": burst["burst_idx"],
            "time_s":    burst["time_s"],
            "rul_s":     rul_s,
            "rul_min":   rul_min,
            "h_max":     burst["h_max"],
            "v_max":     burst["v_max"],
        }
        results.append(result)

        logger.info(
            f"Burst {burst['burst_idx']:>5} | t={burst['time_s']:>8.0f}s "
            f"| peak_h={burst['h_max']:>6.2f}g  peak_v={burst['v_max']:>6.2f}g "
            f"| RUL = {rul_s:>8.0f} s  ({rul_min:>6.1f} min)"
        )

    logger.info(
        f"Pipeline complete | {buffer.bursts_seen} bursts processed "
        f"| {len(results)} predictions made"
    )
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PHM 2012 Live RUL Pipeline")
    p.add_argument("--bearing_path", required=True,
                   help="Path to bearing folder containing acc_*.csv files")

    # Model source — exactly one of these must be provided
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model_path",
                     help="Direct path to a .pt checkpoint file")
    src.add_argument("--from_registry", action="store_true",
                     help="Load the currently deployed model from ModelRegistry")

    p.add_argument("--run_id", default="__deployed__",
                   help="ModelRegistry run_id (used with --from_registry)")
    p.add_argument("--window_size", type=int, default=40)
    p.add_argument("--burst_period", type=float, default=10.0)
    p.add_argument("--no_realtime", action="store_true",
                   help="Disable sleep between bursts (fast replay / test mode)")
    p.add_argument("--max_bursts", type=int, default=None,
                   help="Stop after N bursts (useful for smoke tests)")
    p.add_argument("--rul_scale", type=float, default=30000.0)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.from_registry:
        predictor = LivePredictor.from_registry(
            run_id=args.run_id, rul_scale=args.rul_scale
        )
    else:
        predictor = LivePredictor.from_path(
            args.model_path, rul_scale=args.rul_scale
        )

    results = run_live_pipeline(
        bearing_path=args.bearing_path,
        predictor=predictor,
        window_size=args.window_size,
        burst_period=args.burst_period,
        realtime=not args.no_realtime,
        max_bursts=args.max_bursts,
    )

    if results:
        final = results[-1]
        logger.info(
            f"\n=== Final prediction ===\n"
            f"  Bearing : {args.bearing_path}\n"
            f"  Burst   : {final['burst_idx']}\n"
            f"  Time    : {final['time_s']:.0f} s\n"
            f"  RUL     : {final['rul_s']:.0f} s  ({final['rul_min']:.1f} min)\n"
        )