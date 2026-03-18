import numpy as np
import pandas as pd

# ======================================================
# CONFIG
# ======================================================
BURST_PERIOD      = 10.0   # seconds per burst
FAILURE_THRESHOLD = 20.0   # g — PHM 2012 failure criterion (peak acceleration)


# ======================================================
# FEATURE FUNCTIONS
# ======================================================
def compute_skewness(x):
    N  = len(x)
    mu = np.mean(x)
    s3 = np.std(x, ddof=1) ** 3
    return (np.sum((x - mu) ** 3) / N) / s3 if s3 != 0 else 0.0

def compute_kurtosis(x):
    N  = len(x)
    mu = np.mean(x)
    s4 = np.std(x, ddof=1) ** 4
    return (np.sum((x - mu) ** 4) / N) / s4 - 3 if s4 != 0 else 0.0

def burst_features(sig):
    mx    = float(np.max(sig))
    mn    = float(np.min(sig))
    mean  = float(np.mean(sig))
    sd    = float(np.std(sig, ddof=1))
    rms   = float(np.sqrt(np.mean(sig ** 2)))
    skew  = compute_skewness(sig)
    kurt  = compute_kurtosis(sig)
    crest = mx / rms   if rms  != 0 else 0.0
    form  = rms / mean if mean != 0 else 0.0
    return mx, mn, mean, sd, rms, skew, kurt, crest, form


# ======================================================
# FAILURE TIME DETECTION  (training bearings only)
# ======================================================
def find_failure_time_s(df_feat, threshold=FAILURE_THRESHOLD):
    """
    Scan bursts chronologically. Return the time_s of the FIRST burst where
    peak acceleration (max of |h_max|, |v_max|) crosses the threshold.
    Falls back to the last burst if the threshold is never exceeded.
    """
    df   = df_feat.sort_values("time_s").reset_index(drop=True)
    peak = df[["h_max", "v_max"]].abs().max(axis=1)
    exceeded = df.loc[peak >= threshold, "time_s"]

    if len(exceeded) == 0:
        fallback = float(df["time_s"].max())
        print(f"  WARNING: {threshold} g threshold never crossed — "
              f"using last burst ({fallback:.0f} s) as failure point.")
        return fallback

    failure_s = float(exceeded.iloc[0])
    print(f"  Failure at {failure_s:.0f} s  (burst {int(failure_s / BURST_PERIOD)})")
    return failure_s


# ======================================================
# MAIN EXTRACTION FUNCTION
# ======================================================
def extract_features(vib_parquet_path: str,
                     output_csv_path: str,
                     bearing_name: str,
                     is_test: bool = False):
    """
    Extract burst-level features and label RUL entirely in SECONDS.

    TRAINING bearings (is_test=False):
      - Failure time = first burst where peak accel >= 20 g.
      - Bursts after the failure point are dropped.
      - RUL_s    = failure_time_s - time_s   (0 at failure, max at start)
      - RUL_norm = RUL_s / failure_time_s    (0.0 → 1.0)

    TEST bearings (is_test=True):
      - Recording stops BEFORE failure; failure time is unknown.
      - RUL_s    = last_time_s - time_s      (0 at last recorded burst)
      - RUL_norm = RUL_s / last_time_s
      - The model predicts additional life BEYOND the last burst.
        That prediction is compared against Table 3 only for scoring.
    """
    vib_df = pd.read_parquet(vib_parquet_path)
    print(f"\n[{bearing_name}] Loaded {vib_df.shape[0]:,} rows")

    # ── Chronological burst ordering ──────────────────────────────────────
    file_order = (
        vib_df[["file_id"]].drop_duplicates()
        .sort_values("file_id").reset_index(drop=True)
    )
    file_order["burst_idx"] = np.arange(len(file_order))
    vib_df = vib_df.merge(file_order, on="file_id", how="left")

    # ── Feature extraction ────────────────────────────────────────────────
    rows = []
    for file_id, g in vib_df.groupby("file_id", sort=True):
        h = g["Horizontal_Accel"].values
        v = g["Vertical_Accel"].values

        h_mx, h_mn, h_mean, h_sd, h_rms, h_skew, h_kurt, h_crest, h_form = burst_features(h)
        v_mx, v_mn, v_mean, v_sd, v_rms, v_skew, v_kurt, v_crest, v_form = burst_features(v)

        burst_idx = int(g["burst_idx"].iloc[0])
        time_s    = burst_idx * BURST_PERIOD   # seconds from recording start

        rows.append([
            file_id, burst_idx, time_s,
            h_mx, h_mn, h_mean, h_sd, h_rms, h_skew, h_kurt, h_crest, h_form,
            v_mx, v_mn, v_mean, v_sd, v_rms, v_skew, v_kurt, v_crest, v_form,
        ])

    cols = [
        "file_id", "burst_idx", "time_s",
        "h_max", "h_min", "h_mean", "h_sd", "h_rms", "h_skew", "h_kurt", "h_crest", "h_form",
        "v_max", "v_min", "v_mean", "v_sd", "v_rms", "v_skew", "v_kurt", "v_crest", "v_form",
    ]
    df = pd.DataFrame(rows, columns=cols).sort_values("time_s").reset_index(drop=True)

    # ── RUL labelling (all in seconds) ────────────────────────────────────
    if not is_test:
        # TRAINING: clip to failure point, RUL counts down to 0 there
        failure_s = find_failure_time_s(df)
        df        = df[df["time_s"] <= failure_s].copy().reset_index(drop=True)
        df["RUL_s"]   = (failure_s - df["time_s"]).clip(lower=0.0)
        df["RUL_norm"] = df["RUL_s"] / failure_s
        print(f"  Total life : {failure_s:.0f} s ({failure_s/3600:.2f} h) | "
              f"{len(df)} bursts kept")
    else:
        # TEST: RUL counts down to 0 at last recorded burst.
        # Model output at last burst = predicted life remaining beyond recording.
        last_s = float(df["time_s"].max())
        df["RUL_s"]    = (last_s - df["time_s"]).clip(lower=0.0)
        df["RUL_norm"] = df["RUL_s"] / last_s if last_s > 0 else 0.0
        print(f"  Recording  : {last_s:.0f} s ({last_s/3600:.2f} h) | "
              f"{len(df)} bursts | RUL_s=0 at last burst")

    print(f"  RUL_s  : {df['RUL_s'].max():.1f} s → {df['RUL_s'].min():.1f} s")
    print(f"  RUL_norm: {df['RUL_norm'].max():.4f} → {df['RUL_norm'].min():.4f}")

    df.to_csv(output_csv_path, index=False)
    print(f"  Saved → {output_csv_path}")
    return df


# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":

    BASE = r"C:\Users\29624134\Downloads\Original Data\Full_Test_Set"
    EXTRA = "Original"
    JOBS = [
        # (bearing_name, parquet_in, csv_out, is_test)

        # Training — failure detected via 20 g threshold
        #("Bearing1_1", rf"{BASE}\Bearing1_1\Big File\vibration_consolidated.parquet",
        #               rf"{BASE}\Bearing1_1\Big File\phm2012_vibration_features_{EXTRA}.csv", False),
        #("Bearing1_2", rf"{BASE}\Bearing1_2\Big File\vibration_consolidated.parquet",
        #               rf"{BASE}\Bearing1_2\Big File\phm2012_vibration_features_{EXTRA}.csv", False),
        #("Bearing2_1", rf"{BASE}\Bearing2_1\Big File\vibration_consolidated.parquet",
        #               rf"{BASE}\Bearing2_1\Big File\phm2012_vibration_features_{EXTRA}.csv", False),
        #("Bearing2_2", rf"{BASE}\Bearing2_2\Big File\vibration_consolidated.parquet",
        #               rf"{BASE}\Bearing2_2\Big File\phm2012_vibration_features_{EXTRA}.csv", False),
        #("Bearing3_1", rf"{BASE}\Bearing3_1\Big File\vibration_consolidated.parquet",
        #               rf"{BASE}\Bearing3_1\Big File\phm2012_vibration_features_{EXTRA}.csv", False),
        #("Bearing3_2", rf"{BASE}\Bearing3_2\Big File\vibration_consolidated.parquet",
        #               rf"{BASE}\Bearing3_2\Big File\phm2012_vibration_features_{EXTRA}.csv", False),

        # Test — recording truncated, RUL=0 at last burst, model predicts beyond
        ("Bearing1_3", rf"{BASE}\Bearing1_3\Big File\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing1_3\Big File\phm2012_vibration_features_{EXTRA}.csv", True),
        ("Bearing1_4", rf"{BASE}\Bearing1_4\Big File\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing1_4\Big File\phm2012_vibration_features_{EXTRA}.csv", True),
        ("Bearing1_5", rf"{BASE}\Bearing1_5\Big File\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing1_5\Big File\phm2012_vibration_features_{EXTRA}.csv", True),
        ("Bearing1_6", rf"{BASE}\Bearing1_6\Big File\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing1_6\Big File\phm2012_vibration_features_{EXTRA}.csv", True),
        ("Bearing1_7", rf"{BASE}\Bearing1_7\Big File\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing1_7\Big File\phm2012_vibration_features_{EXTRA}.csv", True),
        ("Bearing2_3", rf"{BASE}\Bearing2_3\Big File\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing2_3\Big File\phm2012_vibration_features_{EXTRA}.csv", True),
        ("Bearing2_4", rf"{BASE}\Bearing2_4\Big File\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing2_4\Big File\phm2012_vibration_features_{EXTRA}.csv", True),
        ("Bearing2_5", rf"{BASE}\Bearing2_5\Big File\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing2_5\Big File\phm2012_vibration_features_{EXTRA}.csv", True),
        ("Bearing2_6", rf"{BASE}\Bearing2_6\Big File\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing2_6\Big File\phm2012_vibration_features_{EXTRA}.csv", True),
        ("Bearing2_7", rf"{BASE}\Bearing2_7\Big File\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing2_7\Big File\phm2012_vibration_features_{EXTRA}.csv", True),
        ("Bearing3_3", rf"{BASE}\Bearing3_3\Big File\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing3_3\Big File\phm2012_vibration_features_{EXTRA}.csv", True),
    ]

    for bearing_name, parquet_in, csv_out, is_test in JOBS:
        try:
            extract_features(parquet_in, csv_out, bearing_name, is_test=is_test)
        except Exception as e:
            print(f"  ERROR [{bearing_name}]: {e}")