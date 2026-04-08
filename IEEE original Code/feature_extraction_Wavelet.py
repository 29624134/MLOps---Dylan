import numpy as np
import pandas as pd
import pywt

# ======================================================
# CONFIG
# ======================================================
BURST_PERIOD      = 10.0   # seconds per burst
FAILURE_THRESHOLD = 20.0   # g — PHM 2012 failure criterion (peak acceleration)
WAVELET           = "db4"  # wavelet family for packet decomposition
WP_LEVEL          = 3      # decomposition level → 2^3 = 8 frequency bands


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


def wavelet_packet_features(sig: np.ndarray, wavelet: str = WAVELET, level: int = WP_LEVEL):
    """
    Compute wavelet packet energy and Shannon entropy for each of the
    2^level frequency bands.

    Returns two lists of length 2^level:
        energies  : sum of squared coefficients per band (proxy for power)
        entropies : Shannon entropy of normalised squared coefficients per band
    """
    wp = pywt.WaveletPacket(data=sig, wavelet=wavelet, mode="symmetric", maxlevel=level)
    nodes = [node.path for node in wp.get_level(level, "freq")]

    energies  = []
    entropies = []
    for node in nodes:
        coeffs = wp[node].data
        energy = float(np.sum(coeffs ** 2))
        energies.append(energy)

        # Shannon entropy — normalise coefficients first to get a probability distribution
        total = energy if energy > 0 else 1.0
        p     = (coeffs ** 2) / total
        p     = p[p > 0]   # avoid log(0)
        entropy = float(-np.sum(p * np.log(p)))
        entropies.append(entropy)

    return energies, entropies


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

    Features per burst (50 total):
        9  horizontal time-domain stats  (h_max … h_form)
        9  vertical   time-domain stats  (v_max … v_form)
        8  horizontal wavelet packet energies   (h_wp_energy_0 … 7)
        8  vertical   wavelet packet energies   (v_wp_energy_0 … 7)
        8  horizontal wavelet packet entropies  (h_wp_entropy_0 … 7)
        8  vertical   wavelet packet entropies  (v_wp_entropy_0 … 7)

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

    n_bands = 2 ** WP_LEVEL  # 8 frequency bands

    # ── Feature extraction ────────────────────────────────────────────────
    rows = []
    for file_id, g in vib_df.groupby("file_id", sort=True):
        h = g["Horizontal_Accel"].values
        v = g["Vertical_Accel"].values

        # Time-domain features
        h_mx, h_mn, h_mean, h_sd, h_rms, h_skew, h_kurt, h_crest, h_form = burst_features(h)
        v_mx, v_mn, v_mean, v_sd, v_rms, v_skew, v_kurt, v_crest, v_form = burst_features(v)

        # Wavelet packet features
        h_energies, h_entropies = wavelet_packet_features(h)
        v_energies, v_entropies = wavelet_packet_features(v)

        burst_idx = int(g["burst_idx"].iloc[0])
        time_s    = burst_idx * BURST_PERIOD

        row = [
            file_id, burst_idx, time_s,
            # Time-domain
            h_mx, h_mn, h_mean, h_sd, h_rms, h_skew, h_kurt, h_crest, h_form,
            v_mx, v_mn, v_mean, v_sd, v_rms, v_skew, v_kurt, v_crest, v_form,
            # Wavelet packet energies
            *h_energies,   # h_wp_energy_0 … h_wp_energy_7
            *v_energies,   # v_wp_energy_0 … v_wp_energy_7
            # Wavelet packet entropies
            *h_entropies,  # h_wp_entropy_0 … h_wp_entropy_7
            *v_entropies,  # v_wp_entropy_0 … v_wp_entropy_7
        ]
        rows.append(row)

    # ── Build column names ────────────────────────────────────────────────
    cols = [
        "file_id", "burst_idx", "time_s",
        "h_max", "h_min", "h_mean", "h_sd", "h_rms", "h_skew", "h_kurt", "h_crest", "h_form",
        "v_max", "v_min", "v_mean", "v_sd", "v_rms", "v_skew", "v_kurt", "v_crest", "v_form",
        *[f"h_wp_energy_{i}"  for i in range(n_bands)],
        *[f"v_wp_energy_{i}"  for i in range(n_bands)],
        *[f"h_wp_entropy_{i}" for i in range(n_bands)],
        *[f"v_wp_entropy_{i}" for i in range(n_bands)],
    ]
    df = pd.DataFrame(rows, columns=cols).sort_values("time_s").reset_index(drop=True)

    # ── RUL labelling (all in seconds) ────────────────────────────────────
    if not is_test:
        failure_s = find_failure_time_s(df)
        df        = df[df["time_s"] <= failure_s].copy().reset_index(drop=True)
        df["RUL_s"]    = (failure_s - df["time_s"]).clip(lower=0.0)
        df["RUL_norm"] = df["RUL_s"] / failure_s
        print(f"  Total life : {failure_s:.0f} s ({failure_s/3600:.2f} h) | "
              f"{len(df)} bursts kept")
    else:
        last_s = float(df["time_s"].max())
        df["RUL_s"]    = (last_s - df["time_s"]).clip(lower=0.0)
        df["RUL_norm"] = df["RUL_s"] / last_s if last_s > 0 else 0.0
        print(f"  Recording  : {last_s:.0f} s ({last_s/3600:.2f} h) | "
              f"{len(df)} bursts | RUL_s=0 at last burst")

    print(f"  Features   : {len(cols) - 3} input cols + RUL_s + RUL_norm")
    print(f"  RUL_s  : {df['RUL_s'].max():.1f} s → {df['RUL_s'].min():.1f} s")
    print(f"  RUL_norm: {df['RUL_norm'].max():.4f} → {df['RUL_norm'].min():.4f}")

    df.to_csv(output_csv_path, index=False)
    print(f"  Saved → {output_csv_path}")
    return df


# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":

    BASE  = r"D:\Data + Models\Data\ieee-phm-2012-data-challenge-dataset-master\All_Test_Sets_wavelet"

    JOBS = [
        # (bearing_name, parquet_in, csv_out, is_test)

        # Training — failure detected via 20 g threshold
        ("Bearing1_1", rf"{BASE}\Bearing1_1\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing1_1\features.csv", False),
        ("Bearing1_2", rf"{BASE}\Bearing1_2\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing1_2\features.csv", False),
        ("Bearing2_1", rf"{BASE}\Bearing2_1\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing2_1\features.csv", False),
        ("Bearing2_2", rf"{BASE}\Bearing2_2\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing2_2\features.csv", False),
        ("Bearing3_1", rf"{BASE}\Bearing3_1\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing3_1\features.csv", False),
        ("Bearing3_2", rf"{BASE}\Bearing3_2\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing3_2\features.csv", False),

        # Test — recording truncated, RUL=0 at last burst, model predicts beyond
        ("Bearing1_3", rf"{BASE}\Bearing1_3\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing1_3\features.csv", True),
        ("Bearing1_4", rf"{BASE}\Bearing1_4\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing1_4\features.csv", True),
        ("Bearing1_5", rf"{BASE}\Bearing1_5\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing1_5\features.csv", True),
        ("Bearing1_6", rf"{BASE}\Bearing1_6\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing1_6\features.csv", True),
        ("Bearing1_7", rf"{BASE}\Bearing1_7\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing1_7\features.csv", True),
        ("Bearing2_3", rf"{BASE}\Bearing2_3\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing2_3\features.csv", True),
        ("Bearing2_4", rf"{BASE}\Bearing2_4\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing2_4\features.csv", True),
        ("Bearing2_5", rf"{BASE}\Bearing2_5\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing2_5\features.csv", True),
        ("Bearing2_6", rf"{BASE}\Bearing2_6\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing2_6\features.csv", True),
        ("Bearing2_7", rf"{BASE}\Bearing2_7\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing2_7\features.csv", True),
        ("Bearing3_3", rf"{BASE}\Bearing3_3\vibration_consolidated.parquet",
                       rf"{BASE}\Bearing3_3\features.csv", True),
    ]

    for bearing_name, parquet_in, csv_out, is_test in JOBS:
        try:
            extract_features(parquet_in, csv_out, bearing_name, is_test=is_test)
        except Exception as e:
            print(f"  ERROR [{bearing_name}]: {e}")