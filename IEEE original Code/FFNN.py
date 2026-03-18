# =========================
# Multi-Bearing Multi-Step RUL Predictor
# Target: RUL_s (seconds) — scaled for training
# =========================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================
TARGET       = "RUL_s"
WINDOW_SIZE  = 40
HORIZON      = 10
BURST_PERIOD = 10.0
BATCH_SIZE   = 32
EPOCHS       = 600
LR           = 1e-5
HIDDEN_UNITS = [128, 64]
DROPOUT      = 0.25
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
PATIENCE     = 30
RUL_SCALE   = 30000.0      # scale factor for RUL

# GT RULs from Table 3 — used ONLY for post-hoc scoring
ACTUAL_RUL_S = {
    "Bearing1_3": 5730,
    "Bearing1_4":  339,
    "Bearing1_5": 1610,
    "Bearing1_6": 1460,
    "Bearing1_7": 7570,
    "Bearing2_3": 7530,
    "Bearing2_4": 1390,
    "Bearing2_5": 3090,
    "Bearing2_6": 1290,
    "Bearing2_7":  580,
    "Bearing3_3":  820,
}

torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================
# DATASET
# =========================
class RULDataset(Dataset):
    def __init__(self, X, y, horizon):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.horizon + 1

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx:idx + self.horizon]

# =========================
# MODEL
# =========================
class RULNet(nn.Module):
    def __init__(self, input_dim, horizon):
        super().__init__()
        layers = []
        prev = input_dim
        for h in HIDDEN_UNITS:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(DROPOUT)]
            prev = h
        layers.append(nn.Linear(prev, horizon))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =========================
# FEATURE ENGINEERING
# =========================
DROP_COLS = ["file_id", "burst_idx", "time_s", TARGET]

def add_rolling_features(df, window):
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    new_cols = {}
    for col in feature_cols:
        new_cols[f"{col}_mean"]  = df[col].rolling(window).mean()
        new_cols[f"{col}_std"]   = df[col].rolling(window).std()
        new_cols[f"{col}_slope"] = df[col].rolling(window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
        )
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df.dropna().reset_index(drop=True)

def load_bearing(csv_path):
    df = pd.read_csv(csv_path)
    if TARGET not in df.columns:
        raise ValueError(f"{TARGET} not found in {csv_path}")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    if len(df) < WINDOW_SIZE:
        return None
    df = add_rolling_features(df, WINDOW_SIZE)
    return df if len(df) > 0 else None

def get_xy(df):
    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop).values
    y = df[TARGET].values
    return X, y

# =========================
# PREPARE TRAIN / VAL
# =========================
def prepare_train_val(train_files, val_files):
    # ── Training bearings
    train_dfs = [df for f in train_files if (df := load_bearing(f)) is not None]
    if not train_dfs:
        raise RuntimeError("No training data could be loaded.")
    train_combined = pd.concat(train_dfs, axis=0).reset_index(drop=True)
    X_train, y_train = get_xy(train_combined)
    y_train = y_train / RUL_SCALE    # scale RUL for training

    # ── Validation bearings
    val_dfs = [df for f in val_files if (df := load_bearing(f)) is not None]
    if not val_dfs:
        raise RuntimeError("No validation data could be loaded.")
    val_combined = pd.concat(val_dfs, axis=0).reset_index(drop=True)
    X_val, y_val = get_xy(val_combined)

    # ── Scale features only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    print(f"Train: {X_train.shape}  Val: {X_val.shape}")
    print(f"RUL_s train range (scaled): {y_train.min():.4f} → {y_train.max():.4f}")
    print(f"RUL_s val range: {y_val.min():.0f} s → {y_val.max():.0f} s")
    return X_train, y_train, X_val, y_val, scaler

def load_test_bearing(csv_path, scaler):
    df = load_bearing(csv_path)
    if df is None:
        return None, None
    X, y = get_xy(df)
    X = scaler.transform(X)
    return X, y

# =========================
# EVALUATION
# =========================
def evaluate(model, X, y, horizon):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(X) - horizon + 1):
            x_in = torch.tensor(X[i:i+1], dtype=torch.float32).to(DEVICE)
            preds.append(model(x_in).cpu().numpy().reshape(-1))
    preds  = np.array(preds)
    y_true = np.array([y[i:i+horizon] for i in range(len(y)-horizon+1)])
    mae  = mean_absolute_error(y_true.flatten(), preds.flatten()*RUL_SCALE)
    rmse = np.sqrt(mean_squared_error(y_true.flatten(), preds.flatten()*RUL_SCALE))
    return mae, rmse, preds, y_true

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    TRAIN_FILES = [
        r"C:\Users\29624134\Downloads\Original Data\Bearing1_1\Big File\phm2012_vibration_features_Original.csv",
        r"C:\Users\29624134\Downloads\Original Data\Bearing1_2\Big File\phm2012_vibration_features_Original.csv",
        r"C:\Users\29624134\Downloads\Original Data\Bearing2_1\Big File\phm2012_vibration_features_Original.csv",
        r"C:\Users\29624134\Downloads\Original Data\Bearing2_2\Big File\phm2012_vibration_features_Original.csv",
        r"C:\Users\29624134\Downloads\Original Data\Bearing3_1\Big File\phm2012_vibration_features_Original.csv",
        r"C:\Users\29624134\Downloads\Original Data\Bearing3_2\Big File\phm2012_vibration_features_Original.csv"
    ]

    VAL_FILES = [
        r"C:\Users\29624134\Downloads\Original Data\Full_Test_Set\Bearing1_3\Big File\phm2012_vibration_features_Original.csv",
        r"C:\Users\29624134\Downloads\Original Data\Full_Test_Set\Bearing2_5\Big File\phm2012_vibration_features_Original.csv"
    ]

    TEST_FILES = [
        r"C:\Users\29624134\Downloads\Original Data\Bearing1_3\Big File\phm2012_vibration_features_Original.csv",
        r"C:\Users\29624134\Downloads\Original Data\Bearing1_4\Big File\phm2012_vibration_features_Original.csv",
        r"C:\Users\29624134\Downloads\Original Data\Bearing1_5\Big File\phm2012_vibration_features_Original.csv",
        r"C:\Users\29624134\Downloads\Original Data\Bearing1_6\Big File\phm2012_vibration_features_Original.csv",
        r"C:\Users\29624134\Downloads\Original Data\Bearing1_7\Big File\phm2012_vibration_features_Original.csv",
        r"C:\Users\29624134\Downloads\Original Data\Bearing2_3\Big File\phm2012_vibration_features_Original.csv",
        r"C:\Users\29624134\Downloads\Original Data\Bearing2_4\Big File\phm2012_vibration_features_Original.csv",
        r"C:\Users\29624134\Downloads\Original Data\Bearing2_5\Big File\phm2012_vibration_features_Original.csv",
        r"C:\Users\29624134\Downloads\Original Data\Bearing2_6\Big File\phm2012_vibration_features_Original.csv",
        r"C:\Users\29624134\Downloads\Original Data\Bearing2_7\Big File\phm2012_vibration_features_Original.csv",
        r"C:\Users\29624134\Downloads\Original Data\Bearing3_3\Big File\phm2012_vibration_features_Original.csv",
    ]

    # ── Train
    X_train, y_train, X_val, y_val, scaler = prepare_train_val(TRAIN_FILES, VAL_FILES)
    train_ds = RULDataset(X_train, y_train, horizon=HORIZON)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = RULNet(X_train.shape[1], horizon=HORIZON).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    #loss_fn = nn.MSELoss()
    loss_fn = nn.SmoothL1Loss(beta=0.1)

    best_val_mae = float("inf")
    best_state = None
    epochs_no_improve = 0

    for e in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(Xb)

        val_mae, _, _, _ = evaluate(model, X_val, y_val, HORIZON)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if e % 20 == 0 or e == 1:
            print(f"Epoch {e:>3}/{EPOCHS} | "
                  f"Train Loss: {total_loss/len(train_loader.dataset):.5f} | "
                  f"Val MAE: {val_mae:.5f} s")

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {e}")
            break

    model.load_state_dict(best_state)

    # ── Test ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  PER-BEARING RESULTS  —  prediction at last recorded burst")
    print("  Compared against PHM 2012 Table 3  (scoring only, not used in training)")
    print("=" * 72)
    print(f"  {'Bearing':<12} {'GT RUL':>9} {'Pred RUL':>10} {'Error':>9} {'Abs%Err':>9}")
    print(f"  {'-'*12} {'-'*9} {'-'*10} {'-'*9} {'-'*9}")

    summary = []

    for test_path in TEST_FILES:
        bearing_name = test_path.split("\\")[-3]
        X_test, y_test = load_test_bearing(test_path, scaler)

        if X_test is None:
            print(f"  [{bearing_name}]  Could not load — skipping.")
            continue

        mae, rmse, y_pred, y_true = evaluate(model, X_test, y_test, HORIZON)

        pred_rul_s = float(np.clip(y_pred[-1, 0]*RUL_SCALE, 0, None))
        gt_rul_s = ACTUAL_RUL_S.get(bearing_name)
        if gt_rul_s is None:
            print(f"  {bearing_name:<12}  no GT entry")
            continue

        error_s = pred_rul_s - gt_rul_s
        abs_pct = abs(error_s) / gt_rul_s * 100

        print(f"  {bearing_name:<12} {gt_rul_s:>7.0f} s {pred_rul_s:>8.0f} s "
              f"{error_s:>+8.0f} s {abs_pct:>8.1f}%")

        summary.append(dict(
            bearing=bearing_name,
            gt_rul_s=gt_rul_s,
            pred_rul_s=pred_rul_s,
            error_s=error_s,
            abs_pct=abs_pct,
            y_pred=y_pred,
            y_true=y_true,
        ))

    if summary:
        mean_err = np.mean([abs(r["error_s"]) for r in summary])
        mean_pct = np.mean([r["abs_pct"] for r in summary])
        print(f"  {'-'*72}")
        print(f"  {'Mean':<12} {'':>9} {'':>10} {mean_err:>+8.0f} s {mean_pct:>8.1f}%")
    print("=" * 72)

# ── Plots ─────────────────────────────────────────────────────────────
    for r in summary:
        bearing_name = r["bearing"]
        y_pred       = r["y_pred"]
        y_true       = r["y_true"]
        gt_rul_s     = r["gt_rul_s"]
        pred_rul_s   = r["pred_rul_s"]

        time_axis = np.arange(len(y_pred)) * BURST_PERIOD

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle(
            f"{bearing_name}  |  GT: {gt_rul_s} s ({gt_rul_s/60:.1f} min)  |  "
            f"Pred: {pred_rul_s:.0f} s ({pred_rul_s/60:.1f} min)",
            fontsize=11
        )

        axes[0].plot(time_axis, y_true[:, 0], label="True RUL_s (in-recording)", linewidth=2)
        axes[0].plot(time_axis, np.clip(y_pred[:, 0]*RUL_SCALE, 0, None),
                     label="Predicted RUL_s", linewidth=2, linestyle="--")
        axes[0].axhline(gt_rul_s, color="red", linestyle=":", linewidth=1.5,
                        label=f"Table 3 GT = {gt_rul_s} s")
        axes[0].set_xlabel("Recording time (s)")
        axes[0].set_ylabel("RUL (seconds)")
        axes[0].set_title("RUL trajectory over recording")
        axes[0].legend(fontsize=8)
        axes[0].grid(True)

        axes[1].bar(["GT RUL", "Predicted"], [gt_rul_s, pred_rul_s],
                    color=["steelblue", "coral"], edgecolor="black", width=0.5)
        y_top = max(gt_rul_s, pred_rul_s) * 1.15
        axes[1].set_ylim(0, y_top)
        for i, v in enumerate([gt_rul_s, pred_rul_s]):
            axes[1].text(i, v + y_top * 0.02,
                         f"{v:.0f} s\n({v/60:.1f} min)", ha="center", fontsize=10)
        axes[1].set_ylabel("Seconds")
        axes[1].set_title("RUL at truncation point vs GT")
        axes[1].grid(axis="y")

        plt.tight_layout()
        plt.show()