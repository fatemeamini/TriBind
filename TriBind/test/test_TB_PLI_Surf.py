import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy import stats

# ✅ IMPORT از مدل جدید
from proteinligandinteractionmodel_scale_modified import (
    TriBranchDTI,
    DTIDataset,
    tri_collate,
    compute_metrics,
    collect_predictions,
    load_pdbbind_index
)

# =========================
# PATHS (EDIT)
# =========================
BASE_PATH = "/path/to/PDBbind"
WEIGHTS_PATH = "/path/to/best_model.pth"
SCALER_PATH = "/path/to/target_scaler_params.json"

BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# SCALER (IDENTICAL TO TRAIN)
# =========================================================
def load_scaler(path):
    with open(path, "r") as f:
        p = json.load(f)
    return p["mean"], p["scale"]


def scale(y, mean, scale):
    return (y - mean) / scale


def inverse(y, mean, scale):
    return y * scale + mean


# =========================================================
# LOAD MODEL (IDENTICAL LOGIC)
# =========================================================
def _load_state_dict_flexible(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    return ckpt_obj


def load_model(weights_path, device):
    model = TriBranchDTI()

    ckpt = torch.load(weights_path, map_location=device)
    state_dict = _load_state_dict_flexible(ckpt)

    cleaned = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()

    print(f"✅ Loaded weights: {weights_path}")
    return model


# =========================================================
# BUILD SPLIT (IDENTICAL TO TRAIN SCRIPT)
# =========================================================
def build_splits():

    df = load_pdbbind_index(BASE_PATH)

    df = df[df["PDB_code"].apply(
        lambda x: os.path.isdir(os.path.join(BASE_PATH, x.lower()))
    )].reset_index(drop=True)

    test_df = df[df["Year"] > 2019].copy()
    remain_df = df[df["Year"] <= 2019].copy()

    remain_df = remain_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    val_size = int(0.15 * len(remain_df))

    val_df = remain_df.iloc[:val_size].copy()
    train_df = remain_df.iloc[val_size:].copy()

    return train_df, val_df, test_df


# =========================================================
# PLOT (EXACT COPY FROM test_eval_TriModel)
# =========================================================
def plot_pred_vs_true(y_true, y_pred, name):

    x = np.asarray(y_true).ravel()
    y = np.asarray(y_pred).ravel()

    slope, intercept, r_value, p_value, slope_stderr = stats.linregress(x, y)

    xbar = x.mean()
    Sxx = np.sum((x - xbar) ** 2)
    yhat = intercept + slope * x

    SSE = np.sum((y - yhat) ** 2)
    n = len(x)
    dfree = n - 2
    s = np.sqrt(SSE / dfree)

    intercept_stderr = s * np.sqrt(1.0 / n + (xbar ** 2) / Sxx)

    xg = np.linspace(x.min(), x.max(), 300)
    yg = intercept + slope * xg

    t95 = stats.t.ppf(0.975, dfree)
    t99 = stats.t.ppf(0.995, dfree)

    se_mean = s * np.sqrt(1.0 / n + (xg - xbar) ** 2 / Sxx)
    se_pred = s * np.sqrt(1.0 + 1.0 / n + (xg - xbar) ** 2 / Sxx)

    ci95_low, ci95_high = yg - t95 * se_mean, yg + t95 * se_mean
    ci99_low, ci99_high = yg - t99 * se_mean, yg + t99 * se_mean

    pi95_low, pi95_high = yg - t95 * se_pred, yg + t95 * se_pred
    pi99_low, pi99_high = yg - t99 * se_pred, yg + t99 * se_pred

    plt.figure(figsize=(7, 7))

    plt.scatter(x, y, s=35, alpha=0.7)

    plt.fill_between(xg, pi99_low, pi99_high, alpha=0.18)
    plt.fill_between(xg, pi95_low, pi95_high, alpha=0.20)

    plt.fill_between(xg, ci99_low, ci99_high, alpha=0.25)
    plt.fill_between(xg, ci95_low, ci95_high, alpha=0.25)

    plt.plot(xg, yg, linewidth=3)
    plt.plot(xg, xg, linestyle="--", alpha=0.35)

    plt.xlabel("True pKi")
    plt.ylabel("Predicted pKi")

    r2 = r_value ** 2

    plt.text(
        0.05,
        0.95,
        f"y = {intercept:.3f} + {slope:.3f}x\nR² = {r2:.3f}",
        transform=plt.gca().transAxes,
        fontsize=11,
        va="top",
    )

    plt.tight_layout()
    plt.savefig(f"{name}_pred_vs_true.png", dpi=300)
    plt.show()

    print(f"✅ Saved plot: {name}_pred_vs_true.png")


# =========================================================
# EVALUATION (CORE)
# =========================================================
def evaluate(df, name, model, mean, scale_):

    df = df.copy()

    # 🔥 scaling EXACTLY like train
    df["Activity"] = scale(df["Activity"].values, mean, scale_)

    loader = DataLoader(
        DTIDataset(df, BASE_PATH),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=tri_collate,
    )

    y_true, y_pred = collect_predictions(loader, model, DEVICE)

    if y_true is None:
        raise RuntimeError(f"{name}: no predictions collected")

    # 🔥 inverse scaling
    y_true = inverse(y_true, mean, scale_)
    y_pred = inverse(y_pred, mean, scale_)

    m = compute_metrics(y_true, y_pred)

    print(f"\n===== {name} METRICS =====")
    print(f"RMSE       : {m['RMSE']:.4f}")
    print(f"MAE        : {m['MAE']:.4f}")
    print(f"CI         : {m['CI']:.4f}")
    print(f"SD         : {m['SD']:.4f}")
    print(f"Pearsonr   : {m['Pearsonr']:.4f}")
    print(f"R predicted: {m['Pearsonr']**2:.4f}")

    plot_pred_vs_true(y_true, y_pred, name)


# =========================================================
# MAIN
# =========================================================
def main():

    mean, scale_ = load_scaler(SCALER_PATH)

    train_df, val_df, test_df = build_splits()

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    model = load_model(WEIGHTS_PATH, DEVICE)

    evaluate(train_df, "Train", model, mean, scale_)
    evaluate(val_df, "Validation", model, mean, scale_)
    evaluate(test_df, "Test", model, mean, scale_)


if __name__ == "__main__":
    main()
