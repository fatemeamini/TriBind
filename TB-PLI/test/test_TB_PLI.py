import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy import stats

from proteinligandinteractionmodel_scale_modified import (
    TriBranchDTI,
    DTIDataset,
    tri_collate,
    compute_metrics,
    collect_predictions,
    load_pdbbind_index
)

# =========================
# PATHS
# =========================
BASE_PATH = "/path/to/PDBbind"
WEIGHTS_PATH = "/path/to/best_model.pth"
SCALER_PATH = "/path/to/target_scaler_params.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# SCALER
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
# MODEL
# =========================================================
def load_model(path):
    model = TriBranchDTI()
    ckpt = torch.load(path, map_location=DEVICE)

    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in ckpt:
                ckpt = ckpt[k]
                break

    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)

    model.to(DEVICE)
    model.eval()
    return model


# =========================================================
# SPLIT (exact same as train)
# =========================================================
def build_splits():

    df = load_pdbbind_index(BASE_PATH)

    df = df[df["PDB_code"].apply(
        lambda x: os.path.isdir(os.path.join(BASE_PATH, x.lower()))
    )].reset_index(drop=True)

    # test split
    test_df = df[df["Year"] > 2019].copy()
    remain_df = df[df["Year"] <= 2019].copy()

    remain_df = remain_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    val_size = int(0.15 * len(remain_df))

    val_df = remain_df.iloc[:val_size].copy()
    train_df = remain_df.iloc[val_size:].copy()

    return train_df, val_df, test_df


# =========================================================
# PLOT FUNCTION (همان test_eval)
# =========================================================
def plot_split(y_true, y_pred, name):

    x = y_true
    y = y_pred

    slope, intercept, r_value, _, slope_stderr = stats.linregress(x, y)

    xbar = x.mean()
    Sxx = np.sum((x - xbar) ** 2)
    yhat = intercept + slope * x

    SSE = np.sum((y - yhat) ** 2)
    n = len(x)
    dfree = n - 2
    s = np.sqrt(SSE / dfree)

    xg = np.linspace(x.min(), x.max(), 300)
    yg = intercept + slope * xg

    t95 = stats.t.ppf(0.975, dfree)

    se_mean = s * np.sqrt(1.0 / n + (xg - xbar) ** 2 / Sxx)
    se_pred = s * np.sqrt(1.0 + 1.0 / n + (xg - xbar) ** 2 / Sxx)

    ci_low, ci_high = yg - t95 * se_mean, yg + t95 * se_mean
    pi_low, pi_high = yg - t95 * se_pred, yg + t95 * se_pred

    plt.figure(figsize=(7, 7))

    plt.scatter(x, y, alpha=0.6)
    plt.fill_between(xg, pi_low, pi_high, alpha=0.2)
    plt.fill_between(xg, ci_low, ci_high, alpha=0.25)

    plt.plot(xg, yg)
    plt.plot(xg, xg, linestyle="--")

    r2 = r_value ** 2

    plt.title(f"{name} | R²={r2:.3f}")
    plt.xlabel("True")
    plt.ylabel("Pred")

    plt.tight_layout()
    plt.savefig(f"{name}_pred_vs_true.png", dpi=300)
    plt.show()


# =========================================================
# EVAL FUNCTION
# =========================================================
def evaluate(df, name, model, mean, scale_):

    df = df.copy()

    # scale target
    df["Activity"] = scale(df["Activity"].values, mean, scale_)

    loader = DataLoader(
        DTIDataset(df, BASE_PATH),
        batch_size=32,
        shuffle=False,
        collate_fn=tri_collate
    )

    y_true, y_pred = collect_predictions(loader, model, DEVICE)

    if y_true is None:
        print(f"{name}: no valid samples")
        return

    # inverse scale
    y_true = inverse(y_true, mean, scale_)
    y_pred = inverse(y_pred, mean, scale_)

    m = compute_metrics(y_true, y_pred)

    print(f"\n===== {name} =====")
    for k, v in m.items():
        print(f"{k:10s}: {v:.4f}")

    plot_split(y_true, y_pred, name)


# =========================================================
# MAIN
# =========================================================
def main():

    mean, scale_ = load_scaler(SCALER_PATH)

    train_df, val_df, test_df = build_splits()

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    model = load_model(WEIGHTS_PATH)

    # ✅ بدون sampler (خیلی مهم)
    evaluate(train_df, "Train", model, mean, scale_)
    evaluate(val_df, "Validation", model, mean, scale_)
    evaluate(test_df, "Test", model, mean, scale_)


if __name__ == "__main__":
    main()
