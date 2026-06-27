import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy import stats

from ProteinLigandUnifiedGraph import (
    UnifiedGraphDTI,
    DTIDataset,
    unified_collate,
    compute_metrics,
    collect_predictions,
)

# =========================
# Paths (EDIT THESE)
# =========================
CSV_PATH     = "/Volumes/Ventoy/MolGen_Project copy/test2016.csv"
WEIGHTS_PATH = "/Volumes/Ventoy/MolGen_Project copy/mol gen project/May30/best_model_unified.pth"
SCALER_PATH  = "/Volumes/Ventoy/MolGen_Project copy/mol gen project/May30/target_scaler_params_unified.json"
DATA_ROOT    = "/Volumes/Ventoy/MolGen_Project copy/pdbbind2020_core_set_285/pdbbind2020_core_set"

# =========================
# Config
# =========================
BATCH_SIZE  = 8
NUM_WORKERS = 4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# Scaler  — loads mean/scale saved during training and inverts
# ================================================================
class SavedStandardScaler:
    """
    Reconstructs the StandardScaler fitted on the training set from
    the JSON file saved by the training script.
    Used to inverse-transform predictions and true values back to
    the original pKi scale before computing metrics.
    """
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            params = json.load(f)
        self.mean_  = float(params["mean"])
        self.scale_ = float(params["scale"])
        print(f"✅ Loaded scaler  — mean: {self.mean_:.6f}  scale: {self.scale_:.6f}")

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.scale_ + self.mean_


# ================================================================
# Helpers
# ================================================================
def _load_state_dict_flexible(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("Unsupported checkpoint format.")


def load_model(weights_path: str, device: torch.device) -> UnifiedGraphDTI:
    model = UnifiedGraphDTI()

    ckpt       = torch.load(weights_path, map_location=device)
    state_dict = _load_state_dict_flexible(ckpt)

    cleaned = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()

    print(f"✅ Loaded weights  : {weights_path}")
    print(f"✅ Device          : {device}")
    if missing:
        print(f"⚠️  Missing keys    : {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"⚠️  Unexpected keys : {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    return model


def load_test_dataframe(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    if "PDB_code" in raw.columns and "Activity" in raw.columns:
        df = raw[["PDB_code", "Activity"]].copy()
    else:
        df = pd.DataFrame({
            "PDB_code": raw.iloc[:, 0].astype(str),
            "Activity": raw.iloc[:, 1].astype(float),
        })
    df["PDB_code"] = df["PDB_code"].astype(str)
    df["Activity"] = df["Activity"].astype(float)
    return df


# ================================================================
# Main
# ================================================================
def main():
    # ── Dataset / Loader ──────────────────────────────────────────
    df = load_test_dataframe(CSV_PATH)
    print(f"Test set size: {len(df)} complexes")

    # NOTE: DTIDataset is used here with raw (unscaled) Activity values.
    # The model outputs scaled predictions, so we inverse-transform
    # BOTH y_pred and y_true after collection.
    test_ds = DTIDataset(df=df, base_path=DATA_ROOT)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=unified_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # ── Load model + scaler ───────────────────────────────────────
    model  = load_model(WEIGHTS_PATH, DEVICE)
    scaler = SavedStandardScaler(SCALER_PATH)

    # ── Collect raw (scaled) predictions ─────────────────────────
    # collect_predictions returns whatever the model outputs and
    # whatever y values DTIDataset supplies — both are in scaled space
    # because DTIDataset feeds Activity as-is and the model was trained
    # on scaled targets.
    y_true_scaled, y_pred_scaled = collect_predictions(test_loader, model, DEVICE)
    if y_true_scaled is None or y_pred_scaled is None:
        raise RuntimeError(
            "No predictions collected. "
            "Check DATA_ROOT structure / parsing errors."
        )

    # ── Inverse-transform to original pKi scale ──────────────────
    # y_true came from the CSV (raw pKi), passed through DTIDataset
    # unchanged, so it is already on the original scale.
    # y_pred is in scaled space (model was trained on scaled targets),
    # so only y_pred needs inverse_transform.
    y_true = y_true_scaled                              # raw pKi from CSV
    y_pred = scaler.inverse_transform(y_pred_scaled)   # scaled → pKi

    # ── Metrics on original pKi scale ────────────────────────────
    m         = compute_metrics(y_true, y_pred)
    pearson_r = m["Pearsonr"]
    r_sq      = float(pearson_r ** 2) if np.isfinite(pearson_r) else float("nan")

    print("\n===== TEST SET METRICS (Unified Graph — original pKi scale) =====")
    print(f"RMSE       : {m['RMSE']:.4f}")
    print(f"MAE        : {m['MAE']:.4f}")
    print(f"CI         : {m['CI']:.4f}")
    print(f"SD         : {m['SD']:.4f}")
    print(f"Pearsonr   : {m['Pearsonr']:.4f}")
    print(f"R²         : {r_sq:.4f}")

    # ── Plot: Pred vs True with regression + CI/PI ────────────────
    x = np.asarray(y_true).ravel()
    y = np.asarray(y_pred).ravel()
    n = len(x)
    if n < 3:
        raise RuntimeError("Need at least 3 points for regression intervals.")

    slope, intercept, r_value, p_value, slope_stderr = stats.linregress(x, y)

    xbar = x.mean()
    Sxx  = np.sum((x - xbar) ** 2)
    yhat = intercept + slope * x
    SSE  = np.sum((y - yhat) ** 2)
    dfree = n - 2
    s     = np.sqrt(SSE / dfree)
    intercept_stderr = s * np.sqrt(1.0 / n + (xbar ** 2) / Sxx)

    xg = np.linspace(x.min(), x.max(), 300)
    yg = intercept + slope * xg

    t95 = stats.t.ppf(0.975, dfree)
    t99 = stats.t.ppf(0.995, dfree)

    se_mean = s * np.sqrt(1.0 / n + (xg - xbar) ** 2 / Sxx)
    se_pred = s * np.sqrt(1.0 + 1.0 / n + (xg - xbar) ** 2 / Sxx)

    ci95_low,  ci95_high  = yg - t95 * se_mean, yg + t95 * se_mean
    ci99_low,  ci99_high  = yg - t99 * se_mean, yg + t99 * se_mean
    pi95_low,  pi95_high  = yg - t95 * se_pred, yg + t95 * se_pred
    pi99_low,  pi99_high  = yg - t99 * se_pred, yg + t99 * se_pred

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=35, alpha=0.7, label="Test Data")

    plt.fill_between(xg, pi99_low, pi99_high, alpha=0.18, label="99% Prediction Interval")
    plt.fill_between(xg, pi95_low, pi95_high, alpha=0.20, label="95% Prediction Interval")
    plt.fill_between(xg, ci99_low, ci99_high, alpha=0.25, label="99% CI (Regression Line)")
    plt.fill_between(xg, ci95_low, ci95_high, alpha=0.25, label="95% CI (Regression Line)")

    plt.plot(xg, yg, linewidth=3,             label="Best-fit line")
    plt.plot(xg, xg, linestyle="--", alpha=0.35)   # identity line

    plt.xlabel("True pKi",      fontsize=14)
    plt.ylabel("Predicted pKi", fontsize=14)
    plt.title("Unified Graph Model — External Test", fontsize=13)

    plt.text(
        0.05, 0.95,
        f"y = {intercept:.3f} (±{intercept_stderr:.3f}) "
        f"+ {slope:.3f} (±{slope_stderr:.3f}) × x\n"
        f"R² = {r_sq:.3f}",
        transform=plt.gca().transAxes,
        fontsize=11, va="top",
    )

    plt.legend(frameon=True)
    plt.tight_layout()

    out_png = os.path.join(os.path.dirname(WEIGHTS_PATH),
                           "test_pred_vs_true_unified.png")
    plt.savefig(out_png, dpi=600)
    print(f"\n✅ Saved plot to: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()