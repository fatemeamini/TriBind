# %%
import os
import numpy as np
import pandas as pd

import torch
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

from GIGN import GIGN

# =========================================================
# PATHS
# =========================================================

GRAPH_ROOT = (
    "/Volumes/Ventoy/MolGen_Project copy/"
    "data/dataset/selected_2016"
)

CSV_PATH = (
    "/Volumes/Ventoy/MolGen_Project copy/test2016.csv"
)

WEIGHTS_PATH = (
    "/Users/fatemeh/GIGN/GIGN/Backup/best_GIGN_surface.pt"
)

# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)

# =========================================================
# LOAD CSV
# =========================================================

df = pd.read_csv(CSV_PATH)

# =========================================================
# LOAD TEST GRAPHS
# =========================================================

graphs = []

missing = []

for _, row in df.iterrows():

    cid = row['pdbid']

    graph_path = os.path.join(
        GRAPH_ROOT,
        cid,
        f"Graph_{cid}_5A.pyg"
    )

    if not os.path.exists(graph_path):

        missing.append(cid)
        continue

    try:

        data = torch.load(graph_path)

        graphs.append(data)

    except Exception as e:

        print(f"[ERROR] {cid}: {e}")

print(f"\nLoaded graphs: {len(graphs)}")

if len(missing) > 0:

    print("\nMissing graphs:")

    for x in missing:
        print(x)

# =========================================================
# DATALOADER
# =========================================================

loader = DataLoader(
    graphs,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# =========================================================
# LOAD MODEL
# =========================================================

model = GIGN(
    node_dim=42,
    hidden_dim=256
).to(device)

model.load_state_dict(
    torch.load(
        WEIGHTS_PATH,
        map_location=device
    )
)

model.eval()

print("\nModel loaded successfully.")

# =========================================================
# PREDICTION
# =========================================================

preds = []
targets = []

with torch.no_grad():

    for data in loader:

        data = data.to(device)

        pred = model(data)

        preds.extend(
            pred.detach().cpu().numpy()
        )

        targets.extend(
            data.y.view(-1).cpu().numpy()
        )

preds = np.array(preds)
targets = np.array(targets)

# =========================================================
# METRICS
# =========================================================

rmse = np.sqrt(
    mean_squared_error(targets, preds)
)

mae = mean_absolute_error(
    targets,
    preds
)

pearson_r, _ = pearsonr(
    targets,
    preds
)

r2 = pearson_r ** 2

# =========================================================
# CI
# =========================================================

def concordance_index(y_true, y_pred):

    n = 0
    h_sum = 0

    for i in range(len(y_true)):

        for j in range(i + 1, len(y_true)):

            if y_true[i] != y_true[j]:

                n += 1

                if (
                    (y_pred[i] > y_pred[j] and y_true[i] > y_true[j])
                    or
                    (y_pred[i] < y_pred[j] and y_true[i] < y_true[j])
                ):

                    h_sum += 1

                elif y_pred[i] == y_pred[j]:

                    h_sum += 0.5

    return h_sum / n

ci = concordance_index(
    targets,
    preds
)

# =========================================================
# SD
# =========================================================

sd = np.std(
    preds - targets
)

# =========================================================
# PRINT RESULTS
# =========================================================

print("\n===== TEST SET METRICS =====")

print(f"RMSE       : {rmse:.4f}")
print(f"MAE        : {mae:.4f}")
print(f"CI         : {ci:.4f}")
print(f"SD         : {sd:.4f}")
print(f"Pearsonr   : {pearson_r:.4f}")
print(f"R predicted: {r2:.4f}")

# =========================================================
# REGRESSION PLOT
# =========================================================

x = targets
y = preds

n = len(x)

slope, intercept, r_value, p_value, slope_stderr = (
    stats.linregress(x, y)
)

xbar = x.mean()

Sxx = np.sum((x - xbar) ** 2)

yhat = intercept + slope * x

SSE = np.sum((y - yhat) ** 2)

dfree = n - 2

s = np.sqrt(SSE / dfree)

intercept_stderr = (
    s * np.sqrt(1.0 / n + (xbar ** 2) / Sxx)
)

xg = np.linspace(
    x.min(),
    x.max(),
    300
)

yg = intercept + slope * xg

t95 = stats.t.ppf(0.975, dfree)
t99 = stats.t.ppf(0.995, dfree)

se_mean = s * np.sqrt(
    1.0 / n + (xg - xbar) ** 2 / Sxx
)

se_pred = s * np.sqrt(
    1.0 + 1.0 / n + (xg - xbar) ** 2 / Sxx
)

# regression CI
ci95_low = yg - t95 * se_mean
ci95_high = yg + t95 * se_mean

ci99_low = yg - t99 * se_mean
ci99_high = yg + t99 * se_mean

# prediction interval
pi95_low = yg - t95 * se_pred
pi95_high = yg + t95 * se_pred

pi99_low = yg - t99 * se_pred
pi99_high = yg + t99 * se_pred

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    x,
    y,
    s=35,
    alpha=0.7,
    label="Test Data"
)

plt.fill_between(
    xg,
    pi99_low,
    pi99_high,
    alpha=0.18,
    label="99% Confidence Interval"
)

plt.fill_between(
    xg,
    pi95_low,
    pi95_high,
    alpha=0.20,
    label="95% Confidence Interval"
)

plt.fill_between(
    xg,
    ci99_low,
    ci99_high,
    alpha=0.25,
    label="99% CI (Regression Line)"
)

plt.fill_between(
    xg,
    ci95_low,
    ci95_high,
    alpha=0.25,
    label="95% CI (Regression Line)"
)

plt.plot(
    xg,
    yg,
    linewidth=3,
    label="Best-fit line"
)

plt.plot(
    xg,
    xg,
    linestyle="--",
    alpha=0.35
)

plt.xlabel(
    "True pKi",
    fontsize=14
)

plt.ylabel(
    "Predicted pKi",
    fontsize=14
)

plt.text(
    0.05,
    0.95,
    f"y = {intercept:.3f} (±{intercept_stderr:.3f}) + "
    f"{slope:.3f} (±{slope_stderr:.3f}) × x\n"
    f"R² = {r2:.3f}",
    transform=plt.gca().transAxes,
    fontsize=11,
    va="top",
)

plt.legend(frameon=True)

plt.tight_layout()

# =========================================================
# SAVE FIGURE
# =========================================================

out_png = os.path.join(
    os.path.dirname(WEIGHTS_PATH),
    "test_pred_vs_true_MYGIGN_2019.png"
)

plt.savefig(
    out_png,
    dpi=300
)

print(f"\nSaved plot to: {out_png}")

plt.show()

# %%