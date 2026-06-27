# %%
import os
import json
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphNorm
from torch.utils.data import WeightedRandomSampler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

from scipy.stats import pearsonr, kendalltau

import matplotlib.pyplot as plt

from model_se3 import SurfaceEGNN

# =========================================================
# REPRODUCIBILITY
# =========================================================

seed = 42

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

# =========================================================
# PATHS
# =========================================================

graph_dir = (
    "/Volumes/Ventoy/MolGen_Project copy/"
    "data/dataset/PDBbind_general"
)

csv_path = (
    "/Volumes/Ventoy/MolGen_Project copy/"
    "data/dataset/PDBbind_general/"
    "index_general2020.csv"
)

save_dir = (
    "/Volumes/Ventoy/MolGen_Project copy/"
    "results_surface"
)

os.makedirs(save_dir, exist_ok=True)

# =========================================================
# LOAD CSV
# =========================================================

df = pd.read_csv(csv_path)

# =========================================================
# LOAD VALID GRAPHS
# =========================================================

graphs = []
targets = []

for _, row in df.iterrows():

    cid = row['pdbid']

    graph_path = os.path.join(
        graph_dir,
        cid,
        f"Graph_{cid}_5A.pyg"
    )

    if not os.path.exists(graph_path):
        continue

    try:

        data = torch.load(graph_path)

        # IMPORTANT
        # ensure y exists
        if not hasattr(data, "y"):
            continue

        graphs.append(data)

        targets.append(
            float(data.y.view(-1)[0])
        )

    except Exception as e:

        print(f"Error loading {cid}: {e}")

print(f"\nTotal valid complexes: {len(graphs)}")

# =========================================================
# SHUFFLE
# =========================================================

combined = list(zip(graphs, targets))

random.shuffle(combined)

graphs, targets = zip(*combined)

graphs = list(graphs)
targets = np.array(targets)

# =========================================================
# SPLIT
# =========================================================

split_idx = int(0.9 * len(graphs))

train_graphs = graphs[:split_idx]
valid_graphs = graphs[split_idx:]

train_targets = targets[:split_idx]
valid_targets = targets[split_idx:]

print(f"Train size: {len(train_graphs)}")
print(f"Valid size: {len(valid_graphs)}")

# =========================================================
# TARGET SCALING
# =========================================================

scaler = StandardScaler()

train_targets_scaled = scaler.fit_transform(
    train_targets.reshape(-1, 1)
).ravel()

valid_targets_scaled = scaler.transform(
    valid_targets.reshape(-1, 1)
).ravel()

print("\n===== TARGET SCALING =====")
print(f"mean  : {scaler.mean_[0]:.6f}")
print(f"scale : {scaler.scale_[0]:.6f}")
print("==========================")

# =========================================================
# SAVE SCALER
# =========================================================

scaler_path = os.path.join(
    save_dir,
    "target_scaler.json"
)

with open(scaler_path, "w") as f:

    json.dump(
        {
            "mean": float(scaler.mean_[0]),
            "scale": float(scaler.scale_[0]),
            "var": float(scaler.var_[0]),
        },
        f,
        indent=2
    )

print(f"\nSaved scaler: {scaler_path}")

# =========================================================
# APPLY SCALED TARGETS
# =========================================================

for i in range(len(train_graphs)):

    train_graphs[i].y = torch.tensor(
        [train_targets_scaled[i]],
        dtype=torch.float32
    )

for i in range(len(valid_graphs)):

    valid_graphs[i].y = torch.tensor(
        [valid_targets_scaled[i]],
        dtype=torch.float32
    )

# =========================================================
# STRATIFICATION
# =========================================================

q1, q2 = np.quantile(
    train_targets,
    [0.33, 0.66]
)

print("\n===== STRATIFICATION =====")
print(f"Low  <= {q1:.4f}")
print(f"Mid  <= {q2:.4f}")
print(f"High >  {q2:.4f}")
print("==========================")

def get_bin(x):

    if x <= q1:
        return 0

    elif x <= q2:
        return 1

    else:
        return 2

train_bins = np.array(
    [get_bin(x) for x in train_targets]
)

# =========================================================
# CLASS WEIGHTS
# =========================================================

counts = np.bincount(
    train_bins,
    minlength=3
).astype(np.float32)

inv = 1.0 / counts

class_weights = inv / inv.mean()

print(f"\nBin counts: {counts.tolist()}")
print(f"Class weights: {class_weights.tolist()}")

sample_weights = np.array(
    [class_weights[b] for b in train_bins],
    dtype=np.float32
)

# =========================================================
# ATTACH SAMPLE WEIGHTS
# =========================================================

for i in range(len(train_graphs)):

    train_graphs[i].sample_weight = torch.tensor(
        [sample_weights[i]],
        dtype=torch.float32
    )

for i in range(len(valid_graphs)):

    valid_graphs[i].sample_weight = torch.tensor(
        [1.0],
        dtype=torch.float32
    )

# =========================================================
# CUSTOM COLLATE
# =========================================================

def custom_collate(batch):

    batch = [x for x in batch if x is not None]

    if len(batch) == 0:
        return None

    weights = torch.stack(
        [
            x.sample_weight
            for x in batch
        ]
    )

    from torch_geometric.data import Batch

    batch_graph = Batch.from_data_list(batch)

    return batch_graph, weights

# =========================================================
# DATALOADERS
# =========================================================

# =========================================================
# STRATIFIED SAMPLER
# =========================================================

sampler = WeightedRandomSampler(

    weights=sample_weights,

    num_samples=len(sample_weights),

    replacement=True
)

# =========================================================
# DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_graphs,
    batch_size=30,
    sampler=sampler,
    num_workers=0
)

valid_loader = DataLoader(
    valid_graphs,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# =========================================================
# MODEL
# =========================================================

model = SurfaceEGNN(
    node_dim=42,
    hidden_dim=256,
    num_layers=5,
    dropout=0.2
).to(device)

# =========================================================
# GRAPHNORM INJECTION
# =========================================================

for name, module in model.named_modules():

    if isinstance(module, nn.Linear):

        pass

# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-5
)

criterion = nn.SmoothL1Loss(
    reduction='none',
    beta=1.0
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=8
)

# =========================================================
# METRICS
# =========================================================

def compute_metrics(y_true, y_pred):

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r, _ = pearsonr(
        y_true,
        y_pred
    )

    tau, _ = kendalltau(
        y_true,
        y_pred
    )

    ci = (tau + 1) / 2

    residuals = y_pred - y_true

    sd = np.std(
        residuals,
        ddof=1
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Pearson": r,
        "CI": ci,
        "SD": sd
    }

# =========================================================
# TRAIN
# =========================================================

def train_epoch():

    model.train()

    total_loss = 0

    for batch in train_loader:

        if batch is None:
            continue

        batch = batch.to(device)

        optimizer.zero_grad()

        pred = model(batch)

        y = batch.y.view(-1, 1)

        w = batch.sample_weight.view(-1, 1)

        loss_vec = criterion(pred, y)

        loss = (loss_vec * w).mean()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# =========================================================
# VALIDATE
# =========================================================

def validate():

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():

        for batch in valid_loader:

            if batch is None:
                continue

            batch = batch.to(device)

            pred = model(batch)

            preds.extend(
                pred.cpu().numpy()
            )

            targets.extend(
                batch.y.cpu().numpy()
            )

    preds = np.array(preds)

    targets = np.array(targets)

    preds = scaler.inverse_transform(
        preds.reshape(-1, 1)
    ).ravel()

    targets = scaler.inverse_transform(
        targets.reshape(-1, 1)
    ).ravel()

    return (
        compute_metrics(targets, preds),
        targets,
        preds
    )

# =========================================================
# PLOT
# =========================================================

def plot_predictions(
    y_true,
    y_pred,
    save_path
):

    plt.figure(figsize=(7, 7))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.5
    )

    mn = min(
        np.min(y_true),
        np.min(y_pred)
    )

    mx = max(
        np.max(y_true),
        np.max(y_pred)
    )

    plt.plot(
        [mn, mx],
        [mn, mx],
        linestyle='--'
    )

    plt.xlabel("True pKd/pKi")
    plt.ylabel("Predicted pKd/pKi")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.show()

# =========================================================
# TRAINING LOOP
# =========================================================

epochs = 100

best_rmse = 999

early_stop_counter = 0

EARLY_STOP = 15

train_losses = []
val_losses = []

for epoch in range(1, epochs + 1):

    train_loss = train_epoch()

    metrics, yt, yp = validate()

    scheduler.step(
        metrics["RMSE"]
    )

    lr = optimizer.param_groups[0]['lr']

    print(
        f"Epoch {epoch:03d} | "
        f"TrainLoss {train_loss:.4f} | "
        f"RMSE {metrics['RMSE']:.4f} | "
        f"MAE {metrics['MAE']:.4f} | "
        f"R {metrics['Pearson']:.4f} | "
        f"CI {metrics['CI']:.4f} | "
        f"SD {metrics['SD']:.4f} | "
        f"LR {lr:.2e}"
    )

    train_losses.append(train_loss)

    val_losses.append(metrics["RMSE"])

    # =====================================
    # SAVE BEST
    # =====================================

    if metrics["RMSE"] < best_rmse:

        best_rmse = metrics["RMSE"]

        early_stop_counter = 0

        save_model_path = os.path.join(
            save_dir,
            "best_GIGN_surface.pt"
        )

        torch.save(
            model.state_dict(),
            save_model_path
        )

        print("\n✅ Best model saved.")

    else:

        early_stop_counter += 1

        if early_stop_counter >= EARLY_STOP:

            print("\n⏹️ Early stopping triggered.")

            break

# =========================================================
# FINAL EVALUATION
# =========================================================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        os.path.join(
            save_dir,
            "best_GIGN_surface.pt"
        ),
        map_location=device
    )
)

metrics, yt, yp = validate()

print("\n===== FINAL VALIDATION =====")

for k, v in metrics.items():

    print(f"{k}: {v:.4f}")

# =========================================================
# SAVE METRICS
# =========================================================

metrics_path = os.path.join(
    save_dir,
    "final_metrics.json"
)

with open(metrics_path, "w") as f:

    json.dump(
        metrics,
        f,
        indent=2
    )

# =========================================================
# PLOTS
# =========================================================

plot_predictions(
    yt,
    yp,
    os.path.join(
        save_dir,
        "pred_vs_true.png"
    )
)

# =========================================================
# LOSS CURVE
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    label="Train Loss"
)

plt.plot(
    val_losses,
    label="Validation RMSE"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

curve_path = os.path.join(
    save_dir,
    "training_curve.png"
)

plt.savefig(
    curve_path,
    dpi=300
)

plt.show()

print("\n✅ Training completed.")