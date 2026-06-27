import os
import glob
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from torch.utils.data import DataLoader

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


# =========================================================
# INLINE MODEL DEFINITION
# ─────────────────────────────────────────────────────────
# Copied verbatim from the uploaded GIGN_My.py so this
# script has zero external imports beyond standard libs.
# =========================================================

import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import global_add_pool
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    D_mu     = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu     = D_mu.view([1, -1])
    D_sigma  = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)


class HIL(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.in_channels  = in_channels
        self.out_channels = out_channels

        self.mlp_node_cov = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Dropout(0.1), nn.LeakyReLU(),
            nn.BatchNorm1d(out_channels)
        )
        self.mlp_node_ncov = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Dropout(0.1), nn.LeakyReLU(),
            nn.BatchNorm1d(out_channels)
        )
        self.mlp_coord_cov  = nn.Sequential(nn.Linear(9, in_channels), nn.SiLU())
        self.mlp_coord_ncov = nn.Sequential(nn.Linear(9, in_channels), nn.SiLU())

    def forward(self, x, edge_index_intra, edge_index_inter, pos=None, size=None):
        row_cov, col_cov   = edge_index_intra
        coord_diff_cov     = pos[row_cov] - pos[col_cov]
        radial_cov         = self.mlp_coord_cov(
            _rbf(torch.norm(coord_diff_cov, dim=-1),
                 D_min=0., D_max=6., D_count=9, device=x.device)
        )
        out_node_intra     = self.propagate(edge_index=edge_index_intra,
                                            x=x, radial=radial_cov, size=size)

        row_ncov, col_ncov = edge_index_inter
        coord_diff_ncov    = pos[row_ncov] - pos[col_ncov]
        radial_ncov        = self.mlp_coord_ncov(
            _rbf(torch.norm(coord_diff_ncov, dim=-1),
                 D_min=0., D_max=6., D_count=9, device=x.device)
        )
        out_node_inter     = self.propagate(edge_index=edge_index_inter,
                                            x=x, radial=radial_ncov, size=size)

        return (self.mlp_node_cov(x + out_node_intra) +
                self.mlp_node_ncov(x + out_node_inter))

    def message(self, x_j: Tensor, x_i: Tensor, radial, index: Tensor):
        return x_j * radial


class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super().__init__()
        self.predict = nn.ModuleList()
        for j in range(n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(d_graph_layer, d_FC_layer))
                self.predict.append(nn.Dropout(dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == n_FC_layer - 1:
                self.predict.append(nn.Linear(d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(d_FC_layer, d_FC_layer))
                self.predict.append(nn.Dropout(dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        return h


class GIGN(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.lin_node = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        self.gconv1   = HIL(hidden_dim, hidden_dim)
        self.gconv2   = HIL(hidden_dim, hidden_dim)
        self.gconv3   = HIL(hidden_dim, hidden_dim)
        self.fc       = FC(hidden_dim, hidden_dim, 3, 0.1, 1)

    def forward(self, data):
        x, edge_index_intra, edge_index_inter, pos = \
            data.x, data.edge_index_intra, data.edge_index_inter, data.pos
        x = self.lin_node(x)
        x = self.gconv1(x, edge_index_intra, edge_index_inter, pos)
        x = self.gconv2(x, edge_index_intra, edge_index_inter, pos)
        x = self.gconv3(x, edge_index_intra, edge_index_inter, pos)
        x = global_add_pool(x, data.batch)
        x = self.fc(x)
        return x.view(-1)


# =========================================================
# PATHS  — edit these
# =========================================================

DATA_DIR   = "/Users/fatemeh/GIGN/GIGN/data/structural_test"
CSV_FILE   = os.path.join(DATA_DIR, "structural_study.csv")
MODEL_PATH = "/Users/fatemeh/GIGN/GIGN/Backup/best_GIGN_surface.pt"
STATS_PATH = os.path.join(DATA_DIR, "surface_norm_stats.npz")
OUT_CSV    = os.path.join(DATA_DIR, "predictions.csv")

GRAPH_TYPE    = "Graph_GIGN"
DIS_THRESHOLD = 5


# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice : {device}")


# =========================================================
# LOAD SURFACE NORMALIZATION STATS
# ─────────────────────────────────────────────────────────
# Written by diagnose.py from the full training set.
# Applied only to pocket atom rows, cols [35:42].
# =========================================================

if not os.path.exists(STATS_PATH):
    raise FileNotFoundError(
        f"surface_norm_stats.npz not found at {STATS_PATH}\n"
        "Run diagnose.py first to compute training surface stats."
    )

stats     = np.load(STATS_PATH)
SURF_MEAN = torch.FloatTensor(stats["mean"])   # (7,)
SURF_STD  = torch.FloatTensor(stats["std"])    # (7,)

print(f"\nSurface norm stats loaded from:\n  {STATS_PATH}")
print(f"  SURF_MEAN : {SURF_MEAN.numpy().round(4).tolist()}")
print(f"  SURF_STD  : {SURF_STD.numpy().round(4).tolist()}")


# =========================================================
# LOAD MODEL
# =========================================================

print("\n==============================")
print("LOADING MODEL")
print("==============================")

model = GIGN(node_dim=42, hidden_dim=256)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=False)
)
model.to(device)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"  Checkpoint : {MODEL_PATH}")
print(f"  node_dim   : 42  (35 graph + 7 surface features)")
print(f"  hidden_dim : 256")
print(f"  Parameters : {n_params:,}")


# =========================================================
# LOAD TEST GRAPHS + APPLY SURFACE NORMALIZATION
# =========================================================

print("\n==============================")
print("LOADING TEST GRAPHS")
print("==============================")

df        = pd.read_csv(CSV_FILE)
graphs    = []
names     = []
real_vals = []
ignored   = []

for _, row in df.iterrows():
    lid      = str(row["Name"]).strip()
    pyg_path = os.path.join(
        DATA_DIR,
        f"{GRAPH_TYPE}-{lid}_{DIS_THRESHOLD}A.pyg"
    )

    if not os.path.exists(pyg_path):
        print(f"  [IGNORED] Missing .pyg : {lid}")
        ignored.append(lid)
        continue

    try:
        data = torch.load(pyg_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [ERROR] Cannot load {lid}: {e}")
        ignored.append(lid)
        continue

    # ── validate node feature width ───────────────────────────────
    if data.x.shape[1] != 42:
        print(f"  [IGNORED] Wrong feature dim "
              f"{data.x.shape[1]} (expected 42) : {lid}")
        ignored.append(lid)
        continue

    # ── apply surface normalization to pocket rows, cols [35:42] ──
    # Pocket rows are where split == 1
    # This mirrors how training graphs were built (surface already
    # normalized before being stored in training .pyg files).
    pocket_mask = (data.split == 1)

    data.x[pocket_mask, 35:] = (
        data.x[pocket_mask, 35:] - SURF_MEAN
    ) / SURF_STD

    graphs.append(data)
    names.append(lid)

    try:
        real_vals.append(float(row["REAL VALUE"]))
    except Exception:
        real_vals.append(np.nan)

print(f"\n  Loaded  : {len(graphs)}")
print(f"  Ignored : {len(ignored)}")

if len(graphs) == 0:
    raise RuntimeError(
        "No graphs loaded. Ensure Step 1 and Step 2 completed "
        "successfully and .pyg files exist in DATA_DIR."
    )

# ── report tensor shape on first sample ───────────────────────────
sample     = graphs[0]
atom_num_l = int((sample.split == 0).sum())
atom_num_p = int((sample.split == 1).sum())

print(f"\n  Sample graph  [{names[0]}]")
print(f"    x                : {tuple(sample.x.shape)}")
print(f"    ligand atoms     : {atom_num_l}  "
      f"cols [0:35]=graph  [35:42]=zeros")
print(f"    pocket atoms     : {atom_num_p}  "
      f"cols [0:35]=graph  [35:42]=surf (normalized)")
print(f"    edge_index_intra : {tuple(sample.edge_index_intra.shape)}")
print(f"    edge_index_inter : {tuple(sample.edge_index_inter.shape)}")
print(f"    y (real pK)      : {sample.y.item():.4f}")


# =========================================================
# COLLATE FUNCTION  — matches training DataLoader style
# =========================================================

def collate_fn(batch):
    return Batch.from_data_list(batch)


# =========================================================
# BATCH PREDICTION
# =========================================================

print("\n==============================")
print("RUNNING PREDICTIONS")
print("==============================")

loader = DataLoader(
    graphs,
    batch_size  = 16,
    shuffle     = False,
    collate_fn  = collate_fn
)

preds = []

with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
        batch       = batch.to(device)
        out         = model(batch)
        batch_preds = out.cpu().numpy().flatten().tolist()
        preds.extend(batch_preds)
        print(f"  Batch {batch_idx+1:>2}/{len(loader)}"
              f"  |  n={len(batch_preds)}"
              f"  |  pK range [{min(batch_preds):.3f},"
              f" {max(batch_preds):.3f}]")

preds = np.array(preds/100)


# =========================================================
# METRICS  (for rows that have a real pK value)
# =========================================================

real_arr   = np.array(real_vals, dtype=np.float64)
valid_mask = ~np.isnan(real_arr)

if valid_mask.sum() > 1:
    from scipy.stats import pearsonr
    r, _  = pearsonr(real_arr[valid_mask], preds[valid_mask])
    rmse  = np.sqrt(np.mean((real_arr[valid_mask] - preds[valid_mask]) ** 2))
    mae   = np.mean(np.abs(real_arr[valid_mask]  - preds[valid_mask]))

    print(f"\n==============================")
    print(f"PERFORMANCE METRICS")
    print(f"==============================")
    print(f"  N (with real values) : {valid_mask.sum()}")
    print(f"  Pearson R            : {r:.4f}")
    print(f"  RMSE                 : {rmse:.4f}")
    print(f"  MAE                  : {mae:.4f}")


# =========================================================
# SAVE RESULTS CSV
# =========================================================

out_df = pd.DataFrame({
    "Name"            : names,
    "REAL_VALUE"      : real_vals,
    "PREDICTED_VALUE" : np.round(preds, 3).tolist(),
})

out_df.to_csv(OUT_CSV, index=False)

print(f"\n==============================")
print(f"RESULTS SAVED")
print(f"==============================")
print(f"  File : {OUT_CSV}")
print(f"  Rows : {len(out_df)}")
print(f"\n{out_df.to_string(index=False)}")