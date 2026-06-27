import os
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import json
from scipy.stats import pearsonr, kendalltau

from torch.optim.lr_scheduler import ReduceLROnPlateau
from rdkit import Chem
from rdkit import RDLogger

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from mol_parser import sdf_to_graph
from protein_parser import protein_to_graph
from interaction_parser_NEW import interaction_to_graph


# ================================================================
# 0) CPU MAX
# ================================================================
def maximize_cpu():
    n = os.cpu_count() or 8
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    torch.set_num_interop_threads(min(4, n))
    print(f"✅ CPU threads set to: {n}")


# ================================================================
# Helpers
# ================================================================
def masked_mean_pool(x: torch.Tensor, batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling per graph but only over masked nodes.
    x: [N, C], batch: [N], mask: [N] bool
    returns [B, C]
    """
    mask = mask.bool()
    if mask.sum() == 0:
        return global_mean_pool(x, batch)
    x_m = x[mask]
    b_m = batch[mask]
    return global_mean_pool(x_m, b_m)


# ================================================================
# Unified Graph Builder
# ================================================================
# Node type codes (kept identical to original convention)
NODE_PROT = 0   # protein branch nodes
NODE_LIG  = 1   # ligand branch nodes
NODE_INT  = 2   # interaction branch nodes

PROJ_DIM  = 128                   # common projected dim per branch
NODE_TYPE_DIM = 3                 # one-hot encoding size
UNIFIED_NODE_DIM = PROJ_DIM + NODE_TYPE_DIM   # 131


def _one_hot_type(node_type_id: int, n_nodes: int) -> torch.Tensor:
    """Return [n_nodes, 3] one-hot tensor for given node_type_id."""
    oh = torch.zeros(n_nodes, NODE_TYPE_DIM, dtype=torch.float32)
    oh[:, node_type_id] = 1.0
    return oh


def build_unified_graph(prot_graph: Data,
                        lig_graph:  Data,
                        int_graph:  Data,
                        cutoff: float = 6.0) -> Data:
    """
    Merge three separate PyG Data objects into one unified graph.

    Node ordering in the merged graph:
        [0 .. Np-1]            → protein nodes   (node_type = NODE_PROT)
        [Np .. Np+Nl-1]        → ligand nodes     (node_type = NODE_LIG)
        [Np+Nl .. Np+Nl+Ni-1]  → interaction nodes (node_type = NODE_INT)

    Edges:
        1. Intra-graph edges from each original graph (index-shifted).
        2. Cross-edges between interaction-graph atoms and their
           counterparts in the protein / ligand graphs, using the
           same 6 Å positional cutoff.

    The interaction graph must carry positional attributes so we can
    match atoms.  We expect:
        int_graph.pos   : [Ni, 3]  3-D coordinates of each int node
        prot_graph.pos  : [Np, 3]
        lig_graph.pos   : [Nl, 3]

    If `.pos` is absent we fall back to zero cross-edges (still valid,
    just no cross-branch message passing).

    Edge attributes for cross-edges are zero-padded to match the
    unified edge_attr dimension (max of the three edge_attr dims).
    """
    Np = prot_graph.x.size(0)
    Nl = lig_graph.x.size(0)
    Ni = int_graph.x.size(0)

    # ── 1. Shift intra-graph edge indices ──────────────────────────
    prot_ei  = prot_graph.edge_index                        # [2, Ep]
    lig_ei   = lig_graph.edge_index  + Np                   # [2, El]
    int_ei   = int_graph.edge_index  + Np + Nl              # [2, Ei]

    # ── 2. Unify edge_attr dims (zero-pad to max) ──────────────────
    ea_p = prot_graph.edge_attr  # [Ep, dp]
    ea_l = lig_graph.edge_attr   # [El, dl]
    ea_i = int_graph.edge_attr   # [Ei, di]

    edge_dim = max(ea_p.size(1), ea_l.size(1), ea_i.size(1))

    def pad_ea(ea):
        if ea.size(1) < edge_dim:
            pad = torch.zeros(ea.size(0), edge_dim - ea.size(1),
                              dtype=ea.dtype, device=ea.device)
            ea = torch.cat([ea, pad], dim=1)
        return ea

    ea_p = pad_ea(ea_p)
    ea_l = pad_ea(ea_l)
    ea_i = pad_ea(ea_i)

    # ── 3. Cross-edges (interaction ↔ protein, interaction ↔ ligand) ─
    cross_src, cross_dst, cross_ea = [], [], []

    has_pos = (hasattr(int_graph, "pos") and int_graph.pos is not None and
               hasattr(prot_graph, "pos") and prot_graph.pos is not None and
               hasattr(lig_graph,  "pos") and lig_graph.pos  is not None)

    if has_pos:
        pos_i = int_graph.pos   # [Ni, 3]
        pos_p = prot_graph.pos  # [Np, 3]
        pos_l = lig_graph.pos   # [Nl, 3]

        # Interaction ↔ Protein cross-edges
        # For each int-node, find protein nodes within cutoff
        # int-node global index = Np + Nl + i
        # prot-node global index = p
        if Np > 0 and Ni > 0:
            # pairwise distances: [Ni, Np]
            diff_ip = pos_i.unsqueeze(1) - pos_p.unsqueeze(0)  # [Ni, Np, 3]
            dist_ip = diff_ip.norm(dim=-1)                      # [Ni, Np]
            ii, pp = torch.where(dist_ip <= cutoff)
            if ii.numel() > 0:
                int_global = (Np + Nl) + ii   # global index of int nodes
                pro_global = pp               # global index of prot nodes
                # bidirectional
                cross_src.append(torch.cat([int_global, pro_global]))
                cross_dst.append(torch.cat([pro_global, int_global]))
                n_cross = ii.numel() * 2
                cross_ea.append(torch.zeros(n_cross, edge_dim,
                                            dtype=ea_p.dtype))

        # Interaction ↔ Ligand cross-edges
        if Nl > 0 and Ni > 0:
            diff_il = pos_i.unsqueeze(1) - pos_l.unsqueeze(0)  # [Ni, Nl, 3]
            dist_il = diff_il.norm(dim=-1)                      # [Ni, Nl]
            ii, ll = torch.where(dist_il <= cutoff)
            if ii.numel() > 0:
                int_global = (Np + Nl) + ii
                lig_global = Np + ll
                cross_src.append(torch.cat([int_global, lig_global]))
                cross_dst.append(torch.cat([lig_global, int_global]))
                n_cross = ii.numel() * 2
                cross_ea.append(torch.zeros(n_cross, edge_dim,
                                            dtype=ea_l.dtype))

    # ── 4. Assemble unified edge_index and edge_attr ───────────────
    all_src = [prot_ei[0], lig_ei[0], int_ei[0]]
    all_dst = [prot_ei[1], lig_ei[1], int_ei[1]]
    all_ea  = [ea_p,       ea_l,      ea_i      ]

    if cross_src:
        all_src += cross_src
        all_dst += cross_dst
        all_ea  += cross_ea

    unified_ei = torch.stack([torch.cat(all_src), torch.cat(all_dst)], dim=0)
    unified_ea = torch.cat(all_ea, dim=0)

    # ── 5. Node features: raw 26-dim kept separate (projectors in model) ─
    # We store branch-raw features + node_type label for the model to use
    x_p = prot_graph.x   # [Np, 26]
    x_l = lig_graph.x    # [Nl, 26]
    x_i = int_graph.x    # [Ni, 26]

    # Pad raw features to the same width (they should all be 26, but be safe)
    raw_dim = max(x_p.size(1), x_l.size(1), x_i.size(1))
    def pad_x(xf):
        if xf.size(1) < raw_dim:
            pad = torch.zeros(xf.size(0), raw_dim - xf.size(1), dtype=xf.dtype)
            xf = torch.cat([xf, pad], dim=1)
        return xf

    x_p = pad_x(x_p)
    x_l = pad_x(x_l)
    x_i = pad_x(x_i)

    unified_x = torch.cat([x_p, x_l, x_i], dim=0)   # [N, 26]

    # node_type label: integer 0/1/2 — used by model for masking & projectors
    node_type = torch.cat([
        torch.full((Np,), NODE_PROT, dtype=torch.long),
        torch.full((Nl,), NODE_LIG,  dtype=torch.long),
        torch.full((Ni,), NODE_INT,  dtype=torch.long),
    ])  # [N]

    # mol_attr lives at graph level (from ligand)
    mol_attr = lig_graph.mol_attr if hasattr(lig_graph, "mol_attr") else \
               torch.zeros(1, 6, dtype=torch.float32)

    unified = Data(
        x         = unified_x,       # [N, 26]  raw per-branch features
        edge_index= unified_ei,       # [2, E]
        edge_attr = unified_ea,       # [E, edge_dim]
        node_type = node_type,        # [N]  integer 0/1/2
        mol_attr  = mol_attr,         # [1, 6]
    )
    return unified


# ================================================================
# GATBlock (identical to original)
# ================================================================
class GATBlock(nn.Module):
    """Conv -> GraphNorm -> ReLU -> Dropout -> (Residual if same dim)"""
    def __init__(self, in_dim, out_dim, heads, edge_dim, dropout=0.2):
        super().__init__()
        self.conv    = GATConv(in_dim, out_dim, heads=heads, edge_dim=edge_dim)
        self.gn      = GraphNorm(out_dim * heads)
        self.act     = nn.ReLU()
        self.drop    = nn.Dropout(dropout)
        self.use_res = (in_dim == out_dim * heads)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.conv(x, edge_index, edge_attr=edge_attr)
        h = self.gn(h, batch)
        h = self.act(h)
        h = self.drop(h)
        if self.use_res:
            h = h + x
        return h


# ================================================================
# Capacity constants (identical to original)
# ================================================================
BASE_OUT          = 96
BASE_HEADS        = 4
USE_INT_3RD_LAYER = True

RAW_NODE_DIM  = 26    # each parser outputs 26-dim node features
PROJ_DIM      = 128   # per-branch projection target


# ================================================================
# Unified Single-Graph DTI Model
# ================================================================
class UnifiedGraphDTI(nn.Module):
    """
    Single-graph variant of TriBranchDTI.

    The three subgraphs are merged into one graph *before* the model.
    Inside the model:
      1. Three branch-specific linear projectors map raw 26-dim → PROJ_DIM.
      2. node_type one-hot (3-dim) is appended → unified node dim = PROJ_DIM+3 = 131.
      3. A single stack of GATBlocks processes the unified graph.
      4. Masked mean-pool by node_type extracts prot_emb / int_emb / lig_emb.
      5. Same 2-stage fusion as TriBranchDTI.
    """
    def __init__(
        self,
        raw_node_dim  = RAW_NODE_DIM,   # 26
        proj_dim      = PROJ_DIM,        # 128
        edge_dim      = 9,               # max edge_attr dim after padding
        lig_attr_dim  = 6,
        dropout       = 0.2,
    ):
        super().__init__()

        # ── Per-branch linear projectors (no bias + LayerNorm for stability) ──
        self.proj_prot = nn.Sequential(
            nn.Linear(raw_node_dim, proj_dim, bias=False),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        self.proj_lig = nn.Sequential(
            nn.Linear(raw_node_dim, proj_dim, bias=False),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        self.proj_int = nn.Sequential(
            nn.Linear(raw_node_dim, proj_dim, bias=False),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )

        unified_node_dim = proj_dim + NODE_TYPE_DIM   # 128 + 3 = 131

        out   = BASE_OUT
        heads = BASE_HEADS
        hid   = out * heads   # 384

        # ── Single shared GAT stack (3 layers, same design as original) ──
        self.gat1 = GATBlock(unified_node_dim, out, heads=heads,
                             edge_dim=edge_dim, dropout=dropout)
        self.gat2 = GATBlock(hid, out, heads=heads,
                             edge_dim=edge_dim, dropout=dropout)
        if USE_INT_3RD_LAYER:
            self.gat3 = GATBlock(hid, out, heads=heads,
                                 edge_dim=edge_dim, dropout=dropout)

        gat_out_dim = hid   # 384

        # ── Stage-1 fusion: prot_emb + int_emb  (identical to original) ──
        self.pi_fc1  = nn.Linear(gat_out_dim + gat_out_dim, 256)
        self.pi_norm = nn.LayerNorm(256)
        self.pi_drop = nn.Dropout(dropout)

        # ── Ligand global attr encoder (identical to original) ──
        self.lig_attr_fc = nn.Sequential(
            nn.Linear(lig_attr_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Stage-2 fusion (identical to original) ──
        self.fc1   = nn.Linear(256 + gat_out_dim + 64, 256)
        self.norm1 = nn.LayerNorm(256)
        self.drop1 = nn.Dropout(dropout)

        self.fc2   = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.drop2 = nn.Dropout(dropout)

        self.out = nn.Linear(128, 1)
        self.act = nn.ReLU()

    def forward(self, unified_data):
        """
        unified_data: a single PyG Data (or Batch) produced by build_unified_graph
            .x          [N, 26]
            .edge_index [2, E]
            .edge_attr  [E, edge_dim]
            .node_type  [N]   int 0/1/2
            .batch      [N]   (added by PyG Batch)
            .mol_attr   [B, 6]
        """
        x         = unified_data.x          # [N, 26]
        edge_index= unified_data.edge_index  # [2, E]
        edge_attr = unified_data.edge_attr   # [E, edge_dim]
        batch     = unified_data.batch       # [N]
        node_type = unified_data.node_type   # [N]
        mol_attr  = unified_data.mol_attr    # [B, 6]  (one row per graph)

        # ── 1. Branch-specific projection ─────────────────────────────
        mask_p = (node_type == NODE_PROT)
        mask_l = (node_type == NODE_LIG)
        mask_i = (node_type == NODE_INT)

        x_proj = torch.zeros(x.size(0), PROJ_DIM, dtype=x.dtype, device=x.device)
        if mask_p.any():
            x_proj[mask_p] = self.proj_prot(x[mask_p])
        if mask_l.any():
            x_proj[mask_l] = self.proj_lig(x[mask_l])
        if mask_i.any():
            x_proj[mask_i] = self.proj_int(x[mask_i])

        # ── 2. Append node_type one-hot ────────────────────────────────
        # [N, 3]
        one_hot = torch.zeros(x.size(0), NODE_TYPE_DIM,
                              dtype=x.dtype, device=x.device)
        one_hot.scatter_(1, node_type.unsqueeze(1), 1.0)

        h = torch.cat([x_proj, one_hot], dim=1)   # [N, 131]

        # ── 3. Shared GAT stack ────────────────────────────────────────
        h = self.gat1(h, edge_index, edge_attr, batch)
        h = self.gat2(h, edge_index, edge_attr, batch)
        if USE_INT_3RD_LAYER:
            h = self.gat3(h, edge_index, edge_attr, batch)

        # ── 4. Masked pooling by node_type ─────────────────────────────
        prot_emb = masked_mean_pool(h, batch, mask_p)   # [B, hid]
        int_emb  = masked_mean_pool(h, batch, mask_i)   # [B, hid]
        lig_emb  = masked_mean_pool(h, batch, mask_l)   # [B, hid]

        # ── 5. Stage-1 fusion (prot + int) ────────────────────────────
        pi = torch.cat([prot_emb, int_emb], dim=1)
        pi = self.pi_drop(self.act(self.pi_norm(self.pi_fc1(pi))))

        # ── 6. Ligand mol_attr encoder ────────────────────────────────
        attr_emb = self.lig_attr_fc(mol_attr)   # [B, 64]

        # ── 7. Stage-2 fusion ─────────────────────────────────────────
        h2 = torch.cat([pi, lig_emb, attr_emb], dim=1)
        h2 = self.drop1(self.act(self.norm1(self.fc1(h2))))
        h2 = self.drop2(self.act(self.norm2(self.fc2(h2))))

        return self.out(h2)   # [B, 1]


# ================================================================
# Dataset — builds unified graph per sample
# ================================================================
class DTIDataset(Dataset):
    def __init__(self, df, base_path, interaction_cutoff=6.0):
        self.df      = df.reset_index(drop=True)
        self.base    = base_path
        self.cutoff  = float(interaction_cutoff)
        self.has_w   = "sample_weight" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pdb = row["PDB_code"].lower()
        y   = torch.tensor([row["Activity"]], dtype=torch.float32)
        w   = torch.tensor([float(row["sample_weight"]) if self.has_w else 1.0],
                           dtype=torch.float32)

        pdb_dir = os.path.join(self.base, pdb)
        if not os.path.isdir(pdb_dir):
            return None

        # ── Locate files ──────────────────────────────────────────────
        strict_sdf = os.path.join(pdb_dir, f"{pdb}_ligand.sdf")
        if os.path.exists(strict_sdf):
            sdf_path = strict_sdf
        else:
            cands = (sorted(glob.glob(os.path.join(pdb_dir, "*ligand*.sdf"))) +
                     sorted(glob.glob(os.path.join(pdb_dir, "*.sdf"))))
            cands = [p for p in cands if os.path.isfile(p)]
            sdf_path = cands[0] if cands else None

        strict_pocket = os.path.join(pdb_dir, f"{pdb}_pocket.pdb")
        if os.path.exists(strict_pocket):
            pdb_path = strict_pocket
        else:
            cands = (sorted(glob.glob(os.path.join(pdb_dir, "*pocket*.pdb"))) +
                     sorted(glob.glob(os.path.join(pdb_dir, "*.pdb"))))
            cands = [p for p in cands if os.path.isfile(p)]
            pdb_path = cands[0] if cands else None

        ply_cands = sorted(glob.glob(os.path.join(pdb_dir, "*.ply")))
        ply_path  = ply_cands[0] if ply_cands else None

        if None in (sdf_path, pdb_path, ply_path):
            return None

        try:
            mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
            if mol is None:
                return None

            lig_graph  = sdf_to_graph(mol)
            if not hasattr(lig_graph, "mol_attr"):
                lig_graph.mol_attr = torch.zeros(1, 6, dtype=torch.float32)

            prot_graph = protein_to_graph(ply_path, pdb_path)
            int_graph  = interaction_to_graph(sdf_path, pdb_path,
                                              cutoff=self.cutoff)

            # Build the unified graph (the only change vs. original dataset)
            unified = build_unified_graph(prot_graph, lig_graph, int_graph,
                                          cutoff=self.cutoff)
            return unified, y, w

        except Exception:
            return None


# ================================================================
# Collate — single graph per sample now
# ================================================================
def unified_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    graphs, ys, ws = zip(*batch)
    return (
        Batch.from_data_list(graphs),
        torch.stack(ys, dim=0),
        torch.stack(ws, dim=0),
    )


# ================================================================
# Metrics (identical to original)
# ================================================================
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_true, y_pred))

    residuals = y_pred - y_true
    sd = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float("nan")

    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        pr = float("nan")
    else:
        pr = float(pearsonr(y_true, y_pred)[0])

    tau = kendalltau(y_true, y_pred, nan_policy="omit").correlation
    ci  = float((tau + 1.0) / 2.0) if tau is not None else float("nan")

    return {"MSE": float(mse), "RMSE": rmse, "MAE": mae,
            "CI": ci, "SD": sd, "Pearsonr": pr}


def collect_predictions(loader, model, device):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            g, y, _ = batch
            g = g.to(device)
            y = y.to(device)
            out = model(g)
            preds.append(out.detach().cpu().numpy())
            tgts.append(y.detach().cpu().numpy())
    if not preds:
        return None, None
    return np.vstack(tgts).ravel(), np.vstack(preds).ravel()


# ================================================================
# Plots (identical to original)
# ================================================================
def plot_mse_curve(train_mse, val_mse, save_path="mse_loss_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_mse, label="Train MSE")
    plt.plot(val_mse,   label="Val MSE")
    plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.title("MSE per Epoch"); plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200); plt.show()
    print(f"Saved: {save_path}")


def plot_pred_vs_true(y_true, y_pred,
                      save_path="pred_vs_true_test.png",
                      title="Pred vs True (Test)"):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.35)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True"); plt.ylabel("Pred")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200); plt.show()
    print(f"Saved: {save_path}")


# ================================================================
# Index loader 
# ================================================================
def extract_year(parts):
    for tok in parts:
        if tok.isdigit() and len(tok) == 4:
            y = int(tok)
            if 1970 <= y <= 2030:
                return y
    return None


def load_pdbbind_index(base_path):
    index_file = os.path.join(base_path, "INDEX_general_PL_minus_tests.2020")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Missing index file: {index_file}")
    rows = []
    with open(index_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            pdb  = parts[0].lower()
            year = extract_year(parts)
            if year is None:
                continue
            try:
                affinity = float(parts[3])
            except ValueError:
                continue
            rows.append((pdb, affinity, year))
    df = pd.DataFrame(rows, columns=["PDB_code", "Activity", "Year"])
    df = df.drop_duplicates("PDB_code").reset_index(drop=True)
    return df


# ================================================================
# Stratified Batch Sampler 
# ================================================================
class StratifiedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, shuffle=True,
                 drop_last=False, generator=None):
        assert batch_size % 3 == 0
        self.labels      = np.asarray(labels, dtype=int)
        self.batch_size  = int(batch_size)
        self.per_class   = self.batch_size // 3
        self.shuffle     = shuffle
        self.drop_last   = drop_last
        self.generator   = generator
        self.class_indices = {
            c: np.where(self.labels == c)[0].tolist() for c in [0, 1, 2]
        }
        self.num_batches = int(np.ceil(len(self.labels) / self.batch_size))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng   = np.random.default_rng() if self.generator is None else self.generator
        pools = {}
        for c in [0, 1, 2]:
            idxs = self.class_indices[c].copy()
            if self.shuffle:
                rng.shuffle(idxs)
            pools[c] = idxs
        ptr = {0: 0, 1: 0, 2: 0}
        for _ in range(self.num_batches):
            batch = []
            for c in [0, 1, 2]:
                need      = self.per_class
                available = len(pools[c]) - ptr[c]
                if available >= need:
                    pick   = pools[c][ptr[c]:ptr[c] + need]
                    ptr[c] += need
                else:
                    pick   = pools[c][ptr[c]:]
                    ptr[c] = len(pools[c])
                    missing = need - len(pick)
                    if self.class_indices[c]:
                        extra = rng.choice(self.class_indices[c],
                                           size=missing, replace=True).tolist()
                        pick  = pick + extra
                    else:
                        extra = rng.choice(np.arange(len(self.labels)),
                                           size=missing, replace=True).tolist()
                        pick  = pick + extra
                batch.extend(pick)
            if self.shuffle:
                rng.shuffle(batch)
            if len(batch) < self.batch_size and self.drop_last:
                continue
            yield batch


# ================================================================
# Training 
# ================================================================
def train():
    maximize_cpu()

    BASE_PATH = "/Volumes/Ventoy/MolGen_Project copy/data/dataset/PDBbind_general"

    df = load_pdbbind_index(BASE_PATH)
    df = df[df["PDB_code"].apply(
        lambda x: os.path.isdir(os.path.join(BASE_PATH, x.lower()))
    )].reset_index(drop=True)

    test_df   = df[df["Year"] > 2019].copy()
    remain_df = df[df["Year"] <= 2019].sample(frac=1.0, random_state=42
                  ).reset_index(drop=True)
    val_size  = int(0.15 * len(remain_df))
    val_df    = remain_df.iloc[:val_size].copy()
    train_df  = remain_df.iloc[val_size:].copy()

    print("Split sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val  : {len(val_df)}")
    print(f"  Test : {len(test_df)} (Year > 2019)")

    # Stratification bins on raw train activity
    y_raw  = train_df["Activity"].values.astype(np.float32)
    q1, q2 = np.quantile(y_raw, [0.33, 0.66])
    print(f"\n===== STRATIFICATION BINS =====")
    print(f"low  <= {q1:.4f}  |  mid <= {q2:.4f}  |  high > {q2:.4f}\n")

    def bin_label(v):
        return 0 if v <= q1 else (1 if v <= q2 else 2)

    train_bins = np.array([bin_label(v) for v in y_raw], dtype=int)
    train_df["bin_label"] = train_bins

    counts      = np.bincount(train_bins, minlength=3).astype(np.float32)
    inv         = 1.0 / np.maximum(counts, 1.0)
    w_per_class = inv / inv.mean()
    train_df["sample_weight"] = np.array([w_per_class[c] for c in train_bins],
                                         dtype=np.float32)

    # Scale targets on train statistics only
    scaler  = StandardScaler()
    train_df["Activity"] = scaler.fit_transform(
        train_df["Activity"].values.reshape(-1, 1)).ravel().astype(np.float32)
    val_df["Activity"]   = scaler.transform(
        val_df["Activity"].values.reshape(-1, 1)).ravel().astype(np.float32)

    print(f"Target scaler — mean: {float(scaler.mean_[0]):.6f}  "
          f"scale: {float(scaler.scale_[0]):.6f}\n")

    scaler_path = os.path.join(os.path.dirname(__file__), "target_scaler_params.json")
    with open(scaler_path, "w") as f:
        json.dump({"type": "StandardScaler",
                   "mean":  float(scaler.mean_[0]),
                   "scale": float(scaler.scale_[0]),
                   "var":   float(scaler.var_[0])}, f, indent=2)
    print(f"✅ Saved target scaler to: {scaler_path}\n")

    num_workers = min(8, os.cpu_count() or 8)
    batch_size  = 30

    train_dataset = DTIDataset(train_df, BASE_PATH, interaction_cutoff=6.0)
    sampler       = StratifiedBatchSampler(
        labels=train_df["bin_label"].values,
        batch_size=batch_size, shuffle=True, drop_last=False)

    train_loader = DataLoader(train_dataset, batch_sampler=sampler,
                              collate_fn=unified_collate,
                              num_workers=num_workers,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(DTIDataset(val_df,  BASE_PATH),
                              batch_size=32, shuffle=False,
                              collate_fn=unified_collate,
                              num_workers=num_workers,
                              persistent_workers=True, prefetch_factor=2)
    test_loader  = DataLoader(DTIDataset(test_df, BASE_PATH),
                              batch_size=32, shuffle=False,
                              collate_fn=unified_collate,
                              num_workers=num_workers,
                              persistent_workers=True, prefetch_factor=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = UnifiedGraphDTI().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss(reduction="none", beta=1.0)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    MAX_EPOCHS         = 100
    EARLY_STOP_PATIENCE= 15
    best_val_rmse      = float("inf")
    patience_counter   = 0
    train_mse_hist, val_mse_hist = [], []

    SAVE_DIR  = "/Volumes/Ventoy/MolGen_Project copy/mol gen project/May30"
    MODEL_PATH= os.path.join(SAVE_DIR, "best_model_unified.pth")

    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            losses = []
            for batch in train_loader:
                if batch is None:
                    continue
                g, y, w = batch
                g = g.to(device); y = y.to(device)
                optimizer.zero_grad()
                pred     = model(g)
                loss_vec = criterion(pred, y)
                loss     = (loss_vec * w).mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if not losses:
                print(f"Epoch {epoch:03d} | no valid batches"); continue

            train_mse = float(np.mean(losses))
            yv_true, yv_pred = collect_predictions(val_loader, model, device)
            if yv_true is None:
                print(f"Epoch {epoch:03d} | TrainMSE {train_mse:.4f} | Val: no valid batches")
                continue

            vm       = compute_metrics(yv_true, yv_pred)
            val_rmse = vm["RMSE"]
            train_mse_hist.append(train_mse)
            val_mse_hist.append(vm["MSE"])
            scheduler.step(val_rmse)
            lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d} | "
                f"TrainMSE {train_mse:.4f} | "
                f"ValRMSE {vm['RMSE']:.4f} | ValMAE {vm['MAE']:.4f} | "
                f"ValCI {vm['CI']:.4f} | ValSD {vm['SD']:.4f} | "
                f"ValPearson {vm['Pearsonr']:.4f} | LR {lr:.2e}"
            )

            if val_rmse < best_val_rmse:
                best_val_rmse    = val_rmse
                patience_counter = 0
                torch.save(model.state_dict(), MODEL_PATH)
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping at epoch {epoch} "
                          f"(best ValRMSE={best_val_rmse:.4f})")
                    break

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted. Will evaluate best saved model if exists.")

    print("✅ Training finished.")

    if train_mse_hist and val_mse_hist:
        plot_mse_curve(train_mse_hist, val_mse_hist,
                       save_path=os.path.join(SAVE_DIR, "mse_loss_curve_unified.png"))

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print("✅ Loaded best checkpoint for final evaluation.")
    else:
        print("⚠️ No checkpoint found. Using current weights.")

    results = []
    for name, loader in [("Train", train_loader),
                         ("Validation", val_loader),
                         ("Test", test_loader)]:
        yt, yp = collect_predictions(loader, model, device)
        if yt is None:
            print(f"{name}: no valid batches"); continue
        m       = compute_metrics(yt, yp)
        m["Split"] = name
        results.append(m)
        print(
            f"\n📌 {name} metrics:\n"
            f"  RMSE      : {m['RMSE']:.4f}\n"
            f"  MAE       : {m['MAE']:.4f}\n"
            f"  CI        : {m['CI']:.4f}\n"
            f"  SD        : {m['SD']:.4f}\n"
            f"  Pearsonr  : {m['Pearsonr']:.4f}\n"
        )
        if name == "Test":
            plot_pred_vs_true(
                yt, yp,
                save_path=os.path.join(SAVE_DIR, "pred_vs_true_test_unified.png"))

    if results:
        dfm = pd.DataFrame(results).set_index("Split")
        dfm.to_csv(os.path.join(SAVE_DIR, "final_metrics_unified.csv"))
        print("Saved: final_metrics_unified.csv")


if __name__ == "__main__":
    train()
