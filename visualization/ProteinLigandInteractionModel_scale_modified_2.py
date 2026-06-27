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
from torch_geometric.data import Batch

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

# Silence RDKit
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
        # If somehow no masked nodes exist, fall back to global mean pool
        return global_mean_pool(x, batch)

    x_m = x[mask]
    b_m = batch[mask]
    return global_mean_pool(x_m, b_m)


class GATBlock(nn.Module):
    """
    Conv -> GraphNorm -> ReLU -> Dropout -> (Residual if same dim)
    """
    def __init__(self, in_dim, out_dim, heads, edge_dim, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.conv = GATConv(in_dim, out_dim, heads=heads, edge_dim=edge_dim)

        self.gn = GraphNorm(out_dim * heads)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.use_res = (in_dim == out_dim * heads)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.conv(x, edge_index, edge_attr=edge_attr)  # [N, out_dim*heads]
        h = self.gn(h, batch)                               # Batch
        h = self.act(h)
        h = self.drop(h)
        if self.use_res:
            h = h + x
        return h


# ================================================================
# Capacity Control (Low-Risk Upgrade)
# ================================================================
BASE_OUT = 96          # change to 64 if you want old size
BASE_HEADS = 4         # change to 2 if you want old size
USE_INT_3RD_LAYER = True


class TriBranchDTI(nn.Module):
    def __init__(
        self,
        prot_node_dim=26, prot_edge_dim=9,
        lig_node_dim=26, lig_edge_dim=9,
        int_node_dim=26, int_edge_dim=9,
        lig_attr_dim=6,
        dropout=0.2
    ):
        super().__init__()

        out = BASE_OUT
        heads = BASE_HEADS
        hid_dim = out * heads

        # ---- Protein branch ----
        self.prot_b1 = GATBlock(prot_node_dim, out, heads=heads, edge_dim=prot_edge_dim, dropout=dropout)
        self.prot_b2 = GATBlock(hid_dim, out, heads=heads, edge_dim=prot_edge_dim, dropout=dropout)
        self.prot_b3 = GATBlock(hid_dim, out, heads=heads, edge_dim=prot_edge_dim, dropout=dropout)

        prot_emb_dim = hid_dim

        # ---- Interaction branch ----
        self.int_b1 = GATBlock(int_node_dim, out, heads=heads, edge_dim=int_edge_dim, dropout=dropout)
        self.int_b2 = GATBlock(hid_dim, out, heads=heads, edge_dim=int_edge_dim, dropout=dropout)

        if USE_INT_3RD_LAYER:
            self.int_b3 = GATBlock(hid_dim, out, heads=heads, edge_dim=int_edge_dim, dropout=dropout)

        int_emb_dim = hid_dim

        # ---- Stage-1 fusion ----
        self.pi_fc1 = nn.Linear(prot_emb_dim + int_emb_dim, 256)
        self.pi_norm = nn.LayerNorm(256)
        self.pi_drop = nn.Dropout(dropout)

        # ---- Ligand branch ----
        self.lig_b1 = GATBlock(lig_node_dim, out, heads=heads, edge_dim=lig_edge_dim, dropout=dropout)
        self.lig_b2 = GATBlock(hid_dim, out, heads=heads, edge_dim=lig_edge_dim, dropout=dropout)
        self.lig_b3 = GATBlock(hid_dim, out, heads=heads, edge_dim=lig_edge_dim, dropout=dropout)

        lig_emb_dim = hid_dim

        self.lig_attr_fc = nn.Sequential(
            nn.Linear(lig_attr_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- Stage-2 fusion ----
        self.fc1 = nn.Linear(256 + lig_emb_dim + 64, 256)
        self.norm1 = nn.LayerNorm(256)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.drop2 = nn.Dropout(dropout)

        self.out = nn.Linear(128, 1)
        self.act = nn.ReLU()

    def forward(self, prot_data, lig_data, int_data):

        # Protein branch
        xp = prot_data.x
        xp = self.prot_b1(xp, prot_data.edge_index, prot_data.edge_attr, prot_data.batch)
        xp = self.prot_b2(xp, prot_data.edge_index, prot_data.edge_attr, prot_data.batch)
        xp = self.prot_b3(xp, prot_data.edge_index, prot_data.edge_attr, prot_data.batch)
        prot_emb = global_mean_pool(xp, prot_data.batch)

        # Interaction branch
        xi = int_data.x
        xi = self.int_b1(xi, int_data.edge_index, int_data.edge_attr, int_data.batch)
        xi = self.int_b2(xi, int_data.edge_index, int_data.edge_attr, int_data.batch)

        if USE_INT_3RD_LAYER:
            xi = self.int_b3(xi, int_data.edge_index, int_data.edge_attr, int_data.batch)

        prot_mask = (int_data.node_type == 1)
        int_emb = masked_mean_pool(xi, int_data.batch, prot_mask)

        # Fusion stage 1
        pi = torch.cat([prot_emb, int_emb], dim=1)
        pi = self.pi_drop(self.act(self.pi_norm(self.pi_fc1(pi))))

        # Ligand branch
        xl = lig_data.x
        xl = self.lig_b1(xl, lig_data.edge_index, lig_data.edge_attr, lig_data.batch)
        xl = self.lig_b2(xl, lig_data.edge_index, lig_data.edge_attr, lig_data.batch)
        xl = self.lig_b3(xl, lig_data.edge_index, lig_data.edge_attr, lig_data.batch)
        lig_emb = global_mean_pool(xl, lig_data.batch)

        attr_emb = self.lig_attr_fc(lig_data.mol_attr)

        # Fusion stage 2
        h = torch.cat([pi, lig_emb, attr_emb], dim=1)
        h = self.drop1(self.act(self.norm1(self.fc1(h))))
        h = self.drop2(self.act(self.norm2(self.fc2(h))))

        return self.out(h)


# ================================================================
# 2) Index loading (target=4th col, year for time split)
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

            pdb = parts[0].lower()
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
# 3) Dataset: returns (prot_graph, lig_graph, int_graph, y)
# ================================================================
class DTIDataset(Dataset):
    def __init__(self, df, base_path, interaction_cutoff=6.0):
        self.df = df.reset_index(drop=True)
        self.base_path = base_path
        self.cutoff = float(interaction_cutoff)
        self.has_weight = "sample_weight" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pdb = row["PDB_code"].lower()
        y = torch.tensor([row["Activity"]], dtype=torch.float32)

        pdb_dir = os.path.join(self.base_path, pdb)
        if not os.path.isdir(pdb_dir):
            return None
        
        w = 1.0
        if self.has_weight:
            w = float(row["sample_weight"])
        w = torch.tensor([w], dtype=torch.float32)

        # ligand sdf
        strict_sdf = os.path.join(pdb_dir, f"{pdb}_ligand.sdf")
        if os.path.exists(strict_sdf):
            sdf_path = strict_sdf
        else:
            sdf_candidates = sorted(glob.glob(os.path.join(pdb_dir, "*ligand*.sdf"))) + \
                             sorted(glob.glob(os.path.join(pdb_dir, "*.sdf")))
            sdf_candidates = [p for p in sdf_candidates if os.path.isfile(p)]
            sdf_path = sdf_candidates[0] if sdf_candidates else None

        # pocket pdb
        strict_pocket = os.path.join(pdb_dir, f"{pdb}_pocket.pdb")
        if os.path.exists(strict_pocket):
            pdb_path = strict_pocket
        else:
            pdb_candidates = sorted(glob.glob(os.path.join(pdb_dir, "*pocket*.pdb"))) + \
                             sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))
            pdb_candidates = [p for p in pdb_candidates if os.path.isfile(p)]
            pdb_path = pdb_candidates[0] if pdb_candidates else None

        # ply for protein surface graph
        ply_candidates = sorted(glob.glob(os.path.join(pdb_dir, "*.ply")))
        ply_path = ply_candidates[0] if ply_candidates else None

        if sdf_path is None or pdb_path is None or ply_path is None:
            return None

        try:
            # ligand graph (needs mol_attr)
            mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
            if mol is None:
                return None
            lig_graph = sdf_to_graph(mol)
            if not hasattr(lig_graph, "mol_attr"):
                # if your sdf_to_graph sometimes doesn't attach mol_attr
                lig_graph.mol_attr = torch.zeros((1, 6), dtype=torch.float32)

            # protein graph from ply+pdb
            prot_graph = protein_to_graph(ply_path, pdb_path)

            # interaction graph from sdf + pocket pdb
            int_graph = interaction_to_graph(sdf_path, pdb_path, cutoff=self.cutoff)

            return prot_graph, lig_graph, int_graph, y, w
        except Exception:
            return None


def tri_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    prots, ligs, inters, ys, ws = zip(*batch)
    return (
        Batch.from_data_list(prots),
        Batch.from_data_list(ligs),
        Batch.from_data_list(inters),
        torch.stack(ys, dim=0),
        torch.stack(ws, dim=0),
    )


# ================================================================
# 4) Metrics
# ================================================================
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))

    residuals = y_pred - y_true
    sd = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float("nan")

    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        pr = float("nan")
    else:
        pr = float(pearsonr(y_true, y_pred)[0])

    tau = kendalltau(y_true, y_pred, nan_policy="omit").correlation
    ci = float((tau + 1.0) / 2.0) if tau is not None else float("nan")

    return {"MSE": float(mse), "RMSE": rmse, "MAE": mae, "CI": ci, "SD": sd, "Pearsonr": pr}


def collect_predictions(loader, model, device):
    model.eval()
    preds, tgts = [], []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            # unpack 5 elements now
            p, l, inter, y, w = batch

            p = p.to(device)
            l = l.to(device)
            inter = inter.to(device)
            y = y.to(device)

            out = model(p, l, inter)

            preds.append(out.detach().cpu().numpy())
            tgts.append(y.detach().cpu().numpy())

    if not preds:
        return None, None

    return np.vstack(tgts).ravel(), np.vstack(preds).ravel()


# ================================================================
# 5) Plots
# ================================================================
def plot_mse_curve(train_mse, val_mse, save_path="mse_loss_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_mse, label="Train MSE")
    plt.plot(val_mse, label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("MSE per Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved: {save_path}")


def plot_pred_vs_true(y_true, y_pred, save_path="pred_vs_true_test.png", title="Pred vs True (Test)"):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.35)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved: {save_path}")


# ================================================================
# 6) Training (time split like paper)
# ================================================================

class StratifiedBatchSampler(Sampler):
    """
    هر batch شامل low/mid/high با نسبت ثابت است.
    - labels: آرایه‌ی int با مقادیر 0/1/2 برای low/mid/high
    - per_class: تعداد نمونه از هر کلاس در هر batch
    """
    def __init__(self, labels, batch_size, shuffle=True, drop_last=False, generator=None):
        assert batch_size % 3 == 0, "برای سادگی batch_size را مضرب 3 بگذار (مثلاً 30/33/36)"
        self.labels = np.asarray(labels, dtype=int)
        self.batch_size = int(batch_size)
        self.per_class = self.batch_size // 3
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

        self.class_indices = {
            c: np.where(self.labels == c)[0].tolist()
            for c in [0, 1, 2]
        }

        # اگر یکی از کلاس‌ها خیلی کم بود، هنوز هم کار می‌کند چون با replacement پر می‌کنیم
        self.num_batches = int(np.ceil(len(self.labels) / self.batch_size))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng() if self.generator is None else self.generator

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
                need = self.per_class

                # اگر کافی نبود، با replacement پر می‌کنیم
                available = len(pools[c]) - ptr[c]
                if available >= need:
                    pick = pools[c][ptr[c]:ptr[c] + need]
                    ptr[c] += need
                else:
                    pick = pools[c][ptr[c]:]
                    ptr[c] = len(pools[c])
                    missing = need - len(pick)
                    if len(self.class_indices[c]) > 0:
                        extra = rng.choice(self.class_indices[c], size=missing, replace=True).tolist()
                        pick = pick + extra
                    else:
                        # کلاس کاملاً تهی (خیلی بعید)، از کل دیتا برمی‌داریم
                        extra = rng.choice(np.arange(len(self.labels)), size=missing, replace=True).tolist()
                        pick = pick + extra

                batch.extend(pick)

            if self.shuffle:
                rng.shuffle(batch)

            if len(batch) < self.batch_size and self.drop_last:
                continue

            yield batch


def train():
    maximize_cpu()

    BASE_PATH = "/Volumes/Ventoy/MolGen_Project copy/data/dataset/PDBbind_general"

    df = load_pdbbind_index(BASE_PATH)
    df = df[df["PDB_code"].apply(lambda x: os.path.isdir(os.path.join(BASE_PATH, x.lower())))]
    df = df.reset_index(drop=True)

    # Time split: test = year > 2019
    test_df = df[df["Year"] > 2019].copy()
    remain_df = df[df["Year"] <= 2019].copy()

    # Shuffle remain set
    remain_df = remain_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 15% validation
    val_size = int(0.15 * len(remain_df))

    val_df = remain_df.iloc[:val_size].copy()
    train_df = remain_df.iloc[val_size:].copy()

    print("Split sizes (before skipping invalid samples at runtime):")
    print(f"  Train: {len(train_df)}")
    print(f"  Val  : {len(val_df)}")
    print(f"  Test : {len(test_df)} (Year > 2019)")

    # ================================================================
    # ✅ A+B) bins برای low/mid/high + sample_weight فقط روی TRAIN
    #     (split تغییر نمی‌کند؛ فقط training loader عوض می‌شود)
    # ================================================================
    y_train_raw = train_df["Activity"].values.astype(np.float32)

    # پیشنهاد ساده: tertile (33% و 66%) یا می‌توانی 20/80 بگذاری
    q1, q2 = np.quantile(y_train_raw, [0.33, 0.66])
    print(f"\n===== STRATIFICATION BINS (on TRAIN raw Activity) =====")
    print(f"low  <= {q1:.4f}")
    print(f"mid  <= {q2:.4f}")
    print(f"high >  {q2:.4f}")
    print("======================================================\n")

    def bin_label(v):
        if v <= q1:
            return 0  # low
        elif v <= q2:
            return 1  # mid
        else:
            return 2  # high

    train_bins = np.array([bin_label(v) for v in y_train_raw], dtype=int)
    train_df["bin_label"] = train_bins

    # وزن‌دهی: inverse frequency (کمترین ریسک)
    counts = np.bincount(train_bins, minlength=3).astype(np.float32)
    inv = 1.0 / np.maximum(counts, 1.0)
    w_per_class = inv / inv.mean()   # نرمال‌سازی میانگین وزن≈1
    print("Bin counts:", counts.tolist())
    print("Class weights (normalized):", w_per_class.tolist())

    train_df["sample_weight"] = np.array([w_per_class[c] for c in train_bins], dtype=np.float32)

    # ================================================================
    # ✅ NEW: Scale Activity using ONLY training set statistics
    # ================================================================
    scaler = StandardScaler()
    train_y = train_df["Activity"].values.reshape(-1, 1).astype(np.float32)
    scaler.fit(train_y)

    # Print scaling params (save these for test-time scaling)
    print("\n===== TARGET SCALING (StandardScaler fitted on TRAIN only) =====")
    print(f"mean_  : {float(scaler.mean_[0]):.6f}")
    print(f"scale_ : {float(scaler.scale_[0]):.6f}")   # std-dev used by StandardScaler
    print("==============================================================\n")

    # Apply the SAME scaling to train/val/test
    train_df["Activity"] = scaler.transform(train_df["Activity"].values.reshape(-1, 1)).astype(np.float32).ravel()
    val_df["Activity"]   = scaler.transform(val_df["Activity"].values.reshape(-1, 1)).astype(np.float32).ravel()
    # test_df["Activity"]  = scaler.transform(test_df["Activity"].values.reshape(-1, 1)).astype(np.float32).ravel()

    # Save scaling params to disk (so you can reuse exactly at test time)
    scaler_path = os.path.join(os.path.dirname(__file__), "target_scaler_params.json")
    with open(scaler_path, "w") as f:
        json.dump(
            {
                "type": "StandardScaler",
                "mean": float(scaler.mean_[0]),
                "scale": float(scaler.scale_[0]),
                "var": float(scaler.var_[0]),
            },
            f,
            indent=2,
        )
    print(f"✅ Saved target scaler params to: {scaler_path}\n")

    num_workers = min(8, os.cpu_count() or 8)

    batch_size = 30  
    train_dataset = DTIDataset(train_df, BASE_PATH, interaction_cutoff=6.0)

    sampler = StratifiedBatchSampler(
        labels=train_df["bin_label"].values,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,     
        collate_fn=tri_collate,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        DTIDataset(val_df, BASE_PATH, interaction_cutoff=6.0),
        batch_size=32,
        shuffle=False,
        collate_fn=tri_collate,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        DTIDataset(test_df, BASE_PATH, interaction_cutoff=6.0),
        batch_size=32,
        shuffle=False,
        collate_fn=tri_collate,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = TriBranchDTI().to(device)

    # Requested LR = 0.0003
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss(reduction="none", beta=1.0)    
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    MAX_EPOCHS = 100
    EARLY_STOP_PATIENCE = 15

    best_val_rmse = float("inf")
    patience_counter = 0

    train_mse_hist, val_mse_hist = [], []

    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            losses = []

            for batch in train_loader:
                if batch is None:
                    continue
                p, l, inter, y, w = batch
                p = p.to(device)
                l = l.to(device)
                inter = inter.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                pred = model(p, l, inter)
                loss_vec = criterion(pred, y)          # [B,1]
                loss = (loss_vec * w).mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if not losses:
                print(f"Epoch {epoch:03d} | TrainMSE nan | (no valid training batches)")
                continue

            train_mse = float(np.mean(losses))

            yv_true, yv_pred = collect_predictions(val_loader, model, device)
            if yv_true is None:
                print(f"Epoch {epoch:03d} | TrainMSE {train_mse:.4f} | Val: no valid batches")
                continue

            vm = compute_metrics(yv_true, yv_pred)
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
                best_val_rmse = val_rmse
                patience_counter = 0
                torch.save(model.state_dict(), "/Volumes/Ventoy/MolGen_Project copy/mol gen project/May30/best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"Early stopping at epoch {epoch} (best ValRMSE={best_val_rmse:.4f})")
                    break

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user (Ctrl+C). Will evaluate best saved model if exists.")

    print("✅ Training finished.")

    # plots (learning curve)
    if train_mse_hist and val_mse_hist:
        plot_mse_curve(train_mse_hist, val_mse_hist, save_path="/Volumes/Ventoy/MolGen_Project copy/mol gen project/May30")

    # Final evaluation from best checkpoint
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("/Volumes/Ventoy/MolGen_Project copy/mol gen project/May30/best_model.pth", map_location=device))
        model.eval()
        print("✅ Loaded best_model.pth for final evaluation.")
    else:
        print("⚠️ best_model.pth not found. Using current model weights.")

    results = []
    for name, loader in [("Train", train_loader), ("Validation", val_loader), ("Test", test_loader)]:
        yt, yp = collect_predictions(loader, model, device)
        if yt is None:
            print(f"{name}: no valid batches")
            continue
        m = compute_metrics(yt, yp)
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
            plot_pred_vs_true(yt, yp, save_path="/Volumes/Ventoy/MolGen_Project copy/mol gen project/May30/pred_vs_true_test.png")

    if results:
        dfm = pd.DataFrame(results).set_index("Split")
        dfm.to_csv("/Volumes/Ventoy/MolGen_Project copy/mol gen project/May30/final_metrics.csv")
        print("Saved metrics: final_metrics.csv")


if __name__ == "__main__":
    train()
