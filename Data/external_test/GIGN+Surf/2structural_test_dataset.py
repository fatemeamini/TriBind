import os
import pickle
import numpy as np
import pandas as pd
import multiprocessing
from itertools import repeat

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset, Batch

from rdkit import Chem
from rdkit import RDLogger
from scipy.spatial import distance_matrix

RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)


# =========================================================
# PIPELINE MATCH REFERENCE
# ─────────────────────────────────────────────────────────
# Training dataset  : dataset_GIGN_general.py
# rdkit file name   : {cid}_5A.rdkit        (in subdir per cid)
# graph file name   : Graph_{cid}_5A.pyg    (in subdir per cid)
#
# Test data (flat folder, single protein):
# rdkit file name   : {ligand_id}_5A.rdkit  (in DATA_DIR)
# graph file name   : Graph_{ligand_id}_5A.pyg (in DATA_DIR)
#
# Node tensor x  (N_total, 42):
#   ligand rows  → 35 graph feats + 7 zeros    (padded)
#   pocket rows  → 35 graph feats + 7 surface  (raw, normalized in Step 3)
#
# Edge building matches training exactly:
#   - both directions added per bond
#   - returns empty (2,0) tensor if no bonds/contacts
# =========================================================


# =========================================================
# ENCODINGS
# =========================================================

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


# =========================================================
# ATOM FEATURES  — identical to dataset_GIGN_general.py
# ─────────────────────────────────────────────────────────
# symbol        10  (9 elements + Unknown)
# degree         7  (0-6)
# impl valence   7  (0-6)
# hybridisation  5  (SP/SP2/SP3/SP3D/SP3D2)
# aromatic       1
# total H        5  (0-4)
# ──────────────────────────────────────────────────────────
# total         35
# =========================================================

def atom_features(mol):
    atom_symbols = ['C','N','O','S','F','P','Cl','Br','I']
    feats = []
    for atom in mol.GetAtoms():
        f = (
            one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) +
            one_of_k_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5,6]) +
            one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6]) +
            one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ]) +
            [atom.GetIsAromatic()] +
            one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4])
        )
        feats.append(np.array(f, dtype=np.float32))
    return torch.tensor(np.array(feats), dtype=torch.float32)   # (N, 35)


# =========================================================
# EDGE INDEX  — identical to dataset_GIGN_general.py
# adds both directions per bond; returns empty (2,0) if no bonds
# =========================================================

def get_edge_index(mol):
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).T   # (2, E)


# =========================================================
# MOLECULE → GRAPH  — identical to dataset_GIGN_general.py
# =========================================================

def mol2graph(mol):
    x          = atom_features(mol)    # (N, 35)
    edge_index = get_edge_index(mol)   # (2, E)
    return x, edge_index


# =========================================================
# INTERACTION GRAPH  — identical to dataset_GIGN_general.py
# adds both directions per contact; returns empty (2,0) if none
# =========================================================

def inter_graph(ligand, pocket, threshold=5.0):
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()

    dist     = distance_matrix(pos_l, pos_p)
    contacts = np.where(dist < threshold)

    atom_num_l = ligand.GetNumAtoms()
    edges      = []

    for i, j in zip(contacts[0], contacts[1]):
        edges.append([i,             j + atom_num_l])
        edges.append([j + atom_num_l, i            ])

    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).T   # (2, E)


# =========================================================
# BUILD GRAPH + SAVE .pyg  — matches dataset_GIGN_general.py
# =========================================================

def mols2graphs(complex_path, label, save_path, dis_threshold=5.0):
    """
    Loads {ligand_id}_5A.rdkit  →  builds PyG Data  →  saves .pyg

    tuple layout (from Step 1):
        ligand          RDKit Mol
        pocket          RDKit Mol  (byres pocket, bonds present)
        surface_feats   np.ndarray (N_pocket, 7)  raw, not normalized

    x shape = (N_ligand + N_pocket, 42):
        ligand  rows → 35 graph feats + 7 zeros      width=42
        pocket  rows → 35 graph feats + 7 surface    width=42  (raw)

    Surface normalization is deferred to Step 3.
    """
    try:
        with open(complex_path, 'rb') as f:
            ligand, pocket, surface_features = pickle.load(f)

        # ── graph features (35-dim each) ──────────────────────────
        x_l, edge_l = mol2graph(ligand)    # (N_l, 35), (2, E_l)
        x_p, edge_p = mol2graph(pocket)    # (N_p, 35), (2, E_p)

        # ── surface features → pocket atoms ───────────────────────
        surf_feat = torch.tensor(
            surface_features, dtype=torch.float32
        )                                   # (N_p, 7)
        surf_dim  = surf_feat.shape[1]      # always 7

        # ── pad ligand with zeros, augment pocket with surface ─────
        x_l = torch.cat(
            [x_l, torch.zeros(x_l.shape[0], surf_dim)], dim=1
        )                                   # (N_l, 42)
        x_p = torch.cat(
            [x_p, surf_feat], dim=1
        )                                   # (N_p, 42)

        assert x_l.shape[1] == x_p.shape[1] == 42, \
            f"Feature width mismatch: x_l={x_l.shape[1]}, x_p={x_p.shape[1]}"

        # ── combined node matrix ───────────────────────────────────
        x = torch.cat([x_l, x_p], dim=0)   # (N_l + N_p, 42)

        # ── intra-molecular edges ──────────────────────────────────
        edge_index_intra = torch.cat(
            [edge_l, edge_p + ligand.GetNumAtoms()], dim=1
        )                                   # (2, E_l + E_p)

        # ── inter-molecular (ligand ↔ pocket) edges ───────────────
        edge_index_inter = inter_graph(
            ligand, pocket, dis_threshold
        )                                   # (2, E_inter)

        # ── label ─────────────────────────────────────────────────
        y = torch.tensor([float(label)], dtype=torch.float32)

        # ── positions ─────────────────────────────────────────────
        pos = torch.tensor(
            np.vstack([
                ligand.GetConformers()[0].GetPositions(),
                pocket.GetConformers()[0].GetPositions()
            ]),
            dtype=torch.float32
        )

        # ── split mask  0=ligand  1=pocket ────────────────────────
        split = torch.cat([
            torch.zeros(ligand.GetNumAtoms(), dtype=torch.long),
            torch.ones( pocket.GetNumAtoms(), dtype=torch.long)
        ])

        data = Data(
            x                = x,
            edge_index_intra = edge_index_intra,
            edge_index_inter = edge_index_inter,
            y                = y,
            pos              = pos,
            split            = split
        )

        torch.save(data, save_path)
        print(f"[SAVED] {os.path.basename(save_path)}"
              f"  x={tuple(x.shape)}"
              f"  intra={tuple(edge_index_intra.shape)}"
              f"  inter={tuple(edge_index_inter.shape)}")

    except Exception as e:
        print(f"[ERROR] {complex_path}: {e}")


# =========================================================
# DATASET
# =========================================================

class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


class GraphDataset(Dataset):

    def __init__(
        self,
        data_dir,
        data_df,
        dis_threshold=5,
        graph_type='Graph_GIGN',
        num_process=8,
        create=False
    ):
        super().__init__()

        self.data_dir      = data_dir
        self.data_df       = data_df
        self.dis_threshold = dis_threshold
        self.graph_type    = graph_type
        self.num_process   = num_process
        self.create        = create

        self.rdkit_paths = []
        self.graph_paths = []
        self.ligand_ids  = []
        self.labels      = []
        self.ignored     = []

        self._prepare()

    # ── required by PyG Dataset ───────────────────────────────────
    def len(self):
        return len(self.graph_paths)

    def get(self, idx):
        return torch.load(self.graph_paths[idx], weights_only=False)

    # ── also support standard DataLoader indexing ─────────────────
    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        return self.get(idx)

    def collate_fn(self, batch):
        return Batch.from_data_list(batch)

    # ── prepare ───────────────────────────────────────────────────
    def _prepare(self):

        print("\n==============================")
        print("DATASET PREPROCESSING")
        print("==============================")

        for _, row in self.data_df.iterrows():
            ligand_id = str(row["Name"]).strip()

            # filenames match Step 1 output and training convention
            # Step 1 saves : {ligand_id}_5A.rdkit   (flat dir)
            # Graph saved  : Graph_GIGN-{ligand_id}_5A.pyg
            rdkit_path = os.path.join(
                self.data_dir,
                f"{ligand_id}_{self.dis_threshold}A.rdkit"
            )
            graph_path = os.path.join(
                self.data_dir,
                f"{self.graph_type}-{ligand_id}_{self.dis_threshold}A.pyg"
            )

            if not os.path.exists(rdkit_path):
                print(f"  [IGNORED] Missing rdkit : {ligand_id}")
                self.ignored.append(ligand_id)
                continue

            # ── validate pocket / surface consistency ──────────────
            try:
                with open(rdkit_path, "rb") as f:
                    _, pocket, surf = pickle.load(f)

                n_pocket = pocket.GetNumAtoms()
                n_bonds  = pocket.GetNumBonds()

                if surf.shape[0] != n_pocket:
                    print(f"  [IGNORED] surface/pocket mismatch "
                          f"({surf.shape[0]} vs {n_pocket}) : {ligand_id}")
                    self.ignored.append(ligand_id)
                    continue

                if n_bonds == 0:
                    print(f"  [WARNING] Pocket has 0 bonds : {ligand_id} "
                          f"— re-run Step 1 (temp PDB fix)")

            except Exception as e:
                print(f"  [IGNORED] Cannot read {ligand_id}: {e}")
                self.ignored.append(ligand_id)
                continue

            try:
                label = float(row["REAL VALUE"])
            except Exception:
                label = 0.0

            self.rdkit_paths.append(rdkit_path)
            self.graph_paths.append(graph_path)
            self.ligand_ids.append(ligand_id)
            self.labels.append(label)

        print(f"\n  Total complexes queued : {len(self.graph_paths)}")
        print(f"  Total ignored          : {len(self.ignored)}")

        if self.ignored:
            with open(os.path.join(self.data_dir, "ignored_dataset.txt"), "w") as f:
                f.write("\n".join(self.ignored))

        # ── create .pyg files ─────────────────────────────────────
        if self.create:
            print(f"\n  Building graphs with {self.num_process} workers ...")

            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(
                mols2graphs,
                zip(
                    self.rdkit_paths,
                    self.labels,
                    self.graph_paths,
                    repeat(self.dis_threshold)
                )
            )
            pool.close()
            pool.join()

            print("\n  Graph generation finished.")

        # ── tensor size report ────────────────────────────────────
        self._report_tensor_sizes()

    def _report_tensor_sizes(self):
        for i, path in enumerate(self.graph_paths):
            if not os.path.exists(path):
                continue
            try:
                data       = torch.load(path, weights_only=False)
                atom_num_l = int((data.split == 0).sum())
                atom_num_p = int((data.split == 1).sum())

                print("\n==============================")
                print(f"TENSOR SIZES  [{self.ligand_ids[i]}]")
                print("==============================")
                print(f"  Ligand atoms           : {atom_num_l}")
                print(f"  Pocket atoms           : {atom_num_p}"
                      f"  ← training range ~80-200")
                print(f"  Total nodes            : {atom_num_l + atom_num_p}")
                print(f"  x                      : {tuple(data.x.shape)}")
                print(f"    ligand  [0:{atom_num_l}]"
                      f"  → 35 graph + 7 zeros    width=42")
                print(f"    pocket  [{atom_num_l}:{atom_num_l+atom_num_p}]"
                      f"  → 35 graph + 7 surface  width=42 (raw)")
                print(f"  edge_index_intra       : {tuple(data.edge_index_intra.shape)}")
                print(f"  edge_index_inter       : {tuple(data.edge_index_inter.shape)}")
                print(f"  y (pK label)           : {data.y.item():.4f}")
                print(f"  NOTE: surface cols [35:42] normalized in Step 3")
                return
            except Exception as e:
                print(f"  [WARN] {path}: {e}")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    DATA_DIR = "/Users/fatemeh/GIGN/GIGN/data/structural_test"
    CSV_FILE = os.path.join(DATA_DIR, "structural_study.csv")

    df = pd.read_csv(CSV_FILE)

    dataset = GraphDataset(
        data_dir      = DATA_DIR,
        data_df       = df,
        graph_type    = "Graph_GIGN",
        dis_threshold = 5,
        create        = True,
        num_process   = 8
    )

    print(f"\nDataset size: {dataset.len()} graphs")