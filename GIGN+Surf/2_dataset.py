# %%
import os
import pickle
import numpy as np
import pandas as pd
import multiprocessing
from itertools import repeat

import torch

from rdkit import Chem
from rdkit import RDLogger

from scipy.spatial import distance_matrix

from torch_geometric.data import Data, Dataset

RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)


# =========================================================
# ENCODINGS
# =========================================================

def one_of_k_encoding_unk(x, allowable_set):

    if x not in allowable_set:
        x = allowable_set[-1]

    return [x == s for s in allowable_set]


# =========================================================
# ATOM FEATURES
# ORIGINAL ARTICLE STYLE
# =========================================================

def atom_features(mol):

    atom_symbols = [
        'C', 'N', 'O', 'S', 'F',
        'P', 'Cl', 'Br', 'I'
    ]

    feats = []

    for atom in mol.GetAtoms():

        f = (

            one_of_k_encoding_unk(
                atom.GetSymbol(),
                atom_symbols + ['Unknown']
            )

            +

            one_of_k_encoding_unk(
                atom.GetDegree(),
                [0, 1, 2, 3, 4, 5, 6]
            )

            +

            one_of_k_encoding_unk(
                atom.GetImplicitValence(),
                [0, 1, 2, 3, 4, 5, 6]
            )

            +

            one_of_k_encoding_unk(
                atom.GetHybridization(),
                [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2
                ]
            )

            +

            [atom.GetIsAromatic()]

            +

            one_of_k_encoding_unk(
                atom.GetTotalNumHs(),
                [0, 1, 2, 3, 4]
            )

        )

        feats.append(
            np.array(f, dtype=np.float32)
        )

    return torch.tensor(
        np.array(feats),
        dtype=torch.float32
    )


# =========================================================
# EDGE INDEX
# =========================================================

def get_edge_index(mol):

    edges = []

    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edges.append([i, j])
        edges.append([j, i])

    if len(edges) == 0:

        return torch.empty(
            (2, 0),
            dtype=torch.long
        )

    return torch.tensor(
        edges,
        dtype=torch.long
    ).T


# =========================================================
# MOLECULE GRAPH
# =========================================================

def mol2graph(mol):

    x = atom_features(mol)

    edge_index = get_edge_index(mol)

    return x, edge_index


# =========================================================
# INTERACTION GRAPH
# EXACT ARTICLE STYLE
# =========================================================

def inter_graph(
    ligand,
    pocket,
    threshold=5.0
):

    pos_l = ligand.GetConformers()[0].GetPositions()

    pos_p = pocket.GetConformers()[0].GetPositions()

    dist = distance_matrix(pos_l, pos_p)

    contacts = np.where(dist < threshold)

    atom_num_l = ligand.GetNumAtoms()

    edges = []

    for i, j in zip(
        contacts[0],
        contacts[1]
    ):

        edges.append([
            i,
            j + atom_num_l
        ])

        edges.append([
            j + atom_num_l,
            i
        ])

    if len(edges) == 0:

        return torch.empty(
            (2, 0),
            dtype=torch.long
        )

    return torch.tensor(
        edges,
        dtype=torch.long
    ).T


# =========================================================
# BUILD GRAPH + SAVE PYG
# =========================================================

def mols2graphs(
    complex_path,
    label,
    save_path,
    dis_threshold=5.0
):

    try:

        # =====================================================
        # LOAD RDKIT COMPLEX
        # SAVED FORMAT:
        # (ligand, pocket, surface_features)
        # =====================================================

        with open(complex_path, 'rb') as f:

            ligand, pocket, surface_features = pickle.load(f)

        # =====================================================
        # ORIGINAL ARTICLE GRAPH
        # =====================================================

        x_l, edge_l = mol2graph(ligand)

        x_p, edge_p = mol2graph(pocket)

        # =====================================================
        # SURFACE FEATURES
        # shape:
        # [N_pocket_atoms, 7]
        # =====================================================

        surf_feat = torch.tensor(
            surface_features,
            dtype=torch.float32
        )

        # =====================================================
        # ADD SURFACE FEATURES
        # ONLY TO POCKET ATOMS
        # =====================================================

        surf_dim = surf_feat.shape[1]

        # ligand gets zero padding
        ligand_pad = torch.zeros(
            (x_l.shape[0], surf_dim),
            dtype=torch.float32
        )

        x_l = torch.cat(
            [x_l, ligand_pad],
            dim=1
        )

        # pocket gets REAL surface features
        x_p = torch.cat(
            [x_p, surf_feat],
            dim=1
        )

        # safety check
        assert x_l.shape[1] == x_p.shape[1]

        # =====================================================
        # CONCAT ALL NODES
        # =====================================================

        x = torch.cat(
            [x_l, x_p],
            dim=0
        )

        # =====================================================
        # INTRA MOLECULAR EDGES
        # =====================================================

        edge_index_intra = torch.cat(
            [
                edge_l,
                edge_p + ligand.GetNumAtoms()
            ],
            dim=1
        )

        # =====================================================
        # INTERACTION EDGES
        # =====================================================

        edge_index_inter = inter_graph(
            ligand,
            pocket,
            dis_threshold
        )

        # =====================================================
        # LABEL
        # =====================================================

        y = torch.tensor(
            [float(label)],
            dtype=torch.float32
        )

        # =====================================================
        # POSITIONS
        # =====================================================

        pos = torch.tensor(

            np.vstack([
                ligand.GetConformers()[0].GetPositions(),
                pocket.GetConformers()[0].GetPositions()
            ]),

            dtype=torch.float32
        )

        # =====================================================
        # SPLIT MASK
        # 0 -> ligand
        # 1 -> pocket
        # =====================================================

        split = torch.cat([

            torch.zeros(
                ligand.GetNumAtoms(),
                dtype=torch.long
            ),

            torch.ones(
                pocket.GetNumAtoms(),
                dtype=torch.long
            )

        ])

        # =====================================================
        # FINAL DATA OBJECT
        # =====================================================

        data = Data(

            x=x,

            edge_index_intra=edge_index_intra,

            edge_index_inter=edge_index_inter,

            y=y,

            pos=pos,

            split=split

        )

        torch.save(data, save_path)

        print(f"[SAVED] {save_path}")

    except Exception as e:

        print(f"[ERROR] {complex_path}: {e}")


# =========================================================
# DATASET
# =========================================================

class GraphDataset(Dataset):

    def __init__(
        self,
        data_dir,
        data_df,
        dis_threshold=5,
        create=False,
        num_process=6
    ):

        super().__init__()

        self.data_dir = data_dir

        self.df = data_df

        self.dis_threshold = dis_threshold

        self.create = create

        self.num_process = num_process

        self.rdkit_paths = []

        self.graph_paths = []

        self.labels = []

        self._prepare()

    # =====================================================
    # REQUIRED BY PYG
    # =====================================================

    def len(self):

        return len(self.graph_paths)

    def get(self, idx):

        return torch.load(
            self.graph_paths[idx]
        )

    # =====================================================
    # PREPARE DATA
    # =====================================================

    def _prepare(self):

        for _, row in self.df.iterrows():

            cid = row['pdbid']

            label = float(
                row['-logKd/Ki']
            )

            complex_dir = os.path.join(
                self.data_dir,
                cid
            )

            rdkit_path = os.path.join(
                complex_dir,
                f"{cid}_5A.rdkit"
            )

            graph_path = os.path.join(
                complex_dir,
                f"Graph_{cid}_5A.pyg"
            )

            if not os.path.exists(rdkit_path):
                continue

            self.rdkit_paths.append(
                rdkit_path
            )

            self.graph_paths.append(
                graph_path
            )

            self.labels.append(label)

        print(
            f"\nTotal complexes: "
            f"{len(self.graph_paths)}"
        )

        # =====================================================
        # CREATE PYG FILES
        # =====================================================

        if self.create:

            print(
                "\nCreating graph files...\n"
            )

            pool = multiprocessing.Pool(
                self.num_process
            )

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

            print(
                "\nGraph generation finished.\n"
            )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    data_dir = (
        "/Volumes/Ventoy/MolGen_Project copy/"
        "data/dataset/selected_107"
    )

    csv_path = (

        "/Volumes/Ventoy/MolGen_Project copy/test2013.csv"

    )

    df = pd.read_csv(csv_path)

    dataset = GraphDataset(

        data_dir=data_dir,

        data_df=df,

        dis_threshold=5.0,

        create=True,

        num_process=6

    )

    print(
        f"\nDataset size: "
        f"{dataset.len()}"
    )

# %%