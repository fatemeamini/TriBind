# interaction_parser.py
# Build a combined protein–ligand interaction graph from:
#   - ligand SDF (with 3D coords)
#   - protein pocket PDB
#
# Output: torch_geometric.data.Data with:
#   x: [N, 26] node features (ligand atoms + protein residues)
#   edge_index: [2, E]
#   edge_attr: [E, 9] edge features (consistent across edge types)
#   node_type: [N] 0=ligand, 1=protein
#   edge_type: [E] 0=ligand, 1=protein, 2=interaction
#
# Key fixes vs your current script:
# - Removes duplicate imports / duplicate typing declarations
# - Ensures data.protein_coords is ALWAYS set (not only when mol_attr exists)
# - Removes unused conf = lig_mol.GetConformer()
# - Adds CLI so you can run it as a script safely (no hardcoded paths)
# - Wraps example usage under if __name__ == "__main__"
# - Robust reading + clearer errors
# - Optional edge coloring in visualization by interaction type (hbond/vdw/electro/hydrophobic)

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import argparse
import numpy as np
import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

from Bio.PDB import PDBParser

# Silence RDKit warnings (including 2D/3D tagging warnings)
RDLogger.DisableLog("rdApp.warning")

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None


# ================================================================
#  Protein residue feature library (copied/adapted from protein_parser.py)
# ================================================================

RESIDUE_LIST = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
]

AA_MOTIF = {
    "ACCEPTOR": ["ASP", "GLU", "HIS"],
    "DONOR": ["ARG", "LYS", "HIS", "TRP"],
    "BOTH": ["SER", "THR", "TYR", "ASN", "GLN"],
    "ALIPHATIC": ["ALA", "ILE", "LEU", "VAL", "MET", "CYS"],
    "AROMATIC": ["PHE", "TYR", "TRP"],
    "NONE": ["GLY", "PRO"]
}

AA_WEIGHTS = {
    "ALA": 89.1, "ARG": 174.2, "ASN": 132.1, "ASP": 133.1, "CYS": 121.2,
    "GLN": 146.2, "GLU": 147.1, "GLY": 75.1, "HIS": 155.2, "ILE": 131.2,
    "LEU": 131.2, "LYS": 146.2, "MET": 149.2, "PHE": 165.2, "PRO": 115.1,
    "SER": 105.1, "THR": 119.1, "TRP": 204.2, "TYR": 181.2, "VAL": 117.1
}

AA_VOLUME = {
    "ALA": 88.6, "ARG": 173.4, "ASN": 114.1, "ASP": 111.1, "CYS": 108.5,
    "GLN": 143.8, "GLU": 138.4, "GLY": 60.1, "HIS": 153.2, "ILE": 166.7,
    "LEU": 166.7, "LYS": 168.6, "MET": 162.9, "PHE": 189.9, "PRO": 112.7,
    "SER": 89.0, "THR": 116.1, "TRP": 227.8, "TYR": 193.6, "VAL": 140.0
}

AA_ATOM_COUNTS = {
    "ALA": [3, 1, 1, 0, 5],
    "ARG": [6, 4, 1, 0, 12],
    "ASN": [4, 2, 2, 0, 6],
    "ASP": [4, 1, 3, 0, 5],
    "CYS": [3, 1, 1, 1, 5],
    "GLN": [5, 2, 2, 0, 8],
    "GLU": [5, 1, 3, 0, 7],
    "GLY": [2, 1, 1, 0, 3],
    "HIS": [6, 3, 1, 0, 7],
    "ILE": [6, 1, 1, 0, 11],
    "LEU": [6, 1, 1, 0, 11],
    "LYS": [6, 2, 1, 0, 12],
    "MET": [5, 1, 1, 1, 9],
    "PHE": [9, 1, 1, 0, 9],
    "PRO": [5, 1, 1, 0, 7],
    "SER": [3, 1, 2, 0, 5],
    "THR": [4, 1, 2, 0, 7],
    "TRP": [11, 2, 1, 0, 10],
    "TYR": [9, 1, 2, 0, 9],
    "VAL": [5, 1, 1, 0, 9]
}

AA_TPSA = {
    "ALA": 63.3, "ARG": 126.0, "ASN": 106.0, "ASP": 100.0, "CYS": 63.3,
    "GLN": 106.0, "GLU": 100.0, "GLY": 63.3, "HIS": 88.0, "ILE": 63.3,
    "LEU": 63.3, "LYS": 89.0, "MET": 63.3, "PHE": 63.3, "PRO": 49.3,
    "SER": 83.6, "THR": 83.6, "TRP": 79.1, "TYR": 83.6, "VAL": 63.3
}

AA_HBOND_COUNTS = {
    "ALA": [2, 3], "ARG": [4, 4], "ASN": [3, 4], "ASP": [3, 5], "CYS": [2, 3],
    "GLN": [3, 4], "GLU": [3, 5], "GLY": [2, 3], "HIS": [3, 3], "ILE": [2, 3],
    "LEU": [2, 3], "LYS": [3, 3], "MET": [2, 3], "PHE": [2, 3], "PRO": [1, 2],
    "SER": [3, 4], "THR": [3, 4], "TRP": [3, 3], "TYR": [3, 4], "VAL": [2, 3]
}

AA_MAX_SASA = {
    "ALA": 121.0, "ARG": 265.0, "ASN": 187.0, "ASP": 187.0, "CYS": 148.0,
    "GLN": 214.0, "GLU": 214.0, "GLY": 97.0, "HIS": 216.0, "ILE": 195.0,
    "LEU": 191.0, "LYS": 230.0, "MET": 203.0, "PHE": 228.0, "PRO": 154.0,
    "SER": 143.0, "THR": 163.0, "TRP": 264.0, "TYR": 255.0, "VAL": 165.0
}


def motif_vector(resname: str) -> np.ndarray:
    r = resname.upper()
    return np.array([
        int(r in AA_MOTIF["ACCEPTOR"]),
        int(r in AA_MOTIF["DONOR"]),
        int(r in AA_MOTIF["BOTH"]),
        int(r in AA_MOTIF["ALIPHATIC"]),
        int(r in AA_MOTIF["AROMATIC"]),
        int(r in AA_MOTIF["NONE"])
    ], dtype=float)


def compute_residue_atomic_features(resname: str) -> np.ndarray:
    counts = AA_ATOM_COUNTS.get(resname, [0, 0, 0, 0, 0])
    weight = AA_WEIGHTS.get(resname, 0.0)
    volume = AA_VOLUME.get(resname, 0.0)
    density = weight / volume if volume > 0 else 0.0
    is_aromatic = 1.0 if resname in ["HIS", "TYR", "TRP", "PHE"] else 0.0

    tpsa = AA_TPSA.get(resname, 0.0)
    hb_donor, hb_acceptor = AA_HBOND_COUNTS.get(resname, [0, 0])
    ref_sasa = AA_MAX_SASA.get(resname, 0.0)

    return np.array([
        counts[0], counts[1], counts[2], counts[3], counts[4],
        weight, volume, density, is_aromatic,
        tpsa, hb_donor, hb_acceptor, ref_sasa
    ], dtype=float)


# ================================================================
#  Simple interaction typing (rule-based)
# ================================================================

HYDROPHOBIC_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
POS_RES = {"ARG", "LYS"}
NEG_RES = {"ASP", "GLU"}


def classify_ligand_atom(atom: Chem.Atom) -> Dict[str, bool]:
    z = atom.GetAtomicNum()
    return {
        "is_donor_acceptor": z in (7, 8),           # N, O
        "is_hydrophobic": z in (6, 16, 17, 35, 53)  # C, S, halogens (coarse)
    }


def residue_charge_sign(resname: str) -> int:
    if resname in POS_RES:
        return 1
    if resname in NEG_RES:
        return -1
    return 0


def edge_interaction_onehot(dist: float, lig_atom: Optional[Chem.Atom], resname: Optional[str]) -> List[int]:
    """
    Returns [hbond, vdw, electrostatic, hydrophobic]
    """
    if lig_atom is None or resname is None:
        return [0, 1, 0, 0]

    lig_info = classify_ligand_atom(lig_atom)
    ch = residue_charge_sign(resname)

    lig_fc = lig_atom.GetFormalCharge()
    if ch != 0 and lig_fc != 0 and ch * lig_fc < 0:
        return [0, 0, 1, 0]

    if dist <= 3.6 and lig_info["is_donor_acceptor"] and resname in (
        AA_MOTIF["DONOR"] + AA_MOTIF["ACCEPTOR"] + AA_MOTIF["BOTH"]
    ):
        return [1, 0, 0, 0]

    if dist <= 4.5 and lig_info["is_hydrophobic"] and resname in HYDROPHOBIC_RES:
        return [0, 0, 0, 1]

    return [0, 1, 0, 0]


def residue_residue_onehot(resA: str, resB: str) -> List[int]:
    chargeA = residue_charge_sign(resA)
    chargeB = residue_charge_sign(resB)
    if chargeA * chargeB < 0:
        return [0, 0, 1, 0]

    if (resA in HYDROPHOBIC_RES) and (resB in HYDROPHOBIC_RES):
        return [0, 0, 0, 1]

    if (resA in (AA_MOTIF["DONOR"] + AA_MOTIF["ACCEPTOR"] + AA_MOTIF["BOTH"])) and \
       (resB in (AA_MOTIF["DONOR"] + AA_MOTIF["ACCEPTOR"] + AA_MOTIF["BOTH"])):
        return [1, 0, 0, 0]

    return [0, 1, 0, 0]


# ================================================================
#  Protein pocket PDB -> residue pseudo-atoms
# ================================================================

@dataclass
class ProteinPocket:
    res_ids: List[Tuple[str, int, str]]   # (chain, resseq, resname)
    coords: np.ndarray                    # [R, 3]
    x: np.ndarray                         # [R, 26]


def load_protein_pocket_from_pdb(pdb_file: str) -> ProteinPocket:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    res_ids: List[Tuple[str, int, str]] = []
    coords: List[np.ndarray] = []
    feats: List[np.ndarray] = []

    model = structure[0]
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            resname = residue.get_resname()
            if resname not in RESIDUE_LIST:
                continue

            if "CA" in residue:
                c = residue["CA"].coord.astype(float)
            else:
                atom_xyz = [a.coord.astype(float) for a in residue.get_atoms()]
                if len(atom_xyz) == 0:
                    continue
                c = np.mean(np.stack(atom_xyz, axis=0), axis=0)

            surf_zeros = np.zeros(7, dtype=float)
            atomic = compute_residue_atomic_features(resname)
            motif = motif_vector(resname)
            x = np.concatenate([surf_zeros, atomic, motif], axis=0)  # 26

            res_ids.append((chain.id, int(residue.id[1]), resname))
            coords.append(c)
            feats.append(x)

    if len(res_ids) == 0:
        raise ValueError(f"No standard residues found in pocket PDB: {pdb_file}")

    return ProteinPocket(
        res_ids=res_ids,
        coords=np.stack(coords, axis=0),
        x=np.stack(feats, axis=0),
    )


# ================================================================
#  Main: build combined interaction graph
# ================================================================

def _unit_direction(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = (b - a).astype(float)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros(3, dtype=float)
    return v / n


def _read_ligand_mol(sdf_path: str) -> Chem.Mol:
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    if suppl is None or len(suppl) == 0 or suppl[0] is None:
        raise ValueError(f"Failed to read ligand SDF: {sdf_path}")
    mol = suppl[0]
    if mol.GetNumAtoms() == 0:
        raise ValueError(f"Ligand SDF has zero atoms: {sdf_path}")

    # Ensure conformer treated as 3D if any z != 0 (silences some tag weirdness)
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        is3d = any(abs(conf.GetAtomPosition(i).z) > 1e-6 for i in range(mol.GetNumAtoms()))
        if is3d:
            conf.Set3D(True)

    return mol


def build_interaction_graph(
    ligand_sdf_path: str,
    protein_pdb_path: str,
    cutoff: float = 6.0,
) -> Data:
    """
    Returns a single torch_geometric Data object for the full interaction graph.
    """
    # Import inside to match your project structure
    from mol_parser import sdf_to_graph

    lig_mol = _read_ligand_mol(ligand_sdf_path)

    lig_graph = sdf_to_graph(lig_mol)
    lig_x = lig_graph.x.detach().cpu().numpy()
    lig_edge_index = lig_graph.edge_index.detach().cpu().numpy()
    lig_edge_attr = np.array(lig_graph.edge_attr.detach().cpu(), dtype=np.float32)
    L = lig_x.shape[0]

    # Ligand atom xyz are last 3 dims in your atom feature
    lig_xyz = lig_x[:, -3:]

    pocket = load_protein_pocket_from_pdb(protein_pdb_path)
    prot_x = pocket.x
    prot_xyz = pocket.coords
    R = prot_x.shape[0]

    lig_x = np.array(lig_x, dtype=np.float32)
    prot_x = np.array(prot_x, dtype=np.float32)
    x = np.concatenate([lig_x, prot_x], axis=0)
    node_type = np.concatenate([np.zeros(L, dtype=np.int64), np.ones(R, dtype=np.int64)], axis=0)

    # --- Protein-protein edges (within cutoff) ---
    prot_edges: List[List[int]] = []
    prot_eattr: List[List[float]] = []

    for i in range(R):
        for j in range(i + 1, R):
            dist = float(np.linalg.norm(prot_xyz[i] - prot_xyz[j]))
            if dist <= cutoff:
                u = L + i
                v = L + j
                dx, dy, dz = _unit_direction(prot_xyz[i], prot_xyz[j]).tolist()
                hbond, vdw, electro, hphob = residue_residue_onehot(pocket.res_ids[i][2], pocket.res_ids[j][2])
                normal_sim = 0.0
                feat = [dist, hbond, vdw, electro, hphob, dx, dy, dz, normal_sim]
                prot_edges += [[u, v], [v, u]]
                prot_eattr += [feat, feat]

    # --- Ligand-protein interaction edges (within cutoff) ---
    inter_edges: List[List[int]] = []
    inter_eattr: List[List[float]] = []

    for a_idx in range(L):
        atom = lig_mol.GetAtomWithIdx(a_idx)
        a_pos = lig_xyz[a_idx]

        for r_idx in range(R):
            r_pos = prot_xyz[r_idx]
            dist = float(np.linalg.norm(a_pos - r_pos))
            if dist <= cutoff:
                u = a_idx
                v = L + r_idx
                dx, dy, dz = _unit_direction(a_pos, r_pos).tolist()
                hbond, vdw, electro, hphob = edge_interaction_onehot(dist, atom, pocket.res_ids[r_idx][2])
                normal_sim = 0.0
                feat = [dist, hbond, vdw, electro, hphob, dx, dy, dz, normal_sim]
                inter_edges += [[u, v], [v, u]]
                inter_eattr += [feat, feat]

    # --- Combine all edges ---
    all_edge_index: List[List[int]] = []
    all_edge_attr: List[List[float]] = []
    edge_type: List[int] = []

    # 0) ligand intramolecular edges (from mol_parser)
    for k in range(lig_edge_index.shape[1]):
        src = int(lig_edge_index[0, k])
        dst = int(lig_edge_index[1, k])
        ea = lig_edge_attr[k]
        if ea.shape[0] < 9:
            ea = np.pad(ea, (0, 9 - ea.shape[0]))
        all_edge_index.append([src, dst])
        all_edge_attr.append(ea.tolist())
        edge_type.append(0)

    # 1) protein intramolecular edges
    for e, ea in zip(prot_edges, prot_eattr):
        all_edge_index.append(e)
        all_edge_attr.append(ea)
        edge_type.append(1)

    # 2) interaction edges
    for e, ea in zip(inter_edges, inter_eattr):
        all_edge_index.append(e)
        all_edge_attr.append(ea)
        edge_type.append(2)

    if len(all_edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 9), dtype=torch.float32)
        edge_type_t = torch.empty((0,), dtype=torch.long)
    else:
        edge_index = torch.tensor(np.array(all_edge_index).T, dtype=torch.long).contiguous()
        edge_attr = torch.tensor(np.array(all_edge_attr), dtype=torch.float32)
        edge_type_t = torch.tensor(np.array(edge_type), dtype=torch.long)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_type=torch.tensor(node_type, dtype=torch.long),
        edge_type=edge_type_t,
    )

    # Metadata for mapping explanations back
    data.ligand_num_nodes = L
    data.protein_num_nodes = R
    data.protein_res_ids = pocket.res_ids
    data.protein_coords = prot_xyz  # ALWAYS set

    if hasattr(lig_graph, "mol_attr"):
        data.mol_attr = lig_graph.mol_attr

    return data


def interaction_to_graph(ligand_sdf_path: str, protein_pdb_path: str, cutoff: float = 6.0) -> Data:
    return build_interaction_graph(ligand_sdf_path, protein_pdb_path, cutoff=cutoff)


# ================================================================
#  2D visualization
# ================================================================

def _pca_2d(coords3d: np.ndarray) -> np.ndarray:
    coords3d = np.asarray(coords3d, dtype=float)
    if coords3d.shape[0] == 0:
        return coords3d[:, :2]
    if PCA is None:
        return coords3d[:, :2]
    return PCA(n_components=2).fit_transform(coords3d)


def _normalize_to_box(xy: np.ndarray, target_center=(0.0, 0.0), target_scale=1.0) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if xy.shape[0] == 0:
        return xy
    mean = xy.mean(axis=0, keepdims=True)
    xy0 = xy - mean
    denom = np.max(np.linalg.norm(xy0, axis=1))
    if denom < 1e-8:
        denom = 1.0
    xy1 = xy0 / denom * target_scale
    xy1 += np.array(target_center)[None, :]
    return xy1


def visualize_interaction_2d(
    data: Data,
    lig_mol: Chem.Mol,
    protein_res_ids: List[Tuple[str, int, str]],
    out_path: Optional[str] = None,
    show_protein_protein: bool = False,
    show_ligand_ligand: bool = True,
    show_interactions: bool = True,
    label_ligand: str = "index",     # "index" or "elem"
    label_residue: str = "full",     # "full" or "num"
    max_interaction_edges: Optional[int] = 300,
    color_interactions: bool = True, # if True, color by edge_attr type
):
    import matplotlib.pyplot as plt

    L = int(getattr(data, "ligand_num_nodes", 0) or 0)
    if L == 0 and hasattr(data, "node_type"):
        L = int((data.node_type == 0).sum().item())
    N = int(data.x.shape[0])
    R = N - L

    # Ligand coords from RDKit 2D depiction
    mol2d = Chem.Mol(lig_mol)
    AllChem.Compute2DCoords(mol2d)
    conf2d = mol2d.GetConformer()
    lig_xy = np.array([[conf2d.GetAtomPosition(i).x, conf2d.GetAtomPosition(i).y] for i in range(L)], dtype=float)
    lig_xy = _normalize_to_box(lig_xy, target_center=(0.0, 0.0), target_scale=1.0)

    # Protein coords from PCA projection of residue 3D coords
    prot_coords3d = np.asarray(getattr(data, "protein_coords", np.zeros((R, 3), dtype=float)))
    if prot_coords3d.shape[0] == R and R > 0:
        prot_xy = _pca_2d(prot_coords3d)
        prot_xy = _normalize_to_box(prot_xy, target_center=(2.2, 0.0), target_scale=1.0)
    else:
        theta = np.linspace(0, 2 * np.pi, max(R, 1), endpoint=False)
        prot_xy = np.stack([np.cos(theta), np.sin(theta)], axis=1) * 1.0 + np.array([2.2, 0.0])[None, :]

    all_xy = np.zeros((N, 2), dtype=float)
    all_xy[:L] = lig_xy
    all_xy[L:] = prot_xy

    ei = data.edge_index.detach().cpu().numpy()
    et = getattr(data, "edge_type", None)
    if et is None:
        edge_type = np.full((ei.shape[1],), 2, dtype=np.int64)
    else:
        edge_type = et.detach().cpu().numpy().astype(np.int64)

    ea = getattr(data, "edge_attr", None)
    if ea is not None:
        edge_attr = ea.detach().cpu().numpy()
    else:
        edge_attr = None

    idx = np.arange(ei.shape[1])
    if max_interaction_edges is not None and max_interaction_edges > 0:
        inter_mask = (edge_type == 2)
        inter_idx = idx[inter_mask]
        if inter_idx.size > max_interaction_edges:
            inter_idx = inter_idx[:max_interaction_edges]
        keep_mask = (~inter_mask)
        keep_mask[inter_idx] = True
        idx = idx[keep_mask]

    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.axis("off")

    # Interaction edge colors based on [hbond, vdw, electro, hydrophobic] = edge_attr[1:5]
    def interaction_style(k: int):
        if not color_interactions or edge_attr is None:
            return ("--", 1.4, 0.8, None)
        t = edge_attr[k, 1:5]  # hbond, vdw, electro, hydrophobic
        if t.shape[0] != 4:
            return ("--", 1.4, 0.8, None)
        # choose label by argmax
        j = int(np.argmax(t))
        # Let matplotlib choose default cycle if color=None; we just vary linestyle/alpha/width slightly
        if j == 0:  # hbond
            return ("--", 2.0, 0.9, None)
        if j == 2:  # electro
            return ("-.", 2.0, 0.9, None)
        if j == 3:  # hydrophobic
            return ("--", 1.8, 0.8, None)
        # vdw
        return ("--", 1.2, 0.6, None)

    # Draw edges
    for k in idx:
        u, v = int(ei[0, k]), int(ei[1, k])
        x0, y0 = all_xy[u]
        x1, y1 = all_xy[v]

        t = edge_type[k]
        if t == 0 and show_ligand_ligand:
            ax.plot([x0, x1], [y0, y1], linestyle="-", linewidth=1.6, alpha=0.7)
        elif t == 1 and show_protein_protein:
            ax.plot([x0, x1], [y0, y1], linestyle=":", linewidth=1.0, alpha=0.25)
        elif t == 2 and show_interactions:
            ls, lw, al, col = interaction_style(k)
            ax.plot([x0, x1], [y0, y1], linestyle=ls, linewidth=lw, alpha=al)

    # Draw nodes
    ax.scatter(all_xy[:L, 0], all_xy[:L, 1], s=60, marker="o")
    ax.scatter(all_xy[L:, 0], all_xy[L:, 1], s=80, marker="s")

    # Labels
    for i in range(L):
        if label_ligand == "elem":
            atom = lig_mol.GetAtomWithIdx(i)
            txt = f"{atom.GetSymbol()}{i}"
        else:
            txt = str(i)
        ax.text(all_xy[i, 0], all_xy[i, 1], txt, fontsize=9, ha="center", va="center")

    for j in range(R):
        chain, resseq, resname = protein_res_ids[j]
        txt = str(resseq) if label_residue == "num" else f"{chain}:{resname}{resseq}"
        ax.text(all_xy[L + j, 0], all_xy[L + j, 1], txt, fontsize=8, ha="center", va="center")

    ax.text(-1.6, 1.3, "Ligand atoms (o)  |  Protein residues (s)", fontsize=11)
    ax.text(-1.6, 1.15, "Edges: ligand-bonds (-), interactions (--/-.), protein-protein (:, optional)", fontsize=10)

    if out_path is not None:
        plt.savefig(out_path, dpi=250, bbox_inches="tight")
    return fig


# ================================================================
#  CLI / Script entrypoint
# ================================================================

def main():
    ap = argparse.ArgumentParser(description="Build interaction graph and optionally render a 2D visualization.")
    ap.add_argument("--ligand_sdf", required=True, help="Path to ligand SDF")
    ap.add_argument("--pocket_pdb", required=True, help="Path to pocket PDB")
    ap.add_argument("--cutoff", type=float, default=6.0, help="Distance cutoff (Å) for protein-protein and ligand-protein edges")
    ap.add_argument("--out_png", default=None, help="If provided, write 2D visualization to this PNG file")
    ap.add_argument("--show_protein_protein", action="store_true", help="Draw protein-protein edges in the visualization")
    ap.add_argument("--max_interaction_edges", type=int, default=300, help="Max interaction edges to draw in visualization")
    args = ap.parse_args()

    data = interaction_to_graph(args.ligand_sdf, args.pocket_pdb, cutoff=args.cutoff)
    print(data)

    if args.out_png:
        lig_mol = _read_ligand_mol(args.ligand_sdf)
        visualize_interaction_2d(
            data,
            lig_mol=lig_mol,
            protein_res_ids=data.protein_res_ids,
            out_path=args.out_png,
            show_protein_protein=args.show_protein_protein,
            max_interaction_edges=args.max_interaction_edges,
        )
        print(f"Saved 2D visualization to: {args.out_png}")


if __name__ == "__main__":
    main()
