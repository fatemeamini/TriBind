import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch.nn as nn
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
import umap
from torch_geometric.data import Data, Batch
import warnings

warnings.filterwarnings('ignore')

# ============================================================
#         ELECTRONEGATIVITY (Pauling scale) MAP
# ============================================================
ELECTRONEGATIVITY_MAP = {
    1: 2.20,  # H
    6: 2.55,  # C
    7: 3.04,  # N
    8: 3.44,  # O
    9: 3.98,  # F
    15: 2.19, # P
    16: 2.58, # S
    17: 3.16, # Cl
    35: 2.96, # Br
    53: 2.66, # I
    # Add other common elements as needed
}

# ============================================================
#                     ATOM FEATURES
# ============================================================

def get_atom_features(mol, conf):
    atom_features = []

    # Mapping tables (unchanged)
    formal_charge_map = [-3, -2, -1, 0, 1, 2, 3]
    hybrid_map = {
        Chem.rdchem.HybridizationType.S: 0,
        Chem.rdchem.HybridizationType.SP: 1,
        Chem.rdchem.HybridizationType.SP2: 2,
        Chem.rdchem.HybridizationType.SP3: 3,
        Chem.rdchem.HybridizationType.SP2D: 4,
        Chem.rdchem.HybridizationType.SP3D: 5,
        Chem.rdchem.HybridizationType.SP3D2: 6
    }

    for i, atom in enumerate(mol.GetAtoms()):
        atomic_num = atom.GetAtomicNum()

        base = [
            atomic_num,
            atom.GetMass(),
            atom.GetDegree(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetTotalValence(),
            atom.GetNumRadicalElectrons(),
            atom.GetTotalNumHs()
        ]

        # 1. NEW: Electronegativity
        en = [ELECTRONEGATIVITY_MAP.get(atomic_num, 0.0)]

        # 2. Formal charge – one-hot (unchanged)
        fc = [0] * 7
        charge = atom.GetFormalCharge()
        if charge in formal_charge_map:
            fc[formal_charge_map.index(charge)] = 1

        # 3. Hybridization – one-hot (unchanged)
        hyb = [0] * 7
        h = atom.GetHybridization()
        if h in hybrid_map:
            hyb[hybrid_map[h]] = 1
            
        # 4. NEW: X, Y, Z Coordinates (requires conformer)
        if conf:
            pos = conf.GetAtomPosition(i)
            xyz = [pos.x, pos.y, pos.z]
        else:
            xyz = [0.0, 0.0, 0.0]

        # Final Node Feature vector
        atom_features.append(base + en + fc + hyb + xyz)

    return atom_features


# ============================================================
#                     BOND LENGTHS
# ============================================================

def compute_bond_length(mol):
    try:
        conf = mol.GetConformer()
    except:
        return []

    bond_lengths = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        pi = np.array(conf.GetAtomPosition(i))
        pj = np.array(conf.GetAtomPosition(j))
        bond_lengths.append(np.linalg.norm(pi - pj))

    return bond_lengths

# ============================================================
#                     ANGLE CALCULATIONS
# ============================================================

def calculate_angle(conf, i, j, k):
    """Calculates the bond angle (in degrees) between atoms i-j-k."""
    try:
        pi = np.array(conf.GetAtomPosition(i))
        pj = np.array(conf.GetAtomPosition(j))
        pk = np.array(conf.GetAtomPosition(k))
        
        vec_ji = pi - pj
        vec_jk = pk - pj
        
        # Ensure vectors are non-zero
        if np.linalg.norm(vec_ji) == 0 or np.linalg.norm(vec_jk) == 0:
            return 0.0

        dot_product = np.dot(vec_ji, vec_jk)
        
        # Clamp value to prevent domain error in arccos due to float precision
        cosine_angle = np.clip(dot_product / (np.linalg.norm(vec_ji) * np.linalg.norm(vec_jk)), -1.0, 1.0)
        
        angle_rad = np.arccos(cosine_angle)
        return np.degrees(angle_rad)
    except Exception:
        return 0.0

def calculate_dihedral(conf, i, j, k, l):
    """Calculates the dihedral angle (in degrees) between atoms i-j-k-l."""
    try:
        pi = np.array(conf.GetAtomPosition(i))
        pj = np.array(conf.GetAtomPosition(j))
        pk = np.array(conf.GetAtomPosition(k))
        pl = np.array(conf.GetAtomPosition(l))
        
        # Use RDKit's built-in function for robust dihedral calculation
        return Chem.rdMolTransforms.GetDihedralDeg(conf, i, j, k, l)
    except Exception:
        return 0.0
    
# ============================================================
#                     BOND FEATURES
# ============================================================

def get_bond_features(bond, bond_length):

    # Bond type (one-hot)
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        bond_type = [1, 0, 0]
    elif bt == Chem.rdchem.BondType.DOUBLE:
        bond_type = [0, 1, 0]
    elif bt == Chem.rdchem.BondType.TRIPLE:
        bond_type = [0, 0, 1]
    else:
        bond_type = [0, 0, 0]

    # Stereochemistry (cis/trans)
    stereo = bond.GetStereo()
    if stereo == Chem.rdchem.BondStereo.STEREOZ:
        stereo_feat = [1, 0]
    elif stereo == Chem.rdchem.BondStereo.STEREOE:
        stereo_feat = [0, 1]
    else:
        stereo_feat = [0, 0]

    # Conjugation
    conj = [1] if bond.GetIsConjugated() else [0]

    return bond_type + stereo_feat + conj + [bond_length]

# ============================================================
#                  GLOBAL MOLECULE FEATURES
# ============================================================

def get_global_features(mol):
    """
    Calculates 6 molecular level properties: TPSA, H-bond counts, Charge, Volume, Density.
    """
    
    # Temporarily remove H's to get heavy-atom only structure for descriptor calculation
    mol_no_h = Chem.RemoveHs(mol) 
    
    # 1. TPSA
    tpsa = rdMolDescriptors.CalcTPSA(mol_no_h)

    # 2. H-bond Donor and Acceptor Counts
    hbd = Descriptors.NumHDonors(mol_no_h)
    hba = Descriptors.NumHAcceptors(mol_no_h)
    
    # 3. Overall Charge (calculated on the original molecule)
    charge = Chem.rdmolops.GetFormalCharge(mol) 
    
    # 4. Volume
    try:
        volume = AllChem.ComputeMolVolume(mol_no_h)
    except Exception:
        volume = 0.0

    # 5. Mass and Density
    mass = Descriptors.MolWt(mol_no_h) 
    density = mass / volume if volume > 0.1 else 0.0
    
    # total_H calculation removed here
    
    # === LIST NOW CONTAINS 6 ITEMS ===
    global_features = [
        tpsa,
        hbd,
        hba,
        charge,
        volume,
        density,
        # total_H REMOVED
    ]
    
    # Return original molecule object for subsequent atom/bond feature extraction
    return torch.tensor(global_features, dtype=torch.float).unsqueeze(0), mol

# ============================================================
#                  SMILES → Graph converter
# ============================================================

def smiles_to_graph(smiles):
    """
    Converts SMILES → PyTorch Geometric Data object
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Add hydrogens and compute 3D conformation (Needed for X,Y,Z & bond lengths)
    mol = Chem.AddHs(mol)

    try:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as e:
        print("3D embedding failed:", e)
        # Fallback to 2D
        conf = None
        mol_attr, mol = get_global_features(mol)
    
    try:
        conf = mol.GetConformer()
    except:
        conf = None

    # 1. NEW: Global Molecular features
    mol_attr, mol = get_global_features(mol)

    # 2. Node features
    atom_features = get_atom_features(mol, conf)
    x = torch.tensor(atom_features, dtype=torch.float)

    # Edges (Unchanged logic)
    edge_index = []
    edge_attr = []

    bond_lengths = compute_bond_length(mol) # uses the existing conformer (conf)

    for bond, bl in zip(mol.GetBonds(), bond_lengths):
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Undirected graph → add both directions
        edge_index += [[i, j], [j, i]]

        bf = get_bond_features(bond, bl)
        edge_attr += [bf, bf]

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        # 1 (bond length) + 3 (type) + 2 (stereo) + 1 (conj) = 7 features
        edge_attr = torch.zeros((0, 7), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        mol_attr=mol_attr # NEW: Global Molecular Features
    )

# ============================================================
#                  SDF → Graph converter
# ============================================================

def sdf_to_graph(mol):
    """
    Converts an RDKit Mol loaded from SDF file into a PyTorch Geometric graph
    WITHOUT modifying atom positions.
    """

    if mol is None:
        raise ValueError("Input mol is None.")

    try:
        conf = mol.GetConformer()
    except:
        raise ValueError("SDF molecule has no conformer (no coordinates).")
        
    # 1. NEW: Global Molecular features
    mol_attr, mol = get_global_features(mol)
    
    # --------- Node features ----------
    atom_features = get_atom_features(mol, conf) # Pass the conformer
    x = torch.tensor(atom_features, dtype=torch.float)

    # --------- Edge features ----------
    # Edges
    edge_index = []
    edge_attr = []

    bond_lengths = compute_bond_length(mol)

    for idx, (bond, bl) in enumerate(zip(mol.GetBonds(), bond_lengths)):
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        atom_i = mol.GetAtomWithIdx(i)
        atom_j = mol.GetAtomWithIdx(j)

        # ------------------------------------------------
        # 1. Calculate Angular Features
        # ------------------------------------------------
        
        # A. Bond Angle (X-i-j) and (i-j-Y)
        avg_bond_angle = 0.0
        angle_count = 0
        
        # Iterate over neighbors of atom i (excluding atom j)
        for neighbor_k in [n.GetIdx() for n in atom_i.GetNeighbors() if n.GetIdx() != j]:
            avg_bond_angle += calculate_angle(conf, neighbor_k, i, j)
            angle_count += 1
            
        # Iterate over neighbors of atom j (excluding atom i)
        for neighbor_k in [n.GetIdx() for n in atom_j.GetNeighbors() if n.GetIdx() != i]:
            avg_bond_angle += calculate_angle(conf, i, j, neighbor_k)
            angle_count += 1
        
        if angle_count > 0:
            avg_bond_angle /= angle_count


        # B. Dihedral Angle (X-i-j-Y)
        avg_dihedral = 0.0
        dihedral_count = 0
        
        # Find all X-i-j-Y paths where X != j and Y != i
        for neighbor_k in [n.GetIdx() for n in atom_i.GetNeighbors() if n.GetIdx() != j]: # X = neighbor_k
            for neighbor_l in [n.GetIdx() for n in atom_j.GetNeighbors() if n.GetIdx() != i]: # Y = neighbor_l
                avg_dihedral += calculate_dihedral(conf, neighbor_k, i, j, neighbor_l)
                dihedral_count += 1

        if dihedral_count > 0:
            avg_dihedral /= dihedral_count

        
        # ------------------------------------------------
        # 2. Compile Edge Features
        # ------------------------------------------------
        
        bf_base = get_bond_features(bond, bl)
        # Final edge features: [Base_Features (7)] + [Avg_Angle (1)] + [Avg_Dihedral (1)]
        bf = bf_base + [avg_bond_angle, avg_dihedral] 
        
        # Add edges (i->j and j->i)
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    # Convert to tensors
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        # Edge feature size increases from 7 to 9
        edge_attr = torch.zeros((0, 9), dtype=torch.float) 
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        mol_attr=mol_attr
    )


