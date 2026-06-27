# protein_graph_builder_clean.py
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
import torch
from chemistry import polarHydrogens, radii

# ================================================================
#     Amino Acid Chemical Motif Classes  (Deep Learning Motifs)
# ================================================================

RESIDUE_LIST = [
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
]

AA_MOTIF = {
    "ACCEPTOR": ["ASP","GLU","HIS"],
    "DONOR": ["ARG","LYS","HIS","TRP"],
    "BOTH": ["SER","THR","TYR","ASN","GLN"],
    "ALIPHATIC": ["ALA","ILE","LEU","VAL","MET","CYS"],
    "AROMATIC": ["PHE","TYR","TRP"],
    "NONE": ["GLY","PRO"]
}


def motif_vector(resname):
    """Return a 6-vector describing the chemical motif class."""
    r = resname.upper()
    return np.array([
        int(r in AA_MOTIF["ACCEPTOR"]),
        int(r in AA_MOTIF["DONOR"]),
        int(r in AA_MOTIF["BOTH"]),
        int(r in AA_MOTIF["ALIPHATIC"]),
        int(r in AA_MOTIF["AROMATIC"]),
        int(r in AA_MOTIF["NONE"])
    ], dtype=float)


def residue_one_hot(resname):
    """20-dim one-hot AA."""
    one_hot = np.zeros(len(RESIDUE_LIST))
    if resname in RESIDUE_LIST:
        one_hot[RESIDUE_LIST.index(resname)] = 1.0
    return one_hot

# ================================================================
#     Standard Amino Acid Reference Data (Library)
# ================================================================

# 1. Molecular Weights (g/mol) - Standard residues
AA_WEIGHTS = {
    "ALA": 89.1, "ARG": 174.2, "ASN": 132.1, "ASP": 133.1, "CYS": 121.2,
    "GLN": 146.2, "GLU": 147.1, "GLY": 75.1, "HIS": 155.2, "ILE": 131.2,
    "LEU": 131.2, "LYS": 146.2, "MET": 149.2, "PHE": 165.2, "PRO": 115.1,
    "SER": 105.1, "THR": 119.1, "TRP": 204.2, "TYR": 181.2, "VAL": 117.1
}

# 2. Standard Volumes (Angstrom^3) - commonly cited Van der Waals volumes
#    (Zamyatnin, A.A., Prog. Biophys. Mol. Biol. 24: 107-123, 1972)
AA_VOLUME = {
    "ALA": 88.6, "ARG": 173.4, "ASN": 114.1, "ASP": 111.1, "CYS": 108.5,
    "GLN": 143.8, "GLU": 138.4, "GLY": 60.1, "HIS": 153.2, "ILE": 166.7,
    "LEU": 166.7, "LYS": 168.6, "MET": 162.9, "PHE": 189.9, "PRO": 112.7,
    "SER": 89.0, "THR": 116.1, "TRP": 227.8, "TYR": 193.6, "VAL": 140.0
}

# 3. Density (g/cm^3 approx or relative density)
#    Often calculated as Weight / Volume. We will compute this on the fly or define it here.
#    Here, I'll calculate it on the fly in the function to save space.

# 4. Standard Atom Counts (C, N, O, S, H) for the neutral state
AA_ATOM_COUNTS = {
    #       C  N  O  S   H
    "ALA": [3, 1, 1, 0,  5],
    "ARG": [6, 4, 1, 0, 12],
    "ASN": [4, 2, 2, 0,  6],
    "ASP": [4, 1, 3, 0,  5],
    "CYS": [3, 1, 1, 1,  5],
    "GLN": [5, 2, 2, 0,  8],
    "GLU": [5, 1, 3, 0,  7],
    "GLY": [2, 1, 1, 0,  3],
    "HIS": [6, 3, 1, 0,  7],
    "ILE": [6, 1, 1, 0, 11],
    "LEU": [6, 1, 1, 0, 11],
    "LYS": [6, 2, 1, 0, 12],
    "MET": [5, 1, 1, 1,  9],
    "PHE": [9, 1, 1, 0,  9],
    "PRO": [5, 1, 1, 0,  7],
    "SER": [3, 1, 2, 0,  5],
    "THR": [4, 1, 2, 0,  7],
    "TRP": [11, 2, 1, 0, 10],
    "TYR": [9, 1, 2, 0,  9],
    "VAL": [5, 1, 1, 0,  9]
}

# 5. Topological Polar Surface Area (TPSA) - approx from PubChem (Angstrom^2)
AA_TPSA = {
    "ALA": 63.3, "ARG": 126.0, "ASN": 106.0, "ASP": 100.0, "CYS": 63.3,
    "GLN": 106.0, "GLU": 100.0, "GLY": 63.3, "HIS": 88.0, "ILE": 63.3,
    "LEU": 63.3, "LYS": 89.0, "MET": 63.3, "PHE": 63.3, "PRO": 49.3,
    "SER": 83.6, "THR": 83.6, "TRP": 79.1, "TYR": 83.6, "VAL": 63.3
}

# 6. Hydrogen Bond Counts (Donor, Acceptor) - Reference values (Free AA)
#    Format: [Donor_Count, Acceptor_Count]
AA_HBOND_COUNTS = {
    "ALA": [2, 3], "ARG": [4, 4], "ASN": [3, 4], "ASP": [3, 5], "CYS": [2, 3],
    "GLN": [3, 4], "GLU": [3, 5], "GLY": [2, 3], "HIS": [3, 3], "ILE": [2, 3],
    "LEU": [2, 3], "LYS": [3, 3], "MET": [2, 3], "PHE": [2, 3], "PRO": [1, 2],
    "SER": [3, 4], "THR": [3, 4], "TRP": [3, 3], "TYR": [3, 4], "VAL": [2, 3]
}

# 7. Theoretical Max SASA (Solvent Accessible Surface Area) - (Angstrom^2)
#    (Based on tripeptide Gly-X-Gly extended conformation values - Tien et al., 2013)
AA_MAX_SASA = {
    "ALA": 121.0, "ARG": 265.0, "ASN": 187.0, "ASP": 187.0, "CYS": 148.0,
    "GLN": 214.0, "GLU": 214.0, "GLY": 97.0, "HIS": 216.0, "ILE": 195.0,
    "LEU": 191.0, "LYS": 230.0, "MET": 203.0, "PHE": 228.0, "PRO": 154.0,
    "SER": 143.0, "THR": 163.0, "TRP": 264.0, "TYR": 255.0, "VAL": 165.0
}


# -----------------------------
# کلاس ProteinGraphBuilder
# -----------------------------
class ProteinGraphBuilder:
    def __init__(self, ply_file, pdb_file, cutoff=6.0):
        self.ply_file = ply_file
        self.pdb_file = pdb_file
        self.cutoff = cutoff

        self.vertices = None
        self.structure = None
        self.residue_to_vertices = defaultdict(list)
        self.residue_features = {}
        self.residue_coords = {}
        self.residue_normals = {}
        self.edges = None
        self.edge_features = None

    # -------------------------------------------------------------
    # 1) خواندن فایل PLY
    # -------------------------------------------------------------
    def read_ply(self):
        with open(self.ply_file, 'r') as f:
            lines = f.readlines()

        vertex_count = None
        for line in lines:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line.strip() == "end_header":
                header_end = lines.index(line) + 1
                break

        assert vertex_count is not None, "PLY file missing 'element vertex' definition."

        data = []
        for i in range(vertex_count):
            parts = lines[header_end + i].strip().split()
            x, y, z = map(float, parts[:3])
            features = list(map(float, parts[3:]))
            data.append([x, y, z] + features)

        self.vertices = np.array(data)

    # -------------------------------------------------------------
    # 2) خواندن فایل PDB
    # -------------------------------------------------------------
    def read_pdb(self):
        parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure("protein", self.pdb_file)

    # -------------------------------------------------------------
    # 3) نگاشت خودکار vertex → residue
    # -------------------------------------------------------------
    def auto_map_vertices_to_residues(self):
        atom_coords = []
        atom_res_ids = []

        for model in self.structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != " ":  # HETATM ignored
                        continue
                    res_id = (chain.id, residue.id[1])
                    for atom in residue.get_atoms():
                        atom_coords.append(atom.coord)
                        atom_res_ids.append(res_id)

        atom_coords = np.array(atom_coords)
        tree = cKDTree(atom_coords)

        for i, v in enumerate(self.vertices[:, :3]):
            dist, idx = tree.query(v)
            res_id = atom_res_ids[idx]
            self.residue_to_vertices[res_id].append(i)

    # -------------------------------------------------------------
    # 4) ویژگی‌های اتمی residue
    # -------------------------------------------------------------
    def compute_atomic_features(self, residue):
        resname = residue.get_resname()
        
        # --- Existing Library Fetches ---
        counts = AA_ATOM_COUNTS.get(resname, [0, 0, 0, 0, 0])
        weight = AA_WEIGHTS.get(resname, 0.0)
        volume = AA_VOLUME.get(resname, 0.0)
        density = weight / volume if volume > 0 else 0.0
        is_aromatic = 1.0 if resname in ["HIS","TYR","TRP","PHE"] else 0.0
        
        # --- NEW Library Fetches ---
        # 1. TPSA
        tpsa = AA_TPSA.get(resname, 0.0)
        
        # 2. H-Bond Donors & Acceptors
        hb_counts = AA_HBOND_COUNTS.get(resname, [0, 0])
        hb_donor = hb_counts[0]
        hb_acceptor = hb_counts[1]
        
        # 3. Reference SASA
        ref_sasa = AA_MAX_SASA.get(resname, 0.0)

        # --- Combine All Features ---
        # Order: 
        # [C, N, O, S, H, Mass, Vol, Dens, Arom, TPSA, H_Donor, H_Accept, Ref_SASA]
        atomic_feat = [
            counts[0], counts[1], counts[2], counts[3], counts[4],
            weight, volume, density, is_aromatic,
            tpsa, hb_donor, hb_acceptor, ref_sasa
        ]
        
        return np.array(atomic_feat, dtype=float)

    # -------------------------------------------------------------
    # 5) ساخت ویژگی نودها
    # -------------------------------------------------------------
    def build_node_features(self):
        for res_id, v_indices in self.residue_to_vertices.items():
            verts = self.vertices[v_indices]
            surf_feat = verts[:, 3:].mean(axis=0)
            coord = verts[:, :3].mean(axis=0)
            normal = verts[:, -3:].mean(axis=0)

            chain, resseq = res_id
            residue = self.structure[0][chain][resseq]
            atomic_feat = self.compute_atomic_features(residue)
            
            # --- تغییر اینجا: اضافه کردن motif_vector ---
            motif_feat = motif_vector(residue.get_resname())
            # -----------------------------------------------

            # الحاق ویژگی‌های سطحی، اتمی، موتیف، و مختصات (coord)
            self.residue_features[res_id] = np.concatenate([surf_feat, atomic_feat, motif_feat, coord])
            self.residue_coords[res_id] = coord
            self.residue_normals[res_id] = normal

    # -------------------------------------------------------------
    # 6) محاسبه نوع تعامل بین دو residue
    # -------------------------------------------------------------
    def compute_interaction_type(self, resA, resB):
        chargeA = 1 if resA.get_resname() in ['ARG','LYS'] else (-1 if resA.get_resname() in ['ASP','GLU'] else 0)
        chargeB = 1 if resB.get_resname() in ['ARG','LYS'] else (-1 if resB.get_resname() in ['ASP','GLU'] else 0)

        if chargeA*chargeB < 0:
            return [0,0,1,0]  # electrostatic

        hydrophobic_res = ['ALA','VAL','LEU','ILE','MET','PHE','TRP','PRO']
        if resA.get_resname() in hydrophobic_res and resB.get_resname() in hydrophobic_res:
            return [0,0,0,1]  # hydrophobic

        donor_acceptor_atoms = ['N','O']
        atomsA = [a.element for a in resA.get_atoms()]
        atomsB = [a.element for a in resB.get_atoms()]
        if any(a in donor_acceptor_atoms for a in atomsA) and any(a in donor_acceptor_atoms for a in atomsB):
            return [1,0,0,0]  # hbond

        return [0,1,0,0]  # vdw

    # -------------------------------------------------------------
    # 7) ساخت یال‌ها و ویژگی‌هایشان
    # -------------------------------------------------------------
    def build_edges(self):
        edges = []
        edge_features = []
        residue_ids = list(self.residue_coords.keys())

        for i, resA_id in enumerate(residue_ids):
            coordA = self.residue_coords[resA_id]
            normalA = self.residue_normals[resA_id]
            chainA, resseqA = resA_id
            resA = self.structure[0][chainA][resseqA]

            for j, resB_id in enumerate(residue_ids):
                if i >= j:
                    continue
                coordB = self.residue_coords[resB_id]
                normalB = self.residue_normals[resB_id]
                chainB, resseqB = resB_id
                resB = self.structure[0][chainB][resseqB]

                dist = np.linalg.norm(coordA - coordB)
                if dist <= self.cutoff:
                    edges.append([i, j])
                    dir_vec = coordB - coordA
                    dir_vec /= np.linalg.norm(dir_vec)
                    normal_sim = np.dot(normalA, normalB)
                    interaction = self.compute_interaction_type(resA, resB)
                    edge_features.append([dist, *interaction, *dir_vec, normal_sim])

        self.edges = np.array(edges)
        self.edge_features = np.array(edge_features)

    # -------------------------------------------------------------
    # 8) گرفتن گراف نودها و یال‌ها
    # -------------------------------------------------------------
    def get_graph(self):
        node_feats = np.array(list(self.residue_features.values()))
        return node_feats, self.edges, self.edge_features

# -----------------------------
# ساخت DataFrame
# -----------------------------
def build_residue_dataframe(builder):
    node_ids = list(builder.residue_features.keys())
    node_feats = np.array(list(builder.residue_features.values()))

    # 1. Surface Features (7)
    surf_cols = ['iface','charge','hbond','hphob','nx','ny','nz']
    
    # 2. Atomic Features (13) - UPDATED with 4 new features
    # Old: count_C...H, mass, volume, density, is_aromatic
    # New: ... + tpsa, hb_donor, hb_acceptor, ref_sasa
    atomic_cols = [
        'count_C','count_N','count_O','count_S','count_H',
        'molecular_weight', 'volume', 'density', 'is_aromatic',
        'TPSA', 'hbond_donor_count', 'hbond_acceptor_count', 'ref_SASA'
    ]
    
    # 3. Motif Features (6)
    motif_cols = [
        'motif_ACCEPTOR', 'motif_DONOR', 'motif_BOTH',
        'motif_ALIPHATIC', 'motif_AROMATIC', 'motif_NONE'
    ]
    
    # 4. Coordinates (3)
    coord_cols = ['coord_x', 'coord_y', 'coord_z']

    # Total Features: 7 + 13 + 6 + 3 = 29
    feature_cols_actual = surf_cols + atomic_cols + motif_cols + coord_cols
    NUM_FEATS = 29  # Updated count

    # Prepare data
    data = []
    for i, (chain, resseq) in enumerate(node_ids):
        residue = builder.structure[0][chain][resseq]
        res_name = residue.get_resname()
        
        feats = node_feats[i]
        
        # Pad with zeros if needed
        feats = list(feats) + [0]*(NUM_FEATS - len(feats))
        
        # Add metadata
        row = feats + [chain, resseq, res_name]
        data.append(row)

    all_cols = feature_cols_actual + ['chain', 'resseq', 'res_name']

    df_nodes = pd.DataFrame(data, columns=all_cols)
    return df_nodes

    # -------------------------------------------------------------------

    chain_resseq_cols = ['chain','resseq']
    all_cols = feature_cols_actual + chain_resseq_cols

    data = []
    for i, (chain, resseq) in enumerate(node_ids):
        feats = node_feats[i]
        # اگر طول feats کمتر از NUM_FEATS بود با صفر پرش کن
        feats = list(feats) + [0]*(NUM_FEATS - len(feats))
        row = feats + [chain, resseq]
        data.append(row)

    df_nodes = pd.DataFrame(data, columns=all_cols)
    return df_nodes

def build_edge_dataframe(builder):
    node_ids = list(builder.residue_features.keys())
    edges = builder.edges
    edge_feats = builder.edge_features
    edge_cols = ['distance','hbond','vdw','electrostatic','hydrophobic','dx','dy','dz','normal_sim']
    data = []
    for k, (i, j) in enumerate(edges):
        row = list(edge_feats[k]) + [f"{node_ids[i][0]}{node_ids[i][1]}", f"{node_ids[j][0]}{node_ids[j][1]}"]
        data.append(row)
    df_edges = pd.DataFrame(data, columns=edge_cols + ['residue_A','residue_B'])
    return df_edges

# -----------------------------
# def protein to graph
# -----------------------------
def protein_to_graph(ply_file, pdb_file, cutoff=6.0):
    builder = ProteinGraphBuilder(ply_file, pdb_file, cutoff=cutoff)
    builder.read_ply()
    builder.read_pdb()
    builder.auto_map_vertices_to_residues()
    
    # These methods modify builder.residue_features, builder.edges, etc. in-place
    builder.build_node_features()
    builder.build_edges()
    
    # Extract features from the builder object
    # residue_features.values() contains [surf + atomic + motif + coord]
    all_node_feats = np.array(list(builder.residue_features.values()))
    
    # If you want to exclude coordinates (the last 3 elements) from the feature tensor x:
    node_feats = all_node_feats[:, :-3] 
    
    edge_index = builder.edges
    edge_attr = builder.edge_features
    
    # Create the torch_geometric Data object
    prot_graph = Data(
        x=torch.tensor(node_feats, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float)
    )
    
    return prot_graph

# -----------------------------
# پلات سه‌بعدی
# -----------------------------
def plot_3d_graph(builder):
    node_ids = list(builder.residue_features.keys())
    # node_feats شامل ویژگی‌های سطح، اتمی، موتیف و مختصات است
    node_feats = np.array(list(builder.residue_features.values()))
    
    # مختصات در 3 ستون آخر هستند
    coords = node_feats[:, -3:]
    edges = builder.edges

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], c='skyblue', s=80)

    # لیبل نودها
    for i, res_id in enumerate(node_ids):
        ax.text(coords[i,0], coords[i,1], coords[i,2], f"{res_id[1]}", color='red')

    # رسم خطوط یال‌ها
    for i, j in edges:
        x = [coords[i,0], coords[j,0]]
        y = [coords[i,1], coords[j,1]]
        z = [coords[i,2], coords[j,2]]
        ax.plot(x, y, z, c='gray', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Residue Nodes with Edges')
    plt.show()

# -----------------------------
# === مثال استفاده ===
# -----------------------------
# -----------------------------
# === مثال استفاده ===
# -----------------------------
if __name__ == "__main__":
    ply_file = "/Volumes/ADATA SC740/Refined-prep/refined-set/3ucj/output/all_feat_3l/pred_surfaces/3UCJ_A.ply"
    pdb_file = "/Volumes/ADATA SC740/Refined-prep/refined-set/3ucj/3ucj_pocket.pdb"

    # 1. Create builder and process data for CSV exports
    builder = ProteinGraphBuilder(ply_file, pdb_file)
    builder.read_ply()
    builder.read_pdb()
    builder.auto_map_vertices_to_residues()
    builder.build_node_features()
    builder.build_edges()

    # 2. Save DataFrames as before
    df_nodes = build_residue_dataframe(builder)
    df_nodes.to_csv("/Users/fatemeh/Desktop/MolGen_Project copy/data/output/protein_nodes_3ucj.csv", index=False)
    print("saved: /Users/fatemeh/Desktop/MolGen_Project copy/data/output/protein_nodes_3ucj.csv")
    
    df_edges = build_edge_dataframe(builder)
    df_edges.to_csv("/Users/fatemeh/Desktop/MolGen_Project copy/data/output/protein_edges_3ucj.csv", index=False)
    print("saved: /Users/fatemeh/Desktop/MolGen_Project copy/data/output/protein_edges_3ucj.csv")

    # 3. Use the edited protein_to_graph function to get the PyTorch Data object
    prt_grph = protein_to_graph(ply_file, pdb_file)
    
    # 4. Print the PyTorch Geometric attributes
    print("\n=== PyTorch Geometric Graph Object ===")
    print("prot graph:", prt_grph)
    print("node features (x) shape:", prt_grph.x.shape)
    print("node features (x):", prt_grph.x)
    print('edge index shape:', prt_grph.edge_index.shape)
    print('edge index:', prt_grph.edge_index)
    print('edge attr shape:', prt_grph.edge_attr.shape)
    print('edge attr:', prt_grph.edge_attr)

    # 5. Optional: Plot the graph
    plot_3d_graph(builder)