# ============================================================
# FINAL COMPLETE INFERENCE SCRIPT
# Single Protein + Multiple MOL2 Ligands
# WITH AUTOMATIC POCKET PLY EXTRACTION
# ============================================================

import os
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from rdkit.Chem import rdmolfiles
from rdkit import Chem

from scipy.spatial import cKDTree

# ============================================================
# IMPORT MODEL
# ============================================================

from ProteinLigandInteractionModel_scale_modified import (
    TriBranchDTI,
)

# ============================================================
# IMPORT YOUR PARSERS
# ============================================================

from mol_parser import sdf_to_graph
from protein_parser import protein_to_graph
from interaction_parser_NEW import interaction_to_graph

# ============================================================
# PATHS
# ============================================================

DATA_FOLDER = "/home/ubuntu/masif-neosurf/Input"

PROTEIN_PDB = os.path.join(DATA_FOLDER, "5nn8.pdb")

FULL_SURFACE_PLY = os.path.join(DATA_FOLDER, "5nn8.ply")

CSV_FILE = os.path.join(DATA_FOLDER, "structural_study.csv")

MODEL_WEIGHTS = "/home/ubuntu/mol_gen_project/Feb28/best_model.pth"

SCALER_JSON = os.path.join(DATA_FOLDER, "target_scaler_params.json")

OUTPUT_CSV = os.path.join(DATA_FOLDER, "final_predictions_2.csv")

# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8

INTERACTION_CUTOFF = 6.0

POCKET_SURFACE_CUTOFF = 8.0

# ============================================================
# LOAD TARGET SCALER
# ============================================================

with open(SCALER_JSON, "r") as f:
    scaler_info = json.load(f)

TARGET_MEAN = scaler_info["mean"]
TARGET_SCALE = scaler_info["scale"]

print("===================================================")
print("Loaded target scaler")
print(f"Mean  : {TARGET_MEAN}")
print(f"Scale : {TARGET_SCALE}")
print("===================================================\n")

# ================================================================
# Load trained model
# ================================================================
def load_model(model_path="/home/ubuntu/mol_gen_project/Feb28/best_model.pth", device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # IMPORTANT:
    # Must EXACTLY match training architecture
    # ------------------------------------------------------------
    global BASE_OUT
    global BASE_HEADS
    global USE_INT_3RD_LAYER

    BASE_OUT = 64
    BASE_HEADS = 2
    USE_INT_3RD_LAYER = True

    # ------------------------------------------------------------
    # Create model with SAME dimensions as training
    # ------------------------------------------------------------
    model = TriBranchDTI(
        prot_node_dim=26,
        prot_edge_dim=9,
        lig_node_dim=26,
        lig_edge_dim=9,
        int_node_dim=26,
        int_edge_dim=9,
        lig_attr_dim=6,
        dropout=0.2
    ).to(device)

    # ------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------
    checkpoint = torch.load(model_path, map_location=device)

    # Handle checkpoints saved in different formats
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # ------------------------------------------------------------
    # Load weights
    # ------------------------------------------------------------
    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    print("\n===================================================")
    print("MODEL LOADING")
    print("===================================================")

    if len(missing) > 0:
        print("\nMissing keys:")
        for k in missing:
            print(k)

    if len(unexpected) > 0:
        print("\nUnexpected keys:")
        for k in unexpected:
            print(k)

    print("\n✅ Model loaded successfully")
    print("===================================================\n")

    model.eval()

    return model

# ============================================================
# READ PLY
# ============================================================

def read_ply_vertices(ply_path):

    with open(ply_path, "r") as f:
        lines = f.readlines()

    vertex_count = None
    header_end = None

    for i, line in enumerate(lines):

        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])

        if line.strip() == "end_header":
            header_end = i + 1
            break

    if vertex_count is None:
        raise ValueError("PLY file missing vertex definition")

    vertices = []

    for i in range(vertex_count):

        parts = lines[header_end + i].strip().split()

        row = list(map(float, parts))

        vertices.append(row)

    vertices = np.array(vertices)

    return vertices, lines[:header_end]

# ============================================================
# WRITE POCKET PLY
# ============================================================

def write_pocket_ply(
    output_ply,
    header_lines,
    selected_vertices
):

    header_new = []

    for line in header_lines:

        if line.startswith("element vertex"):

            header_new.append(
                f"element vertex {len(selected_vertices)}\n"
            )

        else:
            header_new.append(line)

    with open(output_ply, "w") as f:

        for line in header_new:
            f.write(line)

        for row in selected_vertices:

            row_str = " ".join(map(str, row))

            f.write(row_str + "\n")

# ============================================================
# EXTRACT LIGAND-SPECIFIC POCKET SURFACE
# ============================================================

def extract_pocket_surface_ply(
    full_ply_path,
    ligand_mol2_path,
    output_pocket_ply,
    cutoff=8.0
):

    # ========================================================
    # READ FULL PLY
    # ========================================================

    vertices, header = read_ply_vertices(
        full_ply_path
    )

    vertex_xyz = vertices[:, :3]

    # ========================================================
    # READ LIGAND
    # ========================================================

    mol = rdmolfiles.MolFromMol2File(
        ligand_mol2_path,
        removeHs=False
    )

    if mol is None:

        raise ValueError(
            f"Cannot read ligand:\n{ligand_mol2_path}"
        )

    conf = mol.GetConformer()

    ligand_coords = []

    for i in range(mol.GetNumAtoms()):

        pos = conf.GetAtomPosition(i)

        ligand_coords.append([
            pos.x,
            pos.y,
            pos.z
        ])

    ligand_coords = np.array(ligand_coords)

    # ========================================================
    # KD TREE SEARCH
    # ========================================================

    tree = cKDTree(vertex_xyz)

    selected_indices = set()

    for coord in ligand_coords:

        ids = tree.query_ball_point(
            coord,
            r=cutoff
        )

        selected_indices.update(ids)

    selected_indices = sorted(list(selected_indices))

    if len(selected_indices) == 0:

        raise ValueError(
            f"No pocket vertices found for:\n{ligand_mol2_path}"
        )

    pocket_vertices = vertices[selected_indices]

    # ========================================================
    # SAVE POCKET PLY
    # ========================================================

    write_pocket_ply(
        output_pocket_ply,
        header,
        pocket_vertices
    )

    return output_pocket_ply

# ============================================================
# DATASET
# ============================================================

class SingleProteinLigandDataset(Dataset):

    def __init__(
        self,
        csv_file,
        protein_pdb,
        full_surface_ply,
        interaction_cutoff=6.0
    ):

        self.df = pd.read_csv(csv_file)

        self.protein_pdb = protein_pdb

        self.full_surface_ply = full_surface_ply

        self.cutoff = interaction_cutoff

        # ====================================================
        # CHECK REQUIRED COLUMNS
        # ====================================================

        required_cols = [
            "Name",
            "REAL VALUE"
        ]

        for col in required_cols:

            if col not in self.df.columns:

                raise ValueError(
                    f"Missing required column: {col}"
                )

        print("===================================================")
        print(f"Loaded {len(self.df)} ligands")
        print("===================================================\n")

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        ligand_name = str(row["Name"])

        real_value = float(row["REAL VALUE"])

        # =================================================
        # FILE PATHS
        # =================================================

        sdf_path = os.path.join(
            DATA_FOLDER,
            f"{ligand_name}.sdf"
        )

        mol2_path = os.path.join(
            DATA_FOLDER,
            f"{ligand_name}.mol2"
        )

        if not os.path.exists(sdf_path):

            print(f"Missing SDF file: {sdf_path}")

            return None

        if not os.path.exists(mol2_path):

            print(f"Missing MOL2 file: {mol2_path}")

            return None

        try:

            print(f"Processing: {ligand_name}")

            # =================================================
            # READ SDF (used for ligand graph + interaction graph)
            # =================================================

            supplier = Chem.SDMolSupplier(
                sdf_path,
                removeHs=False
            )

            if len(supplier) == 0 or supplier[0] is None:

                print(f"RDKit failed on: {sdf_path}")

                return None

            mol = supplier[0]

            # =================================================
            # CREATE LIGAND GRAPH
            # =================================================

            lig_graph = sdf_to_graph(mol)

            if not hasattr(lig_graph, "mol_attr"):

                lig_graph.mol_attr = torch.zeros(
                    (1, 6),
                    dtype=torch.float32
                )

            # =================================================
            # EXTRACT POCKET SURFACE
            # (still uses MOL2 coordinates)
            # =================================================

            pocket_ply = os.path.join(
                DATA_FOLDER,
                f"{ligand_name}_pocket_surface.ply"
            )

            extract_pocket_surface_ply(
                full_ply_path=self.full_surface_ply,
                ligand_mol2_path=mol2_path,
                output_pocket_ply=pocket_ply,
                cutoff=POCKET_SURFACE_CUTOFF
            )

            # =================================================
            # CREATE PROTEIN GRAPH
            # =================================================

            prot_graph = protein_to_graph(
                pocket_ply,
                self.protein_pdb
            )

            # =================================================
            # CREATE INTERACTION GRAPH
            # IMPORTANT: interaction_parser_NEW expects SDF
            # =================================================

            int_graph = interaction_to_graph(
                sdf_path,
                self.protein_pdb,
                cutoff=self.cutoff
            )
            return (
                prot_graph,
                lig_graph,
                int_graph,
                torch.tensor([real_value], dtype=torch.float32),
                ligand_name
            )

        except Exception as e:

            print("\n========================================")
            print(f"Error processing ligand: {ligand_name}")
            print(str(e))
            print("========================================\n")

            return None

# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

    prots, ligs, inters, ys, names = zip(*batch)

    return (
        Batch.from_data_list(prots),
        Batch.from_data_list(ligs),
        Batch.from_data_list(inters),
        torch.stack(ys),
        names
    )

# ============================================================
# INVERSE TARGET SCALING
# ============================================================

def inverse_transform(y_scaled):

    return (
        y_scaled * TARGET_SCALE
    ) + TARGET_MEAN

# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # DATASET
    # ========================================================

    dataset = SingleProteinLigandDataset(
        csv_file=CSV_FILE,
        protein_pdb=PROTEIN_PDB,
        full_surface_ply=FULL_SURFACE_PLY,
        interaction_cutoff=INTERACTION_CUTOFF
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # PREDICTION
    # ========================================================

    results = []

    print("===================================================")
    print("Starting predictions...")
    print("===================================================\n")

    with torch.no_grad():
        m_int = 3.411
        for batch in loader:

            if batch is None:
                continue

            p, l, inter, y_true, names = batch

            p = p.to(DEVICE)

            l = l.to(DEVICE)

            inter = inter.to(DEVICE)

            pred_scaled = model(
                p,
                l,
                inter
            )

            pred_scaled = (
                pred_scaled
                .cpu()
                .numpy()
                .ravel()
            )

            pred_real = inverse_transform(
                pred_scaled
            )  - m_int

            true_real = (
                y_true
                .numpy()
                .ravel()
            )

            for n, t, pval in zip(
                names,
                true_real,
                pred_real
            ):

                results.append({
                    "Name": n,
                    "REAL VALUE": float(t),
                    "PREDICTED VALUE": float(pval),
                    "ABS ERROR": abs(
                        float(t) - float(pval)
                    )
                })

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("===================================================")
    print("Prediction finished")
    print(f"Saved results to:\n{OUTPUT_CSV}")
    print("===================================================\n")

    # ========================================================
    # FINAL METRICS
    # ========================================================
    if len(results_df) > 0:

        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            r2_score
        )

        y_true = results_df[
            "REAL VALUE"
        ].values

        y_pred = results_df[
            "PREDICTED VALUE"
        ].values

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        r2 = r2_score(
            y_true,
            y_pred
        )

        print("=============== FINAL METRICS =================")

        print(f"RMSE : {rmse:.4f}")

        print(f"MAE  : {mae:.4f}")

        print(f"R2   : {r2:.4f}")

        print("================================================")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
