# %%
import os
import pickle
import numpy as np

from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import pymol

from rdkit import RDLogger
from sklearn.neighbors import NearestNeighbors

RDLogger.DisableLog('rdApp.*')

# %%


# =========================================================
# LOAD PLY SURFACE
# =========================================================
def load_ply_surface(ply_path):

    with open(ply_path, "r") as f:
        lines = f.readlines()

    # -------------------------
    # FIND HEADER END
    # -------------------------
    start_idx = 0

    for i, line in enumerate(lines):

        if line.strip() == "end_header":

            start_idx = i + 1
            break

    surface = []

    # -------------------------
    # READ VERTICES ONLY
    # vertex format:
    # x y z charge hbond hphob iface nx ny nz
    # -------------------------
    for line in lines[start_idx:]:

        vals = line.strip().split()

        # faces start with:
        # 3 94 1417 1090
        if len(vals) != 10:
            continue

        try:

            vals = list(map(float, vals))

            surface.append(vals)

        except:
            continue

    surface = np.array(surface)

    return surface


# =========================================================
# GET ATOM COORDS
# =========================================================
def get_atom_coords(mol):

    conf = mol.GetConformer()

    coords = []

    for i in range(mol.GetNumAtoms()):

        pos = conf.GetAtomPosition(i)

        coords.append([
            pos.x,
            pos.y,
            pos.z
        ])

    return np.array(coords)


# =========================================================
# SURFACE -> ATOM FEATURES
# =========================================================
def surface_to_atom_features(
    atom_coords,
    surface,
    k=20,
    sigma=2.0
):

    # xyz
    surface_xyz = surface[:, :3]

    # 7 surface features:
    # charge hbond hphob iface nx ny nz
    surface_feat = surface[:, 3:]

    # -------------------------
    # KNN
    # -------------------------
    nn = NearestNeighbors(
        n_neighbors=k,
        algorithm='kd_tree'
    )

    nn.fit(surface_xyz)

    distances, indices = nn.kneighbors(atom_coords)

    pooled_features = []

    # -------------------------
    # GAUSSIAN WEIGHTED POOLING
    # -------------------------
    for i in range(len(atom_coords)):

        neigh_feat = surface_feat[indices[i]]

        neigh_dist = distances[i]

        weights = np.exp(
            -(neigh_dist ** 2) / (2 * sigma ** 2)
        )

        weights = weights / (
            weights.sum() + 1e-8
        )

        pooled = np.sum(
            neigh_feat * weights[:, None],
            axis=0
        )

        pooled_features.append(pooled)

    pooled_features = np.array(
        pooled_features,
        dtype=np.float32
    )

    return pooled_features


# %%


def generate_pocket(data_dir, distance=5):

    complex_ids = os.listdir(data_dir)

    ignored_pockets = []

    print("\n==============================")
    print("GENERATING POCKETS")
    print("==============================")

    for cid in complex_ids:

        print(f"\nProcessing Pocket: {cid}")

        complex_dir = os.path.join(data_dir, cid)

        # -------------------------
        # CHECK DIRECTORY
        # -------------------------
        if not os.path.isdir(complex_dir):

            print(f"[IGNORED] Not a directory: {cid}")
            ignored_pockets.append(cid)
            continue

        lig_native_path = os.path.join(
            complex_dir,
            f"{cid}_ligand.mol2"
        )

        protein_path = os.path.join(
            complex_dir,
            f"{cid}_protein.pdb"
        )

        pocket_save_path = os.path.join(
            complex_dir,
            f'Pocket_{distance}A.pdb'
        )

        # -------------------------
        # CHECK INPUT FILES
        # -------------------------
        if not os.path.exists(lig_native_path):

            print(f"[IGNORED] Missing ligand file: {cid}")
            ignored_pockets.append(cid)
            continue

        if not os.path.exists(protein_path):

            print(f"[IGNORED] Missing protein file: {cid}")
            ignored_pockets.append(cid)
            continue

        # -------------------------
        # SKIP IF ALREADY EXISTS
        # -------------------------
        if os.path.exists(pocket_save_path):

            print(f"[SKIPPED] Pocket already exists: {cid}")
            continue

        # -------------------------
        # GENERATE POCKET
        # -------------------------
        try:

            pymol.cmd.load(protein_path)

            pymol.cmd.remove('resn HOH')

            pymol.cmd.load(lig_native_path)

            pymol.cmd.remove('hydrogens')

            pymol.cmd.select(
                'Pocket',
                f'byres {cid}_ligand around {distance}'
            )

            pymol.cmd.save(
                pocket_save_path,
                'Pocket'
            )

            pymol.cmd.delete('all')

        except Exception as e:

            print(f"[ERROR] Pocket generation failed for {cid}")
            print(e)

            ignored_pockets.append(cid)

            pymol.cmd.delete('all')

            continue

    print("\n==============================")
    print("POCKET GENERATION SUMMARY")
    print("==============================")

    print(f"Ignored Pocket Cases: {len(ignored_pockets)}")

    if len(ignored_pockets) > 0:

        print("\nIgnored Pocket IDs:")

        for x in ignored_pockets:
            print(x)

        with open("ignored_pocket_cases.txt", "w") as f:
            for x in ignored_pockets:
                f.write(x + "\n")


# %%


def generate_complex(
    data_dir,
    data_df,
    distance=5,
    input_ligand_format='mol2',
    k=20,
    sigma=2.0
):

    pbar = tqdm(total=len(data_df))

    ignored_complexes = []
    processed_complexes = []

    for i, row in data_df.iterrows():

        cid = row['pdbid']

        try:
            pKa = float(row['-logKd/Ki'])
        except:
            pKa = None

        complex_dir = os.path.join(data_dir, cid)

        # -------------------------
        # CHECK DIRECTORY EXISTS
        # -------------------------
        if not os.path.exists(complex_dir):

            print(f"\n[IGNORED] Missing folder: {cid}")

            ignored_complexes.append(cid)

            pbar.update(1)

            continue

        pocket_path = os.path.join(
            complex_dir,
            f'Pocket_{distance}A.pdb'
        )

        # -------------------------
        # CHECK POCKET EXISTS
        # -------------------------
        if not os.path.exists(pocket_path):

            print(f"\n[IGNORED] Missing pocket file: {cid}")

            ignored_complexes.append(cid)

            pbar.update(1)

            continue

        # -------------------------
        # FIND PLY FILE
        # -------------------------
        ply_path = None

        for file_name in os.listdir(complex_dir):

            if file_name.endswith(".ply"):

                ply_path = os.path.join(
                    complex_dir,
                    file_name
                )

                break

        if ply_path is None:

            print(f"\n[IGNORED] Missing ply file: {cid}")

            ignored_complexes.append(cid)

            pbar.update(1)

            continue

        # -------------------------
        # PREPARE LIGAND
        # -------------------------
        try:

            if input_ligand_format != 'pdb':

                ligand_input_path = os.path.join(
                    complex_dir,
                    f'{cid}_ligand.{input_ligand_format}'
                )

                if not os.path.exists(ligand_input_path):

                    print(f"\n[IGNORED] Missing ligand file: {cid}")

                    ignored_complexes.append(cid)

                    pbar.update(1)

                    continue

                ligand_path = ligand_input_path.replace(
                    f".{input_ligand_format}",
                    ".pdb"
                )

                # only convert if pdb does not exist
                if not os.path.exists(ligand_path):

                    convert_cmd = (
                        f'obabel "{ligand_input_path}" '
                        f'-O "{ligand_path}" -d'
                    )

                    os.system(convert_cmd)

            else:

                ligand_path = os.path.join(
                    complex_dir,
                    f'{cid}_ligand.pdb'
                )

            # -------------------------
            # CHECK CONVERTED FILE
            # -------------------------
            if not os.path.exists(ligand_path):

                print(f"\n[IGNORED] Ligand conversion failed: {cid}")

                ignored_complexes.append(cid)

                pbar.update(1)

                continue

            # -------------------------
            # LOAD LIGAND
            # -------------------------
            ligand = Chem.MolFromPDBFile(
                ligand_path,
                removeHs=True
            )

            if ligand is None:

                print(f"\n[IGNORED] Unable to process ligand: {cid}")

                ignored_complexes.append(cid)

                pbar.update(1)

                continue

            # -------------------------
            # LOAD POCKET
            # -------------------------
            pocket = Chem.MolFromPDBFile(
                pocket_path,
                removeHs=True
            )

            if pocket is None:

                print(f"\n[IGNORED] Unable to process protein: {cid}")

                ignored_complexes.append(cid)

                pbar.update(1)

                continue

            # =====================================================
            # NEW PART:
            # SURFACE FEATURE EXTRACTION USING KNN
            # =====================================================

            # load surface
            surface = load_ply_surface(ply_path)

            # get pocket atom coords
            atom_coords = get_atom_coords(pocket)

            # aggregate surface -> atom features
            surface_features = surface_to_atom_features(
                atom_coords=atom_coords,
                surface=surface,
                k=k,
                sigma=sigma
            )

            # =====================================================
            # SAVE COMPLEX
            # ONLY CHANGE:
            # add surface_features as third item
            # =====================================================

            save_path = os.path.join(
                complex_dir,
                f"{cid}_{distance}A.rdkit"
            )

            # ORIGINAL:
            # complex_data = (ligand, pocket)

            # NEW:
            complex_data = (
                ligand,
                pocket,
                surface_features
            )

            with open(save_path, 'wb') as f:
                pickle.dump(complex_data, f)

            processed_complexes.append(cid)

        except Exception as e:

            print(f"\n[ERROR] Failed processing {cid}")
            print(e)

            ignored_complexes.append(cid)

        pbar.update(1)

    # =============================
    # FINAL SUMMARY
    # =============================
    print("\n==============================")
    print("COMPLEX GENERATION SUMMARY")
    print("==============================")

    print(f"Processed complexes : {len(processed_complexes)}")
    print(f"Ignored complexes   : {len(ignored_complexes)}")

    # -------------------------
    # PRINT IGNORED IDS
    # -------------------------
    if len(ignored_complexes) > 0:

        print("\nIgnored PDB IDs:")

        for x in ignored_complexes:
            print(x)

        with open("ignored_complexes.txt", "w") as f:
            for x in ignored_complexes:
                f.write(x + "\n")

    # -------------------------
    # SAVE SUCCESSFUL IDS
    # -------------------------
    with open("processed_complexes.txt", "w") as f:
        for x in processed_complexes:
            f.write(x + "\n")


# %%


# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':

    distance = 5

    input_ligand_format = 'mol2'

    data_root = '/Volumes/Ventoy/MolGen_Project copy/data/dataset'

    data_dir = os.path.join(
        data_root,
        'PDBbind_general'
    )

    # =====================================================
    # LOAD CSV
    # IMPORTANT:
    # must contain:
    # pdbid
    # -logKd/Ki
    # =====================================================
    data_df = pd.read_csv(
        "/Volumes/Ventoy/MolGen_Project copy/data/dataset/PDBbind_general/index_general2020.csv"
    )

    print("\n==============================")
    print("CSV SUMMARY")
    print("==============================")

    print(data_df.head())

    print("\nTotal rows:", len(data_df))

    # =====================================================
    # FILTER ONLY VALID DIRECTORIES
    # avoids:
    # index/
    # readme/
    # ._ files
    # csv files
    # =====================================================
    valid_dirs = []

    for x in os.listdir(data_dir):

        full_path = os.path.join(data_dir, x)

        if (
            os.path.isdir(full_path)
            and not x.startswith(".")
            and len(x) == 4
        ):
            valid_dirs.append(x)

    valid_dirs = set(valid_dirs)

    data_df = data_df[
        data_df['pdbid'].isin(valid_dirs)
    ]

    print("\nValid complexes:", len(data_df))

    # =====================================================
    # GENERATE COMPLEXES
    # (Pocket already exists)
    # =====================================================
    generate_complex(
        data_dir=data_dir,
        data_df=data_df,
        distance=distance,
        input_ligand_format=input_ligand_format,
        k=20,
        sigma=2.0
    )

# %%
# %%