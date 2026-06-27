"""
Visualize atom importance for a single ligand using Grad-AAM on TriBranchDTI.

Key design decisions for the 2D panel:
  - Ligand is drawn with RDKit 2D coords (canonical, no collisions).
  - Residue bubbles are placed in a RING around the molecule, at the angular
    position derived from the real PCA-projected 3D coords (so the angular
    ordering is physically meaningful), but at a fixed radial distance so
    bubbles never overlap each other or the molecule.
  - Each connector goes to the EXACT ligand atom that has an edge in
    int_data (from interaction_parser), NOT just the spatially nearest atom.
  - When multiple atoms interact with a residue the connector goes to the
    atom with the highest Grad-AAM score.

Usage:  python visualize_one_sample_final.py
"""

import io
import os
import re
import glob

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from Bio.PDB import PDBParser
import cairosvg
from PIL import Image

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog("rdApp.*")

# ── Your project imports ───────────────────────────────────────────────────────
from ProteinLigandInteractionModel_scale_modified_2 import TriBranchDTI
from mol_parser import sdf_to_graph
from protein_parser import protein_to_graph
from interaction_parser_NEW import interaction_to_graph
from grad_aam_tribranch import GradAAM

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH     = "/Users/fatemeh/Downloads/best_model.pth"
BASE_PATH      = "/Users/fatemeh/Desktop"
PDB_CODE       = "5nn8"
OUTPUT_DIR     = "/Users/fatemeh/MGraphDTA-dev/visualization/visualization_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BINDING_CUTOFF = 6.0
TOP_K_RESIDUES = 8

device = torch.device("cpu")

# ══════════════════════════════════════════════════════════════════════════════
#  COLORMAP  (blue -> white -> orange)
# ══════════════════════════════════════════════════════════════════════════════
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "bw_orange",
    [(0.00, "#3a86d4"),
     (0.25, "#7cb8e8"),
     (0.50, "#f0f0f0"),
     (0.75, "#f5a623"),
     (1.00, "#e05c00")],
)

BUBBLE_COLORS = [
    "#c8a4d8", "#a8d8a8", "#90b8e0", "#f5c5a0",
    "#90c8c0", "#f0d080", "#f0a0b8", "#b8d8f0",
]


def score_to_rgb(v: float):
    return CMAP(float(np.clip(v, 0.0, 1.0)))[:3]


# ══════════════════════════════════════════════════════════════════════════════
#  Coordinate helper
# ══════════════════════════════════════════════════════════════════════════════

def load_ligand_3d(mol) -> np.ndarray:
    """Returns (N_atoms, 3) heavy-atom coords from mol conformer."""
    conf = mol.GetConformer()
    return np.array([[*conf.GetAtomPosition(i)] for i in range(mol.GetNumAtoms())])


# ══════════════════════════════════════════════════════════════════════════════
#  Rank residues using EXACT interaction-graph edges from int_data
# ══════════════════════════════════════════════════════════════════════════════

def rank_residues_from_graph(int_data, atom_scores, top_k=8):
    """
    Uses int_data.edge_index to find which ligand atom has an edge to which
    protein residue (ligand node u < L, protein node v >= L).

    The connector for each residue goes to the ligand atom with the highest
    Grad-AAM score among all atoms that interact with it.

    Returns list of dicts sorted by score desc (max top_k):
      { label, res_idx, best_atom, score }
    """
    L          = int(int_data.ligand_num_nodes)
    edge_index = int_data.edge_index.cpu().numpy()   # (2, E)

    # Map protein residue index -> set of interacting ligand atom indices
    res_to_atoms: dict = {}
    for k in range(edge_index.shape[1]):
        u = int(edge_index[0, k])
        v = int(edge_index[1, k])
        if u < L <= v:                              # ligand -> protein
            res_to_atoms.setdefault(v - L, set()).add(u)

    ranked = []
    for res_idx, atoms in res_to_atoms.items():
        scores = [float(atom_scores[a]) for a in atoms if a < len(atom_scores)]
        if not scores:
            continue
        best_atom = list(atoms)[int(np.argmax(scores))]
        res_score = float(np.max(scores))
        chain, resseq, resname = int_data.protein_res_ids[res_idx]
        ranked.append(dict(
            label     = f"{resname}{resseq}",
            res_idx   = res_idx,
            best_atom = best_atom,
            score     = res_score,
        ))

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
#  Compute angle for each residue relative to ligand centroid (via PCA)
# ══════════════════════════════════════════════════════════════════════════════

def compute_residue_angles(ranked_residues, int_data, lig_3d_center):
    """
    Fits a joint PCA on (ligand centroid + residue CA positions) to get a
    shared 2D frame, then computes each residue's angle relative to the
    ligand centroid.  Only the angle is used for placement; radial distance
    is fixed separately.

    Adds key 'angle' (radians) to each dict in-place.
    """
    prot_coords = int_data.protein_coords
    if isinstance(prot_coords, torch.Tensor):
        prot_coords = prot_coords.cpu().numpy()

    res_3d = np.array([prot_coords[r["res_idx"]] for r in ranked_residues])

    # Joint PCA: row 0 = ligand centroid, rows 1.. = residue CAs
    all3d = np.vstack([lig_3d_center.reshape(1, 3), res_3d])
    pca   = PCA(n_components=2)
    all2d = pca.fit_transform(all3d)

    lig2d = all2d[0]
    res2d = all2d[1:]

    for i, r in enumerate(ranked_residues):
        dx = res2d[i, 0] - lig2d[0]
        dy = res2d[i, 1] - lig2d[1]
        r["angle"] = float(np.arctan2(dy, dx))

    return ranked_residues


# ══════════════════════════════════════════════════════════════════════════════
#  Spread bubble angles so circles never overlap
# ══════════════════════════════════════════════════════════════════════════════

def spread_angles(ranked_residues, bubble_radius_canvas=0.100, ring_radius=0.82):
    """
    Sorts residues by angle, then iteratively pushes adjacent bubble centres
    apart until no two circles on the ring overlap.

    Adds key 'bubble_angle' to each dict in-place and returns sorted list.
    """
    n = len(ranked_residues)
    if n == 0:
        return ranked_residues

    # Minimum angular separation so circle edges just touch (+ small margin)
    min_gap = 2.0 * bubble_radius_canvas / ring_radius + 0.04

    ranked_residues = sorted(ranked_residues, key=lambda r: r["angle"])
    angles = np.array([r["angle"] for r in ranked_residues], dtype=float)

    for _ in range(600):
        moved = False
        for i in range(n):
            j = (i + 1) % n
            gap = (angles[j] - angles[i]) % (2 * np.pi)
            if gap < min_gap:
                push = (min_gap - gap) / 2.0
                angles[i] = (angles[i] - push) % (2 * np.pi)
                angles[j] = (angles[j] + push) % (2 * np.pi)
                moved = True
        if not moved:
            break

    for i, r in enumerate(ranked_residues):
        r["bubble_angle"] = float(angles[i])

    return ranked_residues


# ══════════════════════════════════════════════════════════════════════════════
#  Main render function
# ══════════════════════════════════════════════════════════════════════════════

def render_2d_gradaam_panel(mol_noH, scores, ranked_residues,
                             out_png, img_w=700, img_h=560):
    """
    Draw the Grad-AAM 2D panel with residue bubbles in a clean ring layout.

    Parameters
    ----------
    mol_noH          : RDKit Mol (Hs removed)
    scores           : np.ndarray [n_atoms], normalised [0, 1]
    ranked_residues  : list of dicts with keys: label, best_atom, score,
                       angle  (from compute_residue_angles)
    out_png          : output file path
    img_w / img_h    : RDKit SVG canvas size in pixels
    """
    n_atoms = mol_noH.GetNumAtoms()
    scores  = np.array(scores, dtype=float)[:n_atoms]

    # 1. RDKit 2D layout
    rdDepictor.Compute2DCoords(mol_noH)
    conf     = mol_noH.GetConformer()
    xy_rdkit = np.array([[conf.GetAtomPosition(i).x,
                          conf.GetAtomPosition(i).y]
                          for i in range(n_atoms)])

    # 2. Per-atom and per-bond colours
    atom_colors = {i: score_to_rgb(scores[i]) for i in range(n_atoms)}

    bond_colors = {}
    hi_bonds    = []
    for bond in mol_noH.GetBonds():
        bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        mid = tuple((atom_colors[bi][k] + atom_colors[bj][k]) / 2
                    for k in range(3))
        bond_colors[bond.GetIdx()] = mid
        hi_bonds.append(bond.GetIdx())

    hi_radii = {i: 0.45 for i in range(n_atoms)}

    # 3. Render SVG -> PNG
    drawer = rdMolDraw2D.MolDraw2DSVG(img_w, img_h)
    opts   = drawer.drawOptions()
    opts.addAtomIndices      = False
    opts.addStereoAnnotation = False
    opts.bondLineWidth       = 2.5
    opts.atomLabelFontSize   = 0.60
    opts.padding             = 0.12

    rdMolDraw2D.PrepareMolForDrawing(mol_noH)
    drawer.DrawMolecule(
        mol_noH,
        highlightAtoms      = list(range(n_atoms)),
        highlightAtomColors = atom_colors,
        highlightAtomRadii  = hi_radii,
        highlightBonds      = hi_bonds,
        highlightBondColors = bond_colors,
    )
    drawer.FinishDrawing()
    svg_str   = drawer.GetDrawingText()
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode(), scale=2.0)
    mol_img   = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGBA"))

    # 4. Figure  (canvas coordinates: x in [-1, 1], y in [-1, 1])
    fig, ax = plt.subplots(figsize=(12, 9), facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.84, top=0.93, bottom=0.04)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.axis("off")

    # 5. Molecule image centred in the canvas
    MOL_HALF = 0.52          # half-width of the molecule image box
    ax.imshow(mol_img,
              extent=[-MOL_HALF, MOL_HALF, -MOL_HALF, MOL_HALF],
              aspect="auto", zorder=2, interpolation="bilinear")

    # 6. Map RDKit 2D coords -> canvas coords
    pad  = opts.padding
    xmin, xmax = xy_rdkit[:, 0].min(), xy_rdkit[:, 0].max()
    ymin, ymax = xy_rdkit[:, 1].min(), xy_rdkit[:, 1].max()
    xspan = xmax - xmin if xmax > xmin else 1.0
    yspan = ymax - ymin if ymax > ymin else 1.0

    xlo = xmin - xspan * pad;  xhi = xmax + xspan * pad
    ylo = ymin - yspan * pad;  yhi = ymax + yspan * pad

    def rdkit_to_canvas(rx, ry):
        cx = -MOL_HALF + (rx - xlo) / (xhi - xlo) * (2 * MOL_HALF)
        cy = -MOL_HALF + (ry - ylo) / (yhi - ylo) * (2 * MOL_HALF)
        return cx, cy

    atom_canvas = np.array([rdkit_to_canvas(xy_rdkit[i, 0], xy_rdkit[i, 1])
                             for i in range(n_atoms)])

    # 7. Apply angular spreading so no two badges overlap
    # BUBBLE_R matches badge half-width (BW=0.155) for minimum-gap calculation
    BUBBLE_R = 0.155
    RING_R   = 0.82
    ranked_residues = spread_angles(ranked_residues,
                                    bubble_radius_canvas=BUBBLE_R,
                                    ring_radius=RING_R)

    # 8. Draw: solid tapered connector, then badge, then label
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as pe

    for k, r in enumerate(ranked_residues):
        ang   = r["bubble_angle"]
        bx    = RING_R * np.cos(ang)
        by    = RING_R * np.sin(ang)
        color = BUBBLE_COLORS[k % len(BUBBLE_COLORS)]

        # Ligand atom that actually interacts with this residue (from graph)
        atom_idx = min(int(r["best_atom"]), len(atom_canvas) - 1)
        ax_pos, ay_pos = atom_canvas[atom_idx]

        # ── Solid thin connector line (no dashes) ──────────────────
        # Compute point on the bubble edge nearest to the atom, so the
        # line ends cleanly at the bubble boundary, not at its centre.
        dx  = ax_pos - bx
        dy  = ay_pos - by
        d   = np.hypot(dx, dy) + 1e-9
        # Start point: bubble edge
        sx  = bx + dx / d * BUBBLE_R
        sy  = by + dy / d * BUBBLE_R
        # End point: stop a tiny gap before the atom centre
        GAP = 0.018
        ex  = ax_pos - dx / d * GAP
        ey  = ay_pos - dy / d * GAP

        line = plt.Line2D(
            [sx, ex], [sy, ey],
            color="#888888", linewidth=0.9, solid_capstyle="round",
            zorder=3,
        )
        ax.add_line(line)

        # Small filled dot at the atom end to anchor the connector
        ax.plot(ex, ey, "o", color="#888888", markersize=2.2, zorder=4)

        # ── Publication-quality rounded-rectangle badge ─────────────
        # Parse label parts
        m = re.match(r"([A-Z]+)(\d+)", r["label"])
        name_part = m.group(1) if m else r["label"]
        num_part  = m.group(2) if m else ""

        # Badge dimensions
        BW, BH = 0.155, 0.072   # half-width and half-height of badge

        # Outer shadow layer (dark, slightly larger, low alpha)
        shadow = FancyBboxPatch(
            (bx - BW - 0.008, by - BH - 0.008),
            2 * BW + 0.016, 2 * BH + 0.016,
            boxstyle="round,pad=0.012",
            facecolor="#00000022", edgecolor="none",
            zorder=4,
        )
        ax.add_patch(shadow)

        # Badge body with clean border
        badge = FancyBboxPatch(
            (bx - BW, by - BH),
            2 * BW, 2 * BH,
            boxstyle="round,pad=0.012",
            facecolor=color,
            edgecolor="#444444",
            linewidth=0.8,
            zorder=5,
        )
        ax.add_patch(badge)

        # Subtle top-highlight stripe (gives a glossy feel)
        highlight = FancyBboxPatch(
            (bx - BW + 0.008, by + BH * 0.05),
            2 * BW - 0.016, BH * 0.50,
            boxstyle="round,pad=0.006",
            facecolor="#ffffff44", edgecolor="none",
            zorder=6,
        )
        ax.add_patch(highlight)

        # Residue name (bold, larger) and number (lighter, smaller)
        ax.text(
            bx, by + 0.012, name_part,
            ha="center", va="center",
            fontsize=7.0, fontweight="bold",
            color="#1a1a1a",
            zorder=7,
        )
        ax.text(
            bx, by - 0.022, num_part,
            ha="center", va="center",
            fontsize=6.2,
            color="#444444",
            zorder=7,
        )

    # 9. Colorbar
    sm   = cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.02,
                        shrink=0.55, aspect=20)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"])
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label("Grad-AAM score", fontsize=9)

    # 10. Title and save
    ax.set_title(
        f"Grad-AAM  |  {PDB_CODE.upper()}  |  "
        "binding-site residues (interaction-graph edges, non-overlapping ring)",
        fontsize=10, pad=6,
    )
    plt.savefig(out_png, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved -> {out_png}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print(f"Processing complex: {PDB_CODE}")
    print("=" * 70)

    # 1. Load model
    model = TriBranchDTI().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    grad_aam = GradAAM(model, module=model.lig_b3.conv)

    # 2. Locate files
    pdb_dir  = os.path.join(BASE_PATH, PDB_CODE)
    sdf_path = os.path.join(pdb_dir, f"{PDB_CODE}_ligand.sdf")
    pdb_path = os.path.join(pdb_dir, f"{PDB_CODE}_pocket.pdb")

    ply_candidates = sorted(glob.glob(os.path.join(pdb_dir, "*.ply")))
    if not ply_candidates:
        raise FileNotFoundError(f"No .ply file found in {pdb_dir}")
    ply_path = ply_candidates[0]

    # 3. Read molecule
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    if mol is None:
        raise ValueError(f"Failed to read ligand from {sdf_path}")
    mol_noH = Chem.RemoveHs(mol)

    # 4. Build graphs
    lig_data  = sdf_to_graph(mol).to(device)
    prot_data = protein_to_graph(ply_path, pdb_path).to(device)
    int_data  = interaction_to_graph(sdf_path, pdb_path,
                                     cutoff=BINDING_CUTOFF).to(device)

    lig_data.batch  = torch.zeros(lig_data.num_nodes,  dtype=torch.long)
    prot_data.batch = torch.zeros(prot_data.num_nodes, dtype=torch.long)
    int_data.batch  = torch.zeros(int_data.num_nodes,  dtype=torch.long)

    # 5. Grad-AAM
    node_scores, pred_value = grad_aam(prot_data, lig_data, int_data)
    print(f"\nPredicted affinity = {pred_value:.4f}")

    # 6. Heavy-atom scores, normalised
    n_atoms      = mol_noH.GetNumAtoms()
    scores_heavy = np.asarray(node_scores[:n_atoms], dtype=np.float32)

    s_min, s_max = scores_heavy.min(), scores_heavy.max()
    scores_norm  = ((scores_heavy - s_min) / (s_max - s_min)
                    if s_max > s_min else np.full_like(scores_heavy, 0.5))
    print(f"Grad-AAM range: {s_min:.4f} -> {s_max:.4f}")

    # 7. Rank residues via EXACT edges from interaction graph
    ranked = rank_residues_from_graph(int_data, scores_norm,
                                      top_k=TOP_K_RESIDUES)
    print(f"\nTop residues: {[r['label'] for r in ranked]}")

    # 8. Compute angular placement (PCA of 3D CA positions)
    lig_3d        = load_ligand_3d(mol_noH)
    lig_3d_center = lig_3d.mean(axis=0)
    ranked        = compute_residue_angles(ranked, int_data, lig_3d_center)

    # 9. Render 2D panel
    out_png = os.path.join(OUTPUT_DIR, f"{PDB_CODE}_gradaam_2d.png")
    render_2d_gradaam_panel(
        mol_noH         = mol_noH,
        scores          = scores_norm,
        ranked_residues = ranked,
        out_png         = out_png,
    )

    # 10. Bar chart of per-atom importance
    atom_labels     = [f"{mol_noH.GetAtomWithIdx(i).GetSymbol()}{i}"
                       for i in range(n_atoms)]
    atom_colors_bar = [score_to_rgb(scores_norm[i]) for i in range(n_atoms)]

    plt.figure(figsize=(max(8, n_atoms // 2), 4))
    plt.bar(atom_labels, scores_norm, color=atom_colors_bar)
    plt.xlabel("Atom")
    plt.ylabel("Grad-AAM score (normalised)")
    plt.title(f"Ligand Atom Importance - {PDB_CODE.upper()}")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    bar_out = os.path.join(OUTPUT_DIR, f"{PDB_CODE}_gradaam_bar.png")
    plt.savefig(bar_out, dpi=200)
    plt.close()
    print(f"Saved -> {bar_out}")

    # 11. CSV
    csv_out = os.path.join(OUTPUT_DIR, f"{PDB_CODE}_atom_scores.csv")
    with open(csv_out, "w") as f:
        f.write("atom_idx,atom_symbol,score_raw,score_norm\n")
        for i in range(n_atoms):
            sym = mol_noH.GetAtomWithIdx(i).GetSymbol()
            f.write(f"{i},{sym},{scores_heavy[i]:.6f},{scores_norm[i]:.6f}\n")
    print(f"Saved -> {csv_out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
