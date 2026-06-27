# %%
# =========================================================
# COMPLETE PIPELINE
# =========================================================
#
# 1. LOAD ALL PYG FILES FROM:
#       - selected_285
#       - selected_2019
#       - PDBbind_general
#
# 2. EXTRACT EMBEDDINGS USING:
#       from GIGN_My_Mahya import GIGN
#
# 3. PCA
#
# 4. KMEANS CLUSTERING
#
# 5. CLUSTER-BASED SPLIT
#       Biggest clusters   -> TRAIN (~70%)
#       Medium clusters    -> VALID (~15%)
#       Smallest clusters  -> TEST  (~15%)
#
# 6. TRAIN FINAL MODEL USING:
#       from GIGN import GIGN
#
# 7. SAVE:
#       - metrics CSV
#       - split indices
#       - best model
#
# =========================================================

# %%
# =========================================================
# IMPORTS
# =========================================================

import os
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    silhouette_score
)

from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# =========================================================
# MODELS
# =========================================================

# MODEL FOR EMBEDDINGS
from GIGN_My_Mahya import GIGN as EmbeddingGIGN

# FINAL TRAINING MODEL
from GIGN import GIGN

# %%
# =========================================================
# REPRODUCIBILITY
# =========================================================

seed = 42

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)

if torch.cuda.is_available():

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# %%
# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")

# %%
# =========================================================
# DATASET DIRECTORIES
# =========================================================

dataset_dirs = [

    "/Volumes/Ventoy/MolGen_Project copy/data/dataset/selected_285",

    "/Volumes/Ventoy/MolGen_Project copy/data/dataset/selected_2019",

    "/Volumes/Ventoy/MolGen_Project copy/data/dataset/PDBbind_general"
]

# %%
# =========================================================
# LOAD ALL PYG FILES
# =========================================================

graphs = []
graph_ids = []
graph_sources = []

loaded_ids = set()

print("\nLoading all graphs...\n")

for dataset_dir in dataset_dirs:

    dataset_name = os.path.basename(dataset_dir)

    print(f"\nProcessing: {dataset_name}")

    if not os.path.exists(dataset_dir):

        print(f"Directory not found: {dataset_dir}")
        continue

    # -----------------------------------------------------
    # EACH SUBFOLDER = ONE COMPLEX
    # -----------------------------------------------------

    for cid in os.listdir(dataset_dir):

        complex_dir = os.path.join(
            dataset_dir,
            cid
        )

        if not os.path.isdir(complex_dir):
            continue

        # -------------------------------------------------
        # FIND ALL .PYG FILES
        # -------------------------------------------------

        pyg_files = [

            f for f in os.listdir(complex_dir)
            if f.endswith(".pyg")
        ]

        if len(pyg_files) == 0:
            continue

        # -------------------------------------------------
        # LOAD EACH GRAPH
        # -------------------------------------------------

        for pyg_file in pyg_files:

            graph_path = os.path.join(
                complex_dir,
                pyg_file
            )

            unique_id = f"{dataset_name}_{cid}"

            if unique_id in loaded_ids:
                continue

            try:

                data = torch.load(graph_path)

                # -----------------------------------------
                # ENSURE FLOAT TARGET
                # -----------------------------------------

                if hasattr(data, "y"):

                    data.y = data.y.float()

                graphs.append(data)

                graph_ids.append(cid)

                graph_sources.append(dataset_name)

                loaded_ids.add(unique_id)

            except Exception as e:

                print(f"\nError loading:")
                print(graph_path)
                print(e)

print("\n===================================")
print(f"Total graphs loaded: {len(graphs)}")
print("===================================\n")

# %%
# =========================================================
# SAVE METADATA
# =========================================================

metadata_df = pd.DataFrame({

    "pdbid": graph_ids,
    "source": graph_sources
})

metadata_df.to_csv(
    "all_loaded_graphs.csv",
    index=False
)

print("Saved: all_loaded_graphs.csv")

# %%
# =========================================================
# DATALOADER FOR EMBEDDING EXTRACTION
# =========================================================

embedding_loader = DataLoader(
    graphs,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# %%
# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = EmbeddingGIGN(
    node_dim=42,
    hidden_dim=256
).to(device)

# %%
# =========================================================
# EXTRACT EMBEDDINGS
# =========================================================

print("\nExtracting embeddings...\n")

embedding_model.eval()

all_embeddings = []

with torch.no_grad():

    for batch in embedding_loader:

        batch = batch.to(device)

        emb = embedding_model(batch)

        emb = emb.detach().cpu().numpy()

        all_embeddings.append(emb)

all_embeddings = np.concatenate(
    all_embeddings,
    axis=0
)

print(f"Embedding shape: {all_embeddings.shape}")

# %%
# =========================================================
# PCA
# =========================================================

print("\nRunning PCA...\n")

pca = PCA(
    n_components=0.98,
    random_state=seed
)

reduced_emb = pca.fit_transform(
    all_embeddings
)

print(f"PCA reduced shape: {reduced_emb.shape}")

# %%
# %%
# =========================================================
# ELBOW METHOD FOR BEST K
# =========================================================

import matplotlib.pyplot as plt

print("\nRunning Elbow Method...\n")

K_range = range(10, 201, 10)

inertias = []

for k in K_range:

    print(f"Processing K={k}")

    kmeans = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init="auto"
    )

    kmeans.fit(reduced_emb)

    inertias.append(
        kmeans.inertia_
    )

# =========================================================
# PLOT ELBOW CURVE
# =========================================================

plt.figure(figsize=(8, 6))

plt.plot(
    list(K_range),
    inertias,
    marker='o'
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.grid(True)

plt.savefig(
    "elbow_plot.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

print("\nSaved: elbow_plot.png")

# =========================================================
# AUTOMATIC ELBOW DETECTION
# =========================================================

# normalize
x = np.array(list(K_range))
y = np.array(inertias)

x_norm = (x - x.min()) / (x.max() - x.min())
y_norm = (y - y.min()) / (y.max() - y.min())

# line between first and last point
line_vec = np.array([
    x_norm[-1] - x_norm[0],
    y_norm[-1] - y_norm[0]
])

line_vec = line_vec / np.linalg.norm(line_vec)

# distances to line
distances = []

for i in range(len(x_norm)):

    point = np.array([
        x_norm[i] - x_norm[0],
        y_norm[i] - y_norm[0]
    ])

    proj = np.dot(point, line_vec) * line_vec

    orth = point - proj

    distances.append(
        np.linalg.norm(orth)
    )

best_k = x[np.argmax(distances)]

# =========================================================
# SAFETY CONSTRAINT
# =========================================================

# avoid too-small K for dataset splitting

if best_k < 30:

    best_k = 30

print(f"\nSelected K by Elbow Method: {best_k}")

# %%
# =========================================================
# FINAL KMEANS
# =========================================================

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=seed,
    n_init="auto"
)

labels = final_kmeans.fit_predict(
    reduced_emb
)

# %%
# =========================================================
# FINAL KMEANS
# =========================================================

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=seed,
    n_init="auto"
)

labels = final_kmeans.fit_predict(
    reduced_emb
)

# %%
# =========================================================
# CLUSTER ANALYSIS
# =========================================================

cluster_df = pd.DataFrame({

    "idx": np.arange(len(graphs)),
    "cluster": labels
})

cluster_sizes = (

    cluster_df["cluster"]
    .value_counts()
    .sort_values(ascending=False)
)

print("\nCluster sizes:\n")
print(cluster_sizes)

# %%
# =========================================================
# SORT CLUSTERS
# =========================================================

sorted_clusters = cluster_sizes.index.tolist()

total_samples = len(graphs)

target_train = int(0.70 * total_samples)
target_valid = int(0.15 * total_samples)

train_idx = []
valid_idx = []
test_idx = []

train_count = 0
valid_count = 0

# %%
# =========================================================
# SPLIT BASED ON CLUSTER SIZE
# =========================================================

for cluster_id in sorted_clusters:

    cluster_indices = cluster_df[
        cluster_df["cluster"] == cluster_id
    ]["idx"].tolist()

    cluster_size = len(cluster_indices)

    # TRAIN
    if train_count < target_train:

        train_idx.extend(cluster_indices)

        train_count += cluster_size

    # VALID
    elif valid_count < target_valid:

        valid_idx.extend(cluster_indices)

        valid_count += cluster_size

    # TEST
    else:

        test_idx.extend(cluster_indices)

# %%
# =========================================================
# CREATE DATASETS
# =========================================================

train_dataset = [graphs[i] for i in train_idx]
valid_dataset = [graphs[i] for i in valid_idx]
test_dataset  = [graphs[i] for i in test_idx]

print("\n===================================")
print(f"Train size : {len(train_dataset)}")
print(f"Valid size : {len(valid_dataset)}")
print(f"Test size  : {len(test_dataset)}")
print("===================================\n")

# =========================================================
# VISUALIZATION: PCA / t-SNE / UMAP (Publication Quality)
# =========================================================

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# UMAP (make sure: pip install umap-learn)
import umap.umap_ as umap

print("\nGenerating PCA / t-SNE / UMAP plots...\n")

# ---------------------------------------------------------
# CREATE SPLIT MASKS
# ---------------------------------------------------------
n_samples = len(graphs)

split_labels = np.array(["test"] * n_samples)

split_labels[train_idx] = "train"
split_labels[valid_idx] = "valid"

# ---------------------------------------------------------
# COLOR MAP (50 clusters)
# ---------------------------------------------------------
n_clusters_plot = 50
cmap = plt.cm.get_cmap("tab20", n_clusters_plot)

cluster_colors = [cmap(i % n_clusters_plot) for i in labels]

# ---------------------------------------------------------
# MARKERS FOR SPLITS
# ---------------------------------------------------------
marker_map = {
    "train": "o",
    "valid": "s",
    "test": "^"
}

# ---------------------------------------------------------
# REDUCE FOR T-SNE / UMAP INPUT
# (use PCA output already computed: reduced_emb)
# ---------------------------------------------------------

# -------------------------
# t-SNE
# -------------------------
tsne = TSNE(
    n_components=2,
    perplexity=35,
    learning_rate="auto",
    init="pca",
    random_state=42
)

X_tsne = tsne.fit_transform(reduced_emb)

# -------------------------
# UMAP
# -------------------------
umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.1,
    random_state=42
)

X_umap = umap_model.fit_transform(reduced_emb)

# -------------------------
# PCA 2D (for visualization)
# -------------------------
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(all_embeddings)

# ---------------------------------------------------------
# PLOTTING FUNCTION
# ---------------------------------------------------------
def plot_embedding(ax, X, title):
    for split in ["train", "valid", "test"]:
        idx_split = np.where(split_labels == split)[0]

        ax.scatter(
            X[idx_split, 0],
            X[idx_split, 1],
            c=[cluster_colors[i] for i in idx_split],
            marker=marker_map[split],
            s=18,
            alpha=0.75,
            linewidths=0.2,
            edgecolors="black",
            label=split
        )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

# ---------------------------------------------------------
# FIGURE
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(24, 7), dpi=300)

plot_embedding(axes[0], X_pca_2d, "PCA Projection")
plot_embedding(axes[1], X_tsne, "t-SNE Projection")
plot_embedding(axes[2], X_umap, "UMAP Projection")

handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="upper center", ncol=3)

plt.tight_layout()
plt.savefig("embedding_visualization.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved: embedding_visualization.png")

# %%
# =========================================================
# SAVE SPLITS
# =========================================================

np.save(
    "train_indices.npy",
    np.array(train_idx)
)

np.save(
    "valid_indices.npy",
    np.array(valid_idx)
)

np.save(
    "test_indices.npy",
    np.array(test_idx)
)

print("Saved split indices.")

# %%
# =========================================================
# DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# %%
# =========================================================
# FINAL MODEL
# =========================================================

model = GIGN(
    node_dim=42,
    hidden_dim=256
).to(device)

# %%
# =========================================================
# OPTIMIZER & LOSS
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

criterion = nn.MSELoss()

# %%
# =========================================================
# METRICS FUNCTION
# =========================================================
# =========================================================
# CI
# =========================================================

# =========================================================
# CI (CONCORDANCE INDEX)
# =========================================================

def concordance_index(y_true, y_pred):

    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)

    n = 0
    h_sum = 0.0

    for i in range(len(y_true)):

        for j in range(i + 1, len(y_true)):

            # skip equal targets
            if y_true[i] == y_true[j]:
                continue

            n += 1

            # concordant pair
            if (
                (y_pred[i] > y_pred[j] and y_true[i] > y_true[j])
                or
                (y_pred[i] < y_pred[j] and y_true[i] < y_true[j])
            ):

                h_sum += 1

            # tied prediction
            elif y_pred[i] == y_pred[j]:

                h_sum += 0.5

    # avoid division by zero
    if n == 0:
        return 0.0

    return h_sum / n

def calculate_metrics(targets, preds):

    targets = np.array(targets).reshape(-1)
    preds   = np.array(preds).reshape(-1)

    # RMSE
    rmse = np.sqrt(
        mean_squared_error(
            targets,
            preds
        )
    )

    # MAE
    mae = mean_absolute_error(
        targets,
        preds
    )

    # PEARSON
    rp, _ = pearsonr(
        targets,
        preds
    )

    # CI
    ci = concordance_index(
        targets,
        preds
    )

    # SD
    sd = np.std(
        targets - preds
    )

    # R predicted
    ss_res = np.sum(
        (targets - preds) ** 2
    )

    ss_tot = np.sum(
        (targets - np.mean(targets)) ** 2
    )

    r_pred = 1 - (ss_res / ss_tot)

    return {

        "RMSE": rmse,
        "Rp": rp,
        "MAE": mae,
        "CI": ci,
        "SD": sd,
        "R_predicted": r_pred
    }

# %%
# =========================================================
# TRAIN FUNCTION
# =========================================================

def train_epoch():

    model.train()

    total_loss = 0

    preds = []
    targets = []

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        output = model(batch)

        output = output.view(-1)

        y = batch.y.view(-1)

        loss = criterion(
            output,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds.extend(
            output.detach().cpu().numpy()
        )

        targets.extend(
            y.detach().cpu().numpy()
        )

    metrics = calculate_metrics(
        targets,
        preds
    )

    metrics["Loss"] = (
        total_loss / len(train_loader)
    )

    return metrics

# %%
# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate(loader):

    model.eval()

    preds = []
    targets = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            output = model(batch)

            output = output.view(-1)

            y = batch.y.view(-1)

            preds.extend(
                output.detach().cpu().numpy()
            )

            targets.extend(
                y.detach().cpu().numpy()
            )

    metrics = calculate_metrics(
        targets,
        preds
    )

    return metrics

# %%
# =========================================================
# TRAINING LOOP
# =========================================================

epochs = 100

best_val_rmse = 999

history = []

print("\nStarting training...\n")

for epoch in range(1, epochs + 1):

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    train_metrics = train_epoch()

    # -----------------------------------------------------
    # VALID
    # -----------------------------------------------------

    val_metrics = evaluate(
        valid_loader
    )

    # -----------------------------------------------------
    # TEST
    # -----------------------------------------------------

    test_metrics = evaluate(
        test_loader
    )

    # -----------------------------------------------------
    # PRINT
    # -----------------------------------------------------

    print(

        f"Epoch {epoch:03d} | "

        f"Train RMSE: {train_metrics['RMSE']:.4f} | "

        f"Val RMSE: {val_metrics['RMSE']:.4f} | "

        f"Test RMSE: {test_metrics['RMSE']:.4f}"
    )

    # -----------------------------------------------------
    # SAVE HISTORY
    # -----------------------------------------------------

    row = {

        "Epoch": epoch,

        # TRAIN
        "Train_Loss": train_metrics["Loss"],
        "Train_RMSE": train_metrics["RMSE"],
        "Train_Rp": train_metrics["Rp"],
        "Train_MAE": train_metrics["MAE"],
        "Train_CI": train_metrics["CI"],
        "Train_SD": train_metrics["SD"],
        "Train_R_predicted": train_metrics["R_predicted"],

        # VALID
        "Val_RMSE": val_metrics["RMSE"],
        "Val_Rp": val_metrics["Rp"],
        "Val_MAE": val_metrics["MAE"],
        "Val_CI": val_metrics["CI"],
        "Val_SD": val_metrics["SD"],
        "Val_R_predicted": val_metrics["R_predicted"],

        # TEST
        "Test_RMSE": test_metrics["RMSE"],
        "Test_Rp": test_metrics["Rp"],
        "Test_MAE": test_metrics["MAE"],
        "Test_CI": test_metrics["CI"],
        "Test_SD": test_metrics["SD"],
        "Test_R_predicted": test_metrics["R_predicted"]
    }

    history.append(row)

    # -----------------------------------------------------
    # SAVE CSV EACH EPOCH
    # -----------------------------------------------------

    metrics_df = pd.DataFrame(
        history
    )

    metrics_df.to_csv(
        "training_metrics.csv",
        index=False
    )

    # -----------------------------------------------------
    # SAVE BEST MODEL
    # -----------------------------------------------------

    if val_metrics["RMSE"] < best_val_rmse:

        best_val_rmse = val_metrics["RMSE"]

        torch.save(
            model.state_dict(),
            "best_GIGN_surface.pt"
        )

        print("Best model saved.")

# %%
# =========================================================
# SAVE FINAL RESULTS
# =========================================================

final_results = pd.DataFrame({

    "Dataset": [
        "Train",
        "Validation",
        "Test"
    ],

    "RMSE": [
        train_metrics["RMSE"],
        val_metrics["RMSE"],
        test_metrics["RMSE"]
    ],

    "Rp": [
        train_metrics["Rp"],
        val_metrics["Rp"],
        test_metrics["Rp"]
    ],

    "MAE": [
        train_metrics["MAE"],
        val_metrics["MAE"],
        test_metrics["MAE"]
    ],

    "CI": [
        train_metrics["CI"],
        val_metrics["CI"],
        test_metrics["CI"]
    ],

    "SD": [
        train_metrics["SD"],
        val_metrics["SD"],
        test_metrics["SD"]
    ],

    "R_predicted": [
        train_metrics["R_predicted"],
        val_metrics["R_predicted"],
        test_metrics["R_predicted"]
    ]
})

final_results.to_csv(
    "final_results.csv",
    index=False
)

print("\n===================================")
print("Training completed successfully.")
print("===================================")

print("\nSaved files:")
print("1. training_metrics.csv")
print("2. final_results.csv")
print("3. best_GIGN_surface.pt")
print("4. train_indices.npy")
print("5. valid_indices.npy")
print("6. test_indices.npy")
print("7. all_loaded_graphs.csv")
