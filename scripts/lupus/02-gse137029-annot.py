import os

import celltypist
import scanpy as sc

# ------------------- Configurable Parameters -------------------
DATA_PATH = "/Users/aumchampaneri/complement-receptor-blockade/data/lupus-gse137029/gse137029_processed.h5ad"
OUTDIR_DATA = "/Users/aumchampaneri/complement-receptor-blockade/data/lupus-gse137029/"
OUTDIR_PLOTS = (
    "/Users/aumchampaneri/complement-receptor-blockade/plots/lupus-gse137029/"
)
OUTDIR_OUTPUTS = (
    "/Users/aumchampaneri/complement-receptor-blockade/outputs/lupus-gse137029/"
)
os.makedirs(OUTDIR_DATA, exist_ok=True)
os.makedirs(OUTDIR_PLOTS, exist_ok=True)
os.makedirs(OUTDIR_OUTPUTS, exist_ok=True)

ANNOTATED_PATH = os.path.join(OUTDIR_DATA, "GSE137029_merged_qc_celltypist.h5ad")

# ------------------- 1. Load Processed AnnData -------------------
print(f"Loading processed AnnData from {DATA_PATH}")
adata = sc.read_h5ad(DATA_PATH)

# ------------------- 2. Extract gene symbols from var_names -------------------
print("Extracting gene symbols from var_names...")


def extract_symbol(name):
    parts = name.split("\t")
    if len(parts) >= 2:
        return parts[1]
    return name


adata.var_names = [extract_symbol(g) for g in adata.var_names]
print("Extracted gene symbols. Example gene names:", adata.var_names[:10])

# ------------------- 3. Run CellTypist Automated Annotation -------------------
from celltypist.models import Model

model_dir = os.path.expanduser("~/.celltypist/data/models")
model_path = os.path.join(model_dir, "Immune_All_Low.pkl")

if os.path.exists(model_path):
    print("CellTypist model found locally. Loading...")
    model = Model.load(model_path)
else:
    print("CellTypist model not found locally. Downloading...")
    model = celltypist.models.download_models("Immune_All_Low.pkl")

print("Running CellTypist annotation...")
predictions = celltypist.annotate(adata, model=model, majority_voting=True)

# Assign only the main cell type label (majority voting or first column)
if "majority_voting" in predictions.predicted_labels.columns:
    adata.obs["celltypist_label"] = predictions.predicted_labels["majority_voting"]
else:
    adata.obs["celltypist_label"] = predictions.predicted_labels.iloc[:, 0]

# Save cell type labels to CSV
adata.obs["celltypist_label"].to_csv(
    os.path.join(OUTDIR_OUTPUTS, "celltypist_labels.csv")
)

print("CellTypist annotation complete. Labels saved to outputs.")

# ------------------- 4. UMAP Plotting -------------------
print("Plotting UMAP colored by CellTypist label...")
sc.pl.umap(
    adata,
    color="celltypist_label",
    save="_celltypist_umap.png",
    show=False,
)
umap_fig_path = "figures/umap_celltypist_umap.png"
if os.path.exists(umap_fig_path):
    os.rename(umap_fig_path, os.path.join(OUTDIR_PLOTS, "umap_celltypist_umap.png"))
else:
    print(f"WARNING: CellTypist UMAP plot not found at {umap_fig_path}")

# ------------------- 5. Save Annotated AnnData -------------------
adata.write(ANNOTATED_PATH)
print(f"Saved annotated AnnData to {ANNOTATED_PATH}")

print("Automated annotation pipeline complete. Outputs written to:")
print(f"  Data:    {OUTDIR_DATA}")
print(f"  Plots:   {OUTDIR_PLOTS}")
print(f"  Outputs: {OUTDIR_OUTPUTS}")
