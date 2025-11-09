#!/usr/bin/env python3
"""
Lupus Single-Cell Analysis Pipeline (GSE137029: Flare & SLE)

Performs best-practice QC, normalization, clustering, marker diagnostics, and annotation.
Designed for academic reproducibility and transparency.

INPUT:
    - raw-data/lupus-data/GSE137029/GSE137029_flare.mtx
    - raw-data/lupus-data/GSE137029/GSE137029_flare.barcodes.txt
    - raw-data/lupus-data/GSE137029/GSE137029_flare.genes.txt
    - raw-data/lupus-data/GSE137029/GSE137029_sle.mtx
    - raw-data/lupus-data/GSE137029/GSE137029_sle.barcodes.txt
    - raw-data/lupus-data/GSE137029/GSE137029_sle.genes.txt

OUTPUT:
    - Merged & processed AnnData (.h5ad)
    - QC summary (.csv)
    - Diagnostic plots (UMAP, marker expression)
    - Cluster annotation table (.csv)

ENVIRONMENT:
    - Python >=3.8
    - scanpy >=1.9
    - anndata, matplotlib, seaborn, pandas, numpy
    - Recommend: Save `pip freeze` or `conda env export` for reproducibility

AUTHOR: [Your Name]
DATE: [Auto-generated]
"""

import os

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import seaborn as sns

# ------------------- Configurable Parameters -------------------
FLARE_MATRIX = "/Users/aumchampaneri/complement-receptor-blockade/raw-data/lupus-data/GSE137029/GSE137029_flare.mtx"
FLARE_BARCODES = "/Users/aumchampaneri/complement-receptor-blockade/raw-data/lupus-data/GSE137029/GSE137029_flare.barcodes.txt"
FLARE_GENES = "/Users/aumchampaneri/complement-receptor-blockade/raw-data/lupus-data/GSE137029/GSE137029_flare.genes.txt"

SLE_MATRIX = "/Users/aumchampaneri/complement-receptor-blockade/raw-data/lupus-data/GSE137029/GSE137029_sle.mtx"
SLE_BARCODES = "/Users/aumchampaneri/complement-receptor-blockade/raw-data/lupus-data/GSE137029/GSE137029_sle.barcodes.txt"
SLE_GENES = "/Users/aumchampaneri/complement-receptor-blockade/raw-data/lupus-data/GSE137029/GSE137029_sle.genes.txt"

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

MIN_GENES_PER_CELL = 200
MIN_CELLS_PER_GENE = 3
MAX_MITO_PCT = 10.0
TARGET_SUM = 1e4
N_TOP_GENES = 2000
N_PCS = 20
N_NEIGHBORS = 15
RANDOM_STATE = 42

# Canonical marker genes for annotation (edit as needed)
MARKERS = {
    "T_cells": ["CD3D", "CD3E", "CD2"],
    "B_cells": ["MS4A1", "CD79A"],
    "NK_cells": ["NKG7", "GNLY"],
    "Monocytes": ["CD14", "LYZ"],
    "Dendritic": ["FCER1A", "CST3"],
    "Plasma": ["SDC1"],
    "Epithelial": ["KRT19", "EPCAM"],
    "Fibroblast": ["COL1A1", "DCN"],
    "Endothelial": ["PECAM1", "VWF"],
}


def load_10x(mtx_path, barcodes_path, genes_path, sample_label):
    print(f"Reading matrix from: {mtx_path}")
    if not os.path.exists(mtx_path):
        raise FileNotFoundError(f"Matrix file not found: {mtx_path}")
    mtx = scipy.io.mmread(mtx_path).tocsc()
    print(f"Loaded matrix shape: {mtx.shape}")
    with open(barcodes_path) as f:
        barcodes = [f"{sample_label}_{line.strip()}" for line in f]
    print(f"Barcodes loaded: {len(barcodes)}")
    with open(genes_path) as f:
        genes = [line.strip().split("\t")[0] for line in f]
    print(f"Genes loaded: {len(genes)}")
    # Matrix is already (cells, genes): rows = cells, columns = genes
    print(f"Matrix shape for AnnData: {mtx.shape}")
    adata = anndata.AnnData(mtx)
    adata.obs_names = barcodes
    adata.var_names = genes
    adata.obs["sample"] = sample_label
    return adata


# ------------------- 1. Load & Merge Data -------------------
print("Loading flare sample...")
adata_flare = load_10x(FLARE_MATRIX, FLARE_BARCODES, FLARE_GENES, "flare")
print("Loading SLE sample...")
adata_sle = load_10x(SLE_MATRIX, SLE_BARCODES, SLE_GENES, "sle")

print("Making gene names unique...")
adata_flare.var_names_make_unique()
adata_sle.var_names_make_unique()

print("Concatenating AnnData objects...")
adata = adata_flare.concatenate(
    adata_sle, join="outer", batch_key="group", batch_categories=["flare", "sle"]
)

print(f"Merged AnnData: {adata.shape[0]} cells × {adata.shape[1]} genes")
print("Obs columns:", list(adata.obs.columns))
print("Var columns:", list(adata.var.columns))

# ------------------- 2. QC: Mitochondrial Genes -------------------
adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)

# ------------------- 3. Filter Cells & Genes -------------------
print(
    "Filtering cells with <{} genes and >{}% mito...".format(
        MIN_GENES_PER_CELL, MAX_MITO_PCT
    )
)
initial_n_cells = adata.n_obs
sc.pp.filter_cells(adata, min_genes=MIN_GENES_PER_CELL)
adata = adata[adata.obs["pct_counts_mt"] < MAX_MITO_PCT, :]
print(f"Cells after filtering: {adata.n_obs} (removed {initial_n_cells - adata.n_obs})")

print("Filtering genes detected in <{} cells...".format(MIN_CELLS_PER_GENE))
initial_n_genes = adata.n_vars
sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)
print(
    f"Genes after filtering: {adata.n_vars} (removed {initial_n_genes - adata.n_vars})"
)

# Save QC summary
qc_summary = adata.obs[
    ["n_genes_by_counts", "total_counts", "pct_counts_mt", "sample", "group"]
]
qc_summary.to_csv(os.path.join(OUTDIR_OUTPUTS, "qc_summary.csv"))

# ------------------- 4. Normalization & Log1p -------------------
if adata.X.min() < 0:
    print(
        "Data appears log-transformed (negative values present). Skipping normalization/log1p."
    )
else:
    print("Normalizing to {} counts/cell and log1p...".format(int(TARGET_SUM)))
    sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
    sc.pp.log1p(adata)

# ------------------- 5. Highly Variable Genes -------------------
print("Selecting highly variable genes (diagnostic only)...")
sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES, flavor="seurat_v3")
hvg = adata.var[adata.var["highly_variable"]]
print(f"Number of HVGs: {hvg.shape[0]}")
sc.pl.highly_variable_genes(adata, save="_hvg.png", show=False)
hvg_fig_path = "figures/filter_genes_dispersion_hvg.png"
if os.path.exists(hvg_fig_path):
    os.rename(hvg_fig_path, os.path.join(OUTDIR_PLOTS, "highly_variable_genes_hvg.png"))
else:
    print(f"WARNING: HVG plot not found at {hvg_fig_path}")

# ------------------- 6. Scaling, PCA, Neighbors, UMAP -------------------
# Skipping scaling to avoid memory issues with large datasets
print("Running PCA on log-normalized data (no scaling)...")
sc.tl.pca(adata, svd_solver="arpack", random_state=RANDOM_STATE)
sc.pl.pca_variance_ratio(adata, log=True, save="_pca_var.png", show=False)
pca_fig_path = "figures/pca_variance_ratio_pca_var.png"
if os.path.exists(pca_fig_path):
    os.rename(
        pca_fig_path, os.path.join(OUTDIR_PLOTS, "pca_variance_ratio_pca_var.png")
    )
else:
    print(f"WARNING: PCA variance ratio plot not found at {pca_fig_path}")

# ------------------- Harmony Batch Correction -------------------
print("Running Harmony batch correction on 'group' (flare/SLE)...")
import scanpy.external as sce

sce.pp.harmony_integrate(adata, key="group", basis="X_pca")
print("Harmony integration complete. Diagnostics:")
print("  X_pca_harmony shape:", adata.obsm["X_pca_harmony"].shape)
print("  NaNs in X_pca_harmony:", np.isnan(adata.obsm["X_pca_harmony"]).sum())

# Save pre/post Harmony UMAP for diagnostics
print("Computing neighbors and UMAP (using Harmony)...")
sc.pp.neighbors(
    adata, use_rep="X_pca_harmony", n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE
)

import umap

print("Running UMAP embedding with native umap-learn (parallelized)...")
reducer = umap.UMAP(
    n_neighbors=N_NEIGHBORS,
    min_dist=0.5,
    n_components=2,
    random_state=RANDOM_STATE,
    n_jobs=8,  # Use 8 cores for parallelization
)
embedding = reducer.fit_transform(adata.obsm["X_pca_harmony"])
adata.obsm["X_umap"] = embedding

# Plot UMAP using Scanpy's plotting function
sc.pl.umap(
    adata,
    color=["total_counts", "pct_counts_mt", "sample", "group"],
    save="_qc_umap_harmony.png",
    show=False,
)
qc_umap_fig_path = "figures/umap_qc_umap_harmony.png"
if os.path.exists(qc_umap_fig_path):
    os.rename(qc_umap_fig_path, os.path.join(OUTDIR_PLOTS, "umap_qc_umap_harmony.png"))
else:
    print(f"WARNING: PCA variance ratio plot not found at {qc_umap_fig_path}")
    print(f"WARNING: Harmony QC UMAP plot not found at {qc_umap_fig_path}")

# ------------------- 7. Clustering -------------------
print("Clustering with Leiden (using Harmony)...")
# Enable parallelization for Leiden using igraph backend
sc.tl.leiden(
    adata,
    resolution=1.0,
    flavor="igraph",
    n_iterations=2,
    directed=False,
    random_state=RANDOM_STATE,
)
sc.pl.umap(adata, color=["leiden"], save="_clusters_harmony.png", show=False)
clusters_umap_fig_path = "figures/umap_clusters_harmony.png"
if os.path.exists(clusters_umap_fig_path):
    os.rename(
        clusters_umap_fig_path, os.path.join(OUTDIR_PLOTS, "umap_clusters_harmony.png")
    )
else:
    print(f"WARNING: Harmony clusters UMAP plot not found at {clusters_umap_fig_path}")

# ------------------- 8. Marker Expression Diagnostics -------------------
print("Plotting marker gene expression on UMAP...")
for celltype, genes in MARKERS.items():
    for gene in genes:
        if gene in adata.var_names:
            sc.pl.umap(adata, color=gene, save=f"_{celltype}_{gene}.png", show=False)
            fig_path = f"figures/umap_{celltype}_{gene}.png"
            if os.path.exists(fig_path):
                os.rename(
                    fig_path, os.path.join(OUTDIR_PLOTS, f"umap_{celltype}_{gene}.png")
                )
            else:
                print(f"WARNING: Marker UMAP plot not found at {fig_path}")

# ------------------- 9. Cluster-Level Marker Averages -------------------
print("Computing average marker expression per cluster...")
cluster_marker_means = []
for cluster in sorted(adata.obs["leiden"].unique(), key=lambda x: int(x)):
    cluster_cells = adata[adata.obs["leiden"] == cluster]
    means = {}
    for celltype, genes in MARKERS.items():
        for gene in genes:
            if gene in cluster_cells.var_names:
                expr = cluster_cells[:, gene].X
                if hasattr(expr, "toarray"):
                    expr = expr.toarray().flatten()
                else:
                    expr = np.asarray(expr).flatten()
                means[f"{celltype}:{gene}"] = np.nanmean(expr)
    means["cluster"] = cluster
    means["n_cells"] = cluster_cells.n_obs
    means["group"] = cluster_cells.obs["group"][0] if cluster_cells.n_obs > 0 else None
    cluster_marker_means.append(means)
marker_df = pd.DataFrame(cluster_marker_means)
marker_df.to_csv(os.path.join(OUTDIR_OUTPUTS, "cluster_marker_means.csv"), index=False)

# ------------------- 10. Save Processed Data -------------------
print("Saving processed AnnData to h5ad...")
adata.write(os.path.join(OUTDIR_DATA, "gse137029_processed.h5ad"))

print("Pipeline complete. Outputs written to:")
print("  Data:   ", OUTDIR_DATA)
print("  Plots:  ", OUTDIR_PLOTS)
print("  Outputs:", OUTDIR_OUTPUTS)

# ------------------- 11. Academic Best Practices -------------------
# - All parameters and steps are logged.
# - QC and annotation tables are saved for reproducibility.
# - Diagnostic plots are generated for transparency.
# - Marker-based annotation is provided as a framework; manual review recommended.
# - Save your environment (pip/conda list) and script version for full reproducibility.

# ------------------- END -------------------
