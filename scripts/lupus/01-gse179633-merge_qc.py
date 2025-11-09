import gzip
import os

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import umap

# ------------------- Configurable Parameters -------------------
DATA_DIR = (
    "/Users/aumchampaneri/complement-receptor-blockade/raw-data/lupus-data/GSE179633/"
)
OUTDIR_DATA = "/Users/aumchampaneri/complement-receptor-blockade/data/lupus-gse179633/"
OUTDIR_PLOTS = (
    "/Users/aumchampaneri/complement-receptor-blockade/plots/lupus-gse179633/"
)
OUTDIR_OUTPUTS = (
    "/Users/aumchampaneri/complement-receptor-blockade/outputs/lupus-gse179633/"
)
os.makedirs(OUTDIR_DATA, exist_ok=True)
os.makedirs(OUTDIR_PLOTS, exist_ok=True)
os.makedirs(OUTDIR_OUTPUTS, exist_ok=True)

MIN_GENES_PER_CELL = 200
MAX_MITO_PCT = 10.0
MIN_CELLS_PER_GENE = 3
TARGET_SUM = 10000
N_TOP_GENES = 2000
N_PCS = 50
N_NEIGHBORS = 15
RANDOM_STATE = 42

# ------------------- Utility Functions -------------------


def read_tsv(path):
    with gzip.open(path, "rt") as f:
        return [line.strip() for line in f]


def load_10x_sample(matrix_path, barcodes_path, features_path, sample_label):
    print(f"Loading sample: {sample_label}")
    # Read matrix
    mtx = sc.read_mtx(matrix_path).X
    # Read barcodes and features
    barcodes = read_tsv(barcodes_path)
    features = read_tsv(features_path)
    # Auto-detect orientation
    if mtx.shape[0] == len(barcodes) and mtx.shape[1] == len(features):
        # Correct orientation
        pass
    elif mtx.shape[1] == len(barcodes) and mtx.shape[0] == len(features):
        print("Transposing matrix to match AnnData (cells, genes)...")
        mtx = mtx.T
    else:
        raise ValueError(
            f"Matrix shape {mtx.shape} does not match barcodes/features lengths ({len(barcodes)}, {len(features)})"
        )
    # AnnData expects shape (cells, genes)
    adata = anndata.AnnData(mtx)
    adata.obs_names = [f"{sample_label}_{bc}" for bc in barcodes]
    adata.var_names = features
    adata.obs["sample"] = sample_label
    return adata


def find_10x_samples(data_dir):
    # Find all sets of matrix/barcodes/features files
    sample_dict = {}
    for fname in os.listdir(data_dir):
        if fname.endswith("_matrix.mtx.gz"):
            prefix = fname.replace("_matrix.mtx.gz", "")
            matrix_path = os.path.join(data_dir, f"{prefix}_matrix.mtx.gz")
            barcodes_path = os.path.join(data_dir, f"{prefix}_barcodes.tsv.gz")
            features_path = os.path.join(data_dir, f"{prefix}_features.tsv.gz")
            if (
                os.path.exists(matrix_path)
                and os.path.exists(barcodes_path)
                and os.path.exists(features_path)
            ):
                sample_dict[prefix] = {
                    "matrix": matrix_path,
                    "barcodes": barcodes_path,
                    "features": features_path,
                }
    return sample_dict


# ------------------- 1. Auto-detect and Load All Samples -------------------

print("Detecting 10x samples in:", DATA_DIR)
sample_files = find_10x_samples(DATA_DIR)
print(f"Found {len(sample_files)} samples.")

adatas = []
for sample_label, paths in sample_files.items():
    adata = load_10x_sample(
        paths["matrix"], paths["barcodes"], paths["features"], sample_label
    )
    adata.var_names_make_unique()
    adatas.append(adata)

print(f"Loaded {len(adatas)} samples. Concatenating...")

# Merge all samples
adata = adatas[0].concatenate(
    *adatas[1:],
    join="outer",
    batch_key="sample",
    batch_categories=list(sample_files.keys()),
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
print(f"Filtering cells with <{MIN_GENES_PER_CELL} genes and >{MAX_MITO_PCT}% mito...")
initial_n_cells = adata.n_obs
sc.pp.filter_cells(adata, min_genes=MIN_GENES_PER_CELL)
adata = adata[adata.obs["pct_counts_mt"] < MAX_MITO_PCT, :]
print(f"Cells after filtering: {adata.n_obs} (removed {initial_n_cells - adata.n_obs})")

print(f"Filtering genes detected in <{MIN_CELLS_PER_GENE} cells...")
initial_n_genes = adata.n_vars
sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)
print(
    f"Genes after filtering: {adata.n_vars} (removed {initial_n_genes - adata.n_vars})"
)

# Save QC summary
qc_summary = adata.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt", "sample"]]
qc_summary.to_csv(os.path.join(OUTDIR_OUTPUTS, "qc_summary.csv"))

# ------------------- 4. Normalization & Log1p -------------------
if adata.X.min() < 0:
    print(
        "Data appears log-transformed (negative values present). Skipping normalization/log1p."
    )
else:
    print(f"Normalizing to {int(TARGET_SUM)} counts/cell and log1p...")
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
    os.rename(
        hvg_fig_path, os.path.join(OUTDIR_PLOTS, "filter_genes_dispersion_hvg.png")
    )
else:
    print(f"WARNING: HVG plot not found at {hvg_fig_path}")

# ------------------- 6. PCA, Harmony, Neighbors, UMAP -------------------
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

# Harmony batch correction
print("Running Harmony batch correction on 'sample'...")
import scanpy.external as sce

sce.pp.harmony_integrate(adata, key="sample", basis="X_pca")
print("Harmony integration complete. Diagnostics:")
print("  X_pca_harmony shape:", adata.obsm["X_pca_harmony"].shape)
print("  NaNs in X_pca_harmony:", np.isnan(adata.obsm["X_pca_harmony"]).sum())

print("Computing neighbors and UMAP (using Harmony)...")
sc.pp.neighbors(
    adata, use_rep="X_pca_harmony", n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE
)

print("Running UMAP embedding with native umap-learn (parallelized)...")
reducer = umap.UMAP(
    n_neighbors=N_NEIGHBORS,
    min_dist=0.5,
    n_components=2,
    random_state=RANDOM_STATE,
    n_jobs=8,
)
embedding = reducer.fit_transform(adata.obsm["X_pca_harmony"])
adata.obsm["X_umap"] = embedding

sc.pl.umap(
    adata,
    color=["total_counts", "pct_counts_mt", "sample"],
    save="_qc_umap_harmony.png",
    show=False,
)
qc_umap_fig_path = "figures/umap_qc_umap_harmony.png"
if os.path.exists(qc_umap_fig_path):
    os.rename(qc_umap_fig_path, os.path.join(OUTDIR_PLOTS, "umap_qc_umap_harmony.png"))
else:
    print(f"WARNING: Harmony QC UMAP plot not found at {qc_umap_fig_path}")

# ------------------- 7. Clustering -------------------
print("Clustering with Leiden (using Harmony)...")
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

# ------------------- 8. Save Processed Data -------------------
print("Saving processed AnnData to h5ad...")
adata.write(os.path.join(OUTDIR_DATA, "GSE179633_merged_qc.h5ad"))

print("Pipeline complete. Outputs written to:")
print(f"  Data:    {OUTDIR_DATA}")
print(f"  Plots:   {OUTDIR_PLOTS}")
print(f"  Outputs: {OUTDIR_OUTPUTS}")
