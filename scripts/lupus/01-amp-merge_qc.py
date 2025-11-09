"""
01-amp-merge.py

Purpose:
    Merge, QC, and process AMP Phase 1 lupus single-cell RNA-seq datasets (Broad and Metro cohorts).
    Outputs a single AnnData (.h5ad) file for downstream analysis.

Data Provenance:
    - Expression matrices:
        - exprMatrix.tsv: AMP Phase 1, Broad cohort
        - exprMatrix-2.tsv: AMP Phase 1, Metro cohort
    - Metadata:
        - meta.txt: Cluster assignments per cell, decompressed from original gzip
    - All files sourced from: ../raw-data/lupus-data/AMP Phase 1/

Processing Steps:
    - Load and annotate cohorts
    - Merge into a single AnnData object
    - Quality control (min_genes, min_cells, mito content)
    - Diagnostics and metrics saved to CSV
    - Normalization and log1p (if appropriate)
    - Highly variable gene selection (for diagnostics)
    - Scaling, PCA, Harmony batch correction
    - Save processed AnnData object

Outputs:
    - Processed AnnData: amp_phase1_merged.h5ad
    - QC metrics: qc_metrics_amp_phase1.csv
    - Diagnostic plots: figures/ (if desired)

Author: [Your Name]
Date: [YYYY-MM-DD]
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless plotting

import scanpy as sc
import pandas as pd
import numpy as np
import os

# Paths
expr1_path = "/Users/aumchampaneri/complement-receptor-blockade/raw-data/lupus-data/AMP Phase 1/exprMatrix.tsv"
expr2_path = "/Users/aumchampaneri/complement-receptor-blockade/raw-data/lupus-data/AMP Phase 1/exprMatrix-2.tsv"
meta_path = "/Users/aumchampaneri/complement-receptor-blockade/raw-data/lupus-data/AMP Phase 1/meta.txt"
output_path = "/Users/aumchampaneri/complement-receptor-blockade/data/lupus-amp/amp_phase1_merged.h5ad"
qc_metrics_path = "/Users/aumchampaneri/complement-receptor-blockade/outputs/lupus-amp/qc_metrics_amp_phase1.csv"

# Load datasets
print("Loading expression matrices...")
expr1 = pd.read_csv(expr1_path, sep="\t", index_col=0)
expr2 = pd.read_csv(expr2_path, sep="\t", index_col=0)

# Transpose to cells x genes (Scanpy expects cells as rows)
adata1 = sc.AnnData(expr1.T)
adata2 = sc.AnnData(expr2.T)

# Annotate source/cohort
adata1.obs['cohort'] = 'Broad'
adata2.obs['cohort'] = 'Metro'

# Merge datasets
print("Merging datasets...")
adata = adata1.concatenate(adata2, batch_key="cohort", batch_categories=["Broad", "Metro"])

print(f"Initial cells: {adata.n_obs}, genes: {adata.n_vars}")

# QC: Filter cells with fewer than 200 detected genes
sc.pp.filter_cells(adata, min_genes=200)
print(f"After min_genes filter: {adata.n_obs} cells")

# QC: Filter genes detected in fewer than 3 cells
sc.pp.filter_genes(adata, min_cells=3)
print(f"After min_cells filter: {adata.n_vars} genes")

# QC: Mitochondrial filtering
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
print(f"Cells with <10% mito: {(adata.obs['pct_counts_mt'] < 10).sum()} / {adata.n_obs}")
adata = adata[adata.obs['pct_counts_mt'] < 10, :]
print(f"After mito filter: {adata.n_obs} cells")

# Save QC metrics to file
qc_metrics = adata.obs[['n_genes_by_counts', 'total_counts', 'pct_counts_mt']]
qc_metrics.to_csv(qc_metrics_path)

# Remove cells with zero total counts before normalization
zero_count_cells = (adata.X.sum(axis=1) == 0)
print(f"Cells with zero total counts before normalization: {zero_count_cells.sum()}")
adata = adata[~zero_count_cells, :]

# Remove cells with any NaN values before normalization
nan_cells = np.isnan(adata.X).any(axis=1)
print(f"Cells with NaN values before normalization: {nan_cells.sum()}")
adata = adata[~nan_cells, :]

# Diagnostics: Check for negative values before normalization/log1p
print("Min value in adata.X before normalization:", adata.X.min())
print("Max value in adata.X before normalization:", adata.X.max())
if adata.X.min() < 0:
    print("Data appears to be already log-transformed or scaled (contains negative values). Skipping normalization/log1p.")
else:
    print("Normalizing and log-transforming...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

# Remove cells with any NaN values after normalization/log1p
nan_cells_post = np.isnan(adata.X).any(axis=1)
print(f"Cells with NaN values after normalization/log1p: {nan_cells_post.sum()}")
adata = adata[~nan_cells_post, :]

# Highly variable genes (for diagnostics, do not subset)
print("Selecting highly variable genes...")
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Scaling and PCA
print("Scaling and running PCA...")
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')

# Harmony batch correction
print("Running Harmony batch correction...")
import scanpy.external as sce
sce.pp.harmony_integrate(adata, 'cohort')

if 'X_pca_harmony' in adata.obsm:
    print("Harmony-corrected PCA found. Using for downstream analysis.")
    print("Shape of X_pca_harmony:", adata.obsm['X_pca_harmony'].shape)
    print("Any NaNs in X_pca_harmony?", np.isnan(adata.obsm['X_pca_harmony']).any())
    print("Min/Max in X_pca_harmony:", adata.obsm['X_pca_harmony'].min(), adata.obsm['X_pca_harmony'].max())
    adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
else:
    print("WARNING: Harmony-corrected PCA not found. Using default PCA.")

# UMAP and clustering
print("Computing neighbors, UMAP, and clustering after Harmony...")
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
sc.tl.umap(adata)
sc.tl.leiden(adata)

# Load metadata (decompress if needed)
print("Loading metadata...")
meta = pd.read_csv(meta_path, sep="\t", skiprows=2, names=["NAME", "Cluster"])
meta["NAME"] = meta["NAME"].str.replace('\"', '')

# Map cluster assignments to AnnData
print("Mapping cluster assignments to AnnData...")
adata.obs['Cluster'] = adata.obs_names.map(dict(zip(meta["NAME"], meta["Cluster"])))

# Save processed AnnData object
print(f"Saving AnnData object to {output_path} ...")
adata.write(output_path)

print("Processing complete. Output written to", output_path)
print(f"QC metrics written to {qc_metrics_path}")
