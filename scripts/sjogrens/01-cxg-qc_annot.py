#!/usr/bin/env python3
"""
Sjӧgren’s Syndrome Single-Cell Analysis Pipeline (Cellxgene .h5ad)

Performs best-practice QC, normalization, clustering, marker diagnostics, and annotation.
Designed for academic reproducibility and transparency.

INPUT:
    - raw-data/sjogrens-data/31380664-ba9c-49d1-9961-b2bf4f7131a2.h5ad

OUTPUT:
    - Processed AnnData (.h5ad)
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
import sys

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

# --- Logging, versioning, and environment info ---
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
try:
    from utils.common import log_environment, save_version_info, setup_logging
except ImportError:
    from sjogrens.utils.common import log_environment, save_version_info, setup_logging

# ------------------- Configurable Parameters -------------------
DATA_PATH = "../../raw-data/sjogrens-data/31380664-ba9c-49d1-9961-b2bf4f7131a2.h5ad"
OUTDIR_DATA = "../../data/sjogrens-cxg/"
OUTDIR_PLOTS = "../../plots/sjogrens-cxg/"
OUTDIR_OUTPUTS = "../../outputs/sjogrens-cxg/"
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


def main():
    os.makedirs(OUTDIR_DATA, exist_ok=True)
    os.makedirs(OUTDIR_PLOTS, exist_ok=True)
    os.makedirs(OUTDIR_OUTPUTS, exist_ok=True)
    logger = setup_logging()
    save_version_info(OUTDIR_OUTPUTS)
    log_environment(logger)

    logger.info(f"Loading AnnData from: {DATA_PATH}")
    adata = sc.read_h5ad(DATA_PATH)
    logger.info(f"AnnData loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")
    logger.info(f"Obs columns: {list(adata.obs.columns)}")
    logger.info(f"Var columns: {list(adata.var.columns)}")

    # 2. QC: Mitochondrial Genes
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    # 3. Filter Cells & Genes
    logger.info(
        f"Filtering cells with <{MIN_GENES_PER_CELL} genes and >{MAX_MITO_PCT}% mito..."
    )
    initial_n_cells = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=MIN_GENES_PER_CELL)
    adata = adata[adata.obs["pct_counts_mt"] < MAX_MITO_PCT, :]
    logger.info(
        f"Cells after filtering: {adata.n_obs} (removed {initial_n_cells - adata.n_obs})"
    )

    logger.info(f"Filtering genes detected in <{MIN_CELLS_PER_GENE} cells...")
    initial_n_genes = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)
    logger.info(
        f"Genes after filtering: {adata.n_vars} (removed {initial_n_genes - adata.n_vars})"
    )

    # Save QC summary
    qc_summary = adata.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]]
    qc_summary.to_csv(os.path.join(OUTDIR_OUTPUTS, "qc_summary.csv"))

    # 4. Normalization & Log1p
    if adata.X.min() < 0:
        logger.warning(
            "Data appears log-transformed (negative values present). Skipping normalization/log1p."
        )
    else:
        logger.info(f"Normalizing to {int(TARGET_SUM)} counts/cell and log1p...")
        sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
        sc.pp.log1p(adata)

    # 5. Highly Variable Genes
    logger.info("Selecting highly variable genes (diagnostic only)...")
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES, flavor="seurat_v3")
    hvg = adata.var[adata.var["highly_variable"]]
    logger.info(f"Number of HVGs: {hvg.shape[0]}")
    sc.pl.highly_variable_genes(adata, save="_hvg.png", show=False)
    hvg_fig_path = "figures/filter_genes_dispersion_hvg.png"
    if os.path.exists(hvg_fig_path):
        os.rename(
            hvg_fig_path, os.path.join(OUTDIR_PLOTS, "highly_variable_genes_hvg.png")
        )
    else:
        logger.warning(f"HVG plot not found at {hvg_fig_path}")

    # 6. Scaling, PCA, Neighbors, UMAP
    logger.info("Scaling data (max_value=10)...")
    sc.pp.scale(adata, max_value=10)
    logger.info("Running PCA...")
    sc.tl.pca(adata, svd_solver="arpack", n_comps=N_PCS, random_state=RANDOM_STATE)
    sc.pl.pca_variance_ratio(adata, log=True, save="_pca_var.png", show=False)
    pca_fig_path = "figures/pca_variance_ratio_pca_var.png"
    if os.path.exists(pca_fig_path):
        os.rename(
            pca_fig_path, os.path.join(OUTDIR_PLOTS, "pca_variance_ratio_pca_var.png")
        )
    else:
        logger.warning(f"PCA variance ratio plot not found at {pca_fig_path}")

    logger.info("Computing neighbors and UMAP...")
    sc.pp.neighbors(
        adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS, random_state=RANDOM_STATE
    )
    sc.tl.umap(adata, random_state=RANDOM_STATE)
    sc.pl.umap(
        adata, color=["total_counts", "pct_counts_mt"], save="_qc_umap.png", show=False
    )
    qc_umap_fig_path = "figures/umap_qc_umap.png"
    if os.path.exists(qc_umap_fig_path):
        os.rename(qc_umap_fig_path, os.path.join(OUTDIR_PLOTS, "umap_qc_umap.png"))
    else:
        logger.warning(f"QC UMAP plot not found at {qc_umap_fig_path}")

    # 7. Clustering
    logger.info("Clustering with Leiden...")
    sc.tl.leiden(adata, resolution=1.0, random_state=RANDOM_STATE)
    sc.pl.umap(adata, color=["leiden"], save="_clusters.png", show=False)
    clusters_umap_fig_path = "figures/umap_clusters.png"
    if os.path.exists(clusters_umap_fig_path):
        os.rename(
            clusters_umap_fig_path, os.path.join(OUTDIR_PLOTS, "umap_clusters.png")
        )
    else:
        logger.warning(f"Clusters UMAP plot not found at {clusters_umap_fig_path}")

    # 8. Marker Expression Diagnostics
    logger.info("Plotting marker gene expression on UMAP...")
    for celltype, genes in MARKERS.items():
        for gene in genes:
            if gene in adata.var_names:
                sc.pl.umap(
                    adata, color=gene, save=f"_{celltype}_{gene}.png", show=False
                )
                fig_path = f"figures/umap_{celltype}_{gene}.png"
                if os.path.exists(fig_path):
                    os.rename(
                        fig_path,
                        os.path.join(OUTDIR_PLOTS, f"umap_{celltype}_{gene}.png"),
                    )
                else:
                    logger.warning(f"Marker UMAP plot not found at {fig_path}")

    # 9. Cluster-Level Marker Averages
    logger.info("Computing average marker expression per cluster...")
    cluster_marker_means = []
    for cluster in sorted(adata.obs["leiden"].unique(), key=lambda x: int(x)):
        cluster_cells = adata[adata.obs["leiden"] == cluster]
        means = {}
        for celltype, genes in MARKERS.items():
            for gene in genes:
                if gene in cluster_cells.var_names:
                    expr = np.asarray(cluster_cells[:, gene].X).flatten()
                    means[f"{celltype}:{gene}"] = np.nanmean(expr)
        means["cluster"] = cluster
        means["n_cells"] = cluster_cells.n_obs
        cluster_marker_means.append(means)
    marker_df = pd.DataFrame(cluster_marker_means)
    marker_df.to_csv(
        os.path.join(OUTDIR_OUTPUTS, "cluster_marker_means.csv"), index=False
    )

    # 10. Save Processed Data
    logger.info("Saving processed AnnData...")
    adata.write(os.path.join(OUTDIR_DATA, "sjogrens_processed.h5ad"))

    logger.info("Pipeline complete. Outputs written to:")
    logger.info(f"  Data:   {OUTDIR_DATA}")
    logger.info(f"  Plots:  {OUTDIR_PLOTS}")
    logger.info(f"  Outputs:{OUTDIR_OUTPUTS}")


if __name__ == "__main__":
    main()
