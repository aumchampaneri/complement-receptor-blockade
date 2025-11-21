#!/usr/bin/env python3
"""
-01-gse137029-qc_annot.train_scvi.v1.py
+01-gse137029-qc_annot.py

Pipeline for QC and Harmony batch correction of CellxGene PBMC dataset (GSE137029).
- Extracts raw counts from .h5ad (CellxGene format)
- Performs best-practice QC (mito, gene/cell filtering)
- Applies Harmony batch correction on PCA space
- Retains original cellxgene annotation fields
- Outputs processed AnnData, UMAP, and summary plots

INPUT:
    - /Users/aumchampaneri/complement-receptor-blockade/raw-data/lupus-data/GSE137029/4118e166-34f5-4c1f-9eed-c64b90a3dace.h5ad

OUTPUT:
    - Processed AnnData (.h5ad)
    - UMAP plot (.png)
    - QC summary (.csv)

ENVIRONMENT:
    - Python >=3.8
    - scanpy >=1.9
    - harmonypy
    - anndata, matplotlib, pandas, numpy, scipy

AUTHOR: Expert Engineer
DATE: 2024-06
"""

import logging
import os
import sys
import warnings

import anndata
import matplotlib

matplotlib.use("Agg")
# Suppress non-critical warnings for cleaner logs
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="`flavor='seurat_v3'` expects raw count data, but non-integers were found.",
)
warnings.filterwarnings(
    "ignore", message="zero-centering a sparse array/matrix densifies it."
)

try:
    import harmonypy
except ImportError:
    print("harmonypy is required. Install with: pip install harmonypy")
    sys.exit(1)

# ------------------- Configurable Parameters -------------------
DATA_PATH = "/Users/aumchampaneri/complement-receptor-blockade/raw-data/lupus-data/GSE137029/4118e166-34f5-4c1f-9eed-c64b90a3dace.h5ad"
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

PROCESSED_H5AD = os.path.join(OUTDIR_DATA, "gse137029_qc_harmony.h5ad")
QC_METRICS_CSV = os.path.join(OUTDIR_OUTPUTS, "qc_metrics_gse137029.csv")
UMAP_PNG = os.path.join(OUTDIR_PLOTS, "umap_harmony.png")

MIN_GENES_PER_CELL = 200
MIN_CELLS_PER_GENE = 3
MAX_MITO_PCT = 20.0
N_PCS = 30
N_NEIGHBORS = 15
RANDOM_STATE = 42

# Choose a batch key present in cellxgene obs (e.g., 'sample_uuid', 'donor_id', or 'Processing_Cohort')
BATCH_KEY = "sample_uuid"

# ------------------- Utility Functions -------------------


def extract_raw_counts(adata: anndata.AnnData) -> anndata.AnnData:
    """Extract raw counts with safe guards."""
    if hasattr(adata, "raw") and adata.raw is not None:
        print("Using adata.raw.X for raw counts.")
        raw = anndata.AnnData(
            X=adata.raw.X, obs=adata.obs.copy(), var=adata.raw.var.copy()
        )
        return raw
    if "counts" in getattr(adata, "layers", {}):
        print("Using adata.layers['counts'] for raw counts.")
        a = adata.copy()
        a.X = a.layers["counts"]
        return a
    warnings.warn("No raw counts found; using adata.X (may be normalized).")
    return adata.copy()


def fix_var_index(adata: anndata.AnnData):
    """Ensure .var.index and .var_names are string Index, not CategoricalIndex (cellxgene fix)."""
    adata.var.index = adata.var.index.astype(str)
    adata.var_names = adata.var.index.astype(str)
    adata.var.reset_index(drop=True, inplace=True)
    # Remove any leftover categorical dtypes in adata.var
    for col in adata.var.columns:
        if isinstance(adata.var[col].dtype, pd.CategoricalDtype):
            adata.var[col] = adata.var[col].astype(str)
    return adata


def compute_qc_metrics(adata: anndata.AnnData) -> anndata.AnnData:
    """Compute standard QC metrics via scanpy helper (adds n_genes_by_counts, total_counts, pct_counts_mt)."""
    # Ensure var_names is a pandas Index of strings
    adata.var_names = pd.Index(adata.var_names.astype(str))
    mito_mask = adata.var_names.str.upper().str.startswith("MT-")
    adata.var["mt"] = mito_mask
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    if "pct_counts_mt" not in adata.obs:
        adata.obs["pct_counts_mt"] = 0.0
    return adata


def filter_qc(
    adata: anndata.AnnData, min_genes=200, min_cells=3, max_mito=20.0
) -> anndata.AnnData:
    """Filter cells & genes and remove high-mito cells."""
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if "pct_counts_mt" in adata.obs:
        before = adata.n_obs
        adata = adata[adata.obs["pct_counts_mt"] < max_mito].copy()
        print(f"Filtered {before - adata.n_obs} cells by mitochondrial % > {max_mito}.")
    return adata


# ------------------- Main Pipeline -------------------


def inspect_anndata_structure(adata):
    import sys

    print("\n=== AnnData Structure Inspection ===")
    print(f"Shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
    print("\n.obs columns:")
    print(list(adata.obs.columns))
    print("\n.var columns:")
    print(list(adata.var.columns))
    print("\nSample .obs (first 5 rows):")
    print(adata.obs.head())
    print("\nSample .var (first 5 rows):")
    print(adata.var.head())
    # Show unique values for likely batch fields
    likely_batch_fields = [
        c
        for c in adata.obs.columns
        if any(x in c.lower() for x in ["batch", "sample", "donor", "cohort"])
    ]
    for col in likely_batch_fields:
        print(f"\nUnique values in obs['{col}'] (up to 10 shown):")
        print(adata.obs[col].unique()[:10])
    print("\n=== End of Inspection ===\n")
    sys.exit(0)


def main():
    print("Loading CellxGene AnnData ...")
    adata = sc.read_h5ad(DATA_PATH)
    print(f"Original AnnData: {adata.n_obs} cells × {adata.n_vars} genes")

    # Extract raw counts
    adata = extract_raw_counts(adata)
    print(
        f"AnnData after raw count extraction: {adata.n_obs} cells × {adata.n_vars} genes"
    )

    # Fix CategoricalIndex in .var (cellxgene fix)
    adata = fix_var_index(adata)
    # Ensure gene names are unique for downstream analysis
    adata.var_names_make_unique()

    # Proceed directly to QC and Harmony batch correction using sample_uuid as batch key

    # Compute QC metrics
    adata = compute_qc_metrics(adata)

    # Save QC metrics before filtering
    qc_columns = [
        c
        for c in ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
        if c in adata.obs.columns
    ]
    adata.obs[qc_columns].to_csv(QC_METRICS_CSV)
    print(f"Saved QC metrics to {QC_METRICS_CSV}")

    # Filter cells/genes
    adata = filter_qc(
        adata,
        min_genes=MIN_GENES_PER_CELL,
        min_cells=MIN_CELLS_PER_GENE,
        max_mito=MAX_MITO_PCT,
    )
    print(f"AnnData after QC filtering: {adata.n_obs} cells × {adata.n_vars} genes")

    # Normalize and log1p
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Highly variable genes (optional, but speeds up PCA)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=2000, subset=False, flavor="seurat_v3"
    )
    print(
        f"Annotated {adata.var['highly_variable'].sum()} highly variable genes (all genes retained)."
    )

    # Save processed AnnData with all genes (for DE/pseudobulk)
    adata.write(PROCESSED_H5AD)
    print(f"Saved processed AnnData to {PROCESSED_H5AD} (all genes retained)")

    # Now subset to HVGs for dimensionality reduction
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"Subset to {adata.n_vars} highly variable genes for PCA/Harmony/UMAP.")

    # Scale and PCA
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=N_PCS, svd_solver="arpack", random_state=RANDOM_STATE)
    print("PCA complete.")

    # Harmony batch correction
    if BATCH_KEY not in adata.obs.columns:
        raise ValueError(
            f"BATCH_KEY '{BATCH_KEY}' not found in adata.obs. Available: {list(adata.obs.columns)}"
        )
    print(f"Running Harmony batch correction on '{BATCH_KEY}' ...")
    ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, BATCH_KEY)
    adata.obsm["X_pca_harmony"] = ho.Z_corr.T
    print("Harmony correction complete.")

    # Neighbors and UMAP on Harmony-corrected PCA
    sc.pp.neighbors(
        adata,
        use_rep="X_pca_harmony",
        n_neighbors=N_NEIGHBORS,
        random_state=RANDOM_STATE,
    )
    sc.tl.umap(adata, random_state=RANDOM_STATE)

    # Save processed AnnData
    # (Already saved above, before HVG subsetting)

    # Plot UMAP colored by original cellxgene annotation (if available)
    color_fields = []
    for field in [
        "cell_type",
        "cell_ontology_class",
        "sample_uuid",
        "donor_id",
        "Processing_Cohort",
    ]:
        if field in adata.obs.columns:
            color_fields.append(field)
    if not color_fields:
        color_fields = [None]

    for field in color_fields:
        plt.figure(figsize=(8, 6))
        sc.pl.umap(adata, color=field, show=False)
        plt.title(f"UMAP (Harmony, colored by {field})" if field else "UMAP (Harmony)")
        plt.tight_layout()
        out_png = (
            UMAP_PNG if field is None else UMAP_PNG.replace(".png", f"_{field}.png")
        )
        plt.savefig(out_png, dpi=150)
        print(f"Saved UMAP plot to {out_png}")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
