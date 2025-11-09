#!/usr/bin/env python3

"""
Pseudo-bulk Differential Expression Analysis for EMTAB Lupus Data
=================================================================
- Aggregates raw counts per gene per cell type per donor (pseudo-bulk)
- Runs DE using PyDESeq2 (models biological replicates)
- Outputs per-cell-type DE results and sample metadata

INPUT:
    - AnnData (processed & annotated): data/lupus-emtab/emtab_annotated.h5ad

OUTPUT:
    - Pseudo-bulk count matrices (one per cell type): outputs/lupus-emtab/dea/pseudobulk_{celltype}.csv
    - Sample metadata table: outputs/lupus-emtab/dea/pseudobulk_sample_metadata.csv
    - Per-cell-type DE results: outputs/lupus-emtab/dea/pseudobulk_{celltype}_DESeq2_py.csv

ENVIRONMENT:
    - Python >=3.8
    - scanpy >=1.9
    - anndata, pandas, numpy, pydeseq2

AUTHOR: Automated Pipeline (refactored for pseudo-bulk DE)
DATE: 2024
"""

import logging
import os
import re
import sys

import numpy as np
import pandas as pd
import scanpy as sc

# ------------------- Configurable Parameters -------------------
DATA_PATH = "data/lupus-emtab/emtab_annotated.h5ad"
OUTDIR = "outputs/lupus-emtab/dea/"


# ------------------- Logging Setup -------------------
def setup_logging():
    logger = logging.getLogger("emtab_dea")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger


logger = setup_logging()


# ------------------- Utility Functions -------------------
def check_metadata(adata, required_obs):
    for col in required_obs:
        if col not in adata.obs.columns:
            logger.error(f"Missing '{col}' in adata.obs. Please add this metadata.")
            raise ValueError(f"Missing '{col}' in adata.obs. Please add this metadata.")
    logger.info(f"All required metadata columns present: {required_obs}")


def aggregate_pseudobulk(adata, outdir):
    cell_types = adata.obs["cell_type"].unique()
    donors = adata.obs["sample_id"].unique()
    logger.info(
        f"Found {len(cell_types)} cell types and {len(donors)} donors (sample_id)."
    )

    # --- Diagnostic block: print min/max for .X and .raw.X for a few cell types ---
    logger.info(
        "Diagnostic: Checking min/max values in .X and .raw.X for first 3 cell types..."
    )
    for cell_type in cell_types[:3]:
        adata_ct = adata[adata.obs["cell_type"] == cell_type]
        logger.info(f"  Cell type: {cell_type}")
        if hasattr(adata_ct, "X"):
            X = adata_ct.X
            if hasattr(X, "toarray"):
                arr = X.toarray()
            else:
                arr = X
            logger.info(f"    .X min: {np.min(arr):.4f}, max: {np.max(arr):.4f}")
        if adata_ct.raw is not None:
            Xraw = adata_ct.raw.X
            if hasattr(Xraw, "toarray"):
                arr_raw = Xraw.toarray()
            else:
                arr_raw = Xraw
            logger.info(
                f"    .raw.X min: {np.min(arr_raw):.4f}, max: {np.max(arr_raw):.4f}"
            )
        else:
            logger.info("    .raw is None")

    sample_metadata = []
    for cell_type in cell_types:
        logger.info(f"Aggregating pseudo-bulk counts for cell type: {cell_type}")
        adata_ct = adata[adata.obs["cell_type"] == cell_type]
        donors_in_ct = adata_ct.obs["sample_id"].unique()
        gene_names = list(adata.var_names)
        pseudobulk_matrix = pd.DataFrame(
            index=gene_names, columns=donors_in_ct, dtype=float
        )
        for donor in donors_in_ct:
            mask = adata_ct.obs["sample_id"] == donor
            adata_sub = adata_ct[mask]
            # Use raw counts if available, else X
            X = adata_sub.raw.X if adata_sub.raw is not None else adata_sub.X
            if hasattr(X, "toarray"):
                counts = np.asarray(X.toarray())
            else:
                counts = np.asarray(X)
            if counts.shape[0] == 0:
                continue
            summed = counts.sum(axis=0)
            gene_idx_map = {g: i for i, g in enumerate(adata_sub.var_names)}
            for gene in gene_names:
                idx = gene_idx_map[gene]
                pseudobulk_matrix.at[gene, donor] = summed[idx]
            disease = adata_sub.obs["disease"].iloc[0]
            sample_metadata.append(
                {
                    "sample": f"{cell_type}__{donor}",
                    "donor": donor,
                    "celltype": cell_type,
                    "disease": disease,
                }
            )
        safe_celltype = re.sub(r"[^a-zA-Z0-9_]", "_", str(cell_type))
        out_csv = os.path.join(outdir, f"pseudobulk_{safe_celltype}.csv")
        pseudobulk_matrix = pseudobulk_matrix.fillna(0).astype(int)
        pseudobulk_matrix.to_csv(out_csv)
        logger.info(f"  Saved pseudo-bulk count matrix: {out_csv}")
    sample_metadata_df = pd.DataFrame(sample_metadata)
    sample_metadata_csv = os.path.join(outdir, "pseudobulk_sample_metadata.csv")
    sample_metadata_df.to_csv(sample_metadata_csv, index=False)
    logger.info(f"Saved sample metadata table: {sample_metadata_csv}")
    logger.info("Pseudo-bulk aggregation complete.")
    return sample_metadata_df, cell_types


def run_deseq2(cell_types, outdir, sample_metadata_df):
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except Exception as e:
        logger.error(
            f"pydeseq2 import failed. Exception details below:\n{type(e).__name__}: {e}\n"
            "Please install it with 'pip install pydeseq2' to run DE analysis in Python."
        )
        raise e

    for cell_type in cell_types:
        safe_celltype = re.sub(r"[^a-zA-Z0-9_]", "_", str(cell_type))
        pseudobulk_csv = os.path.join(outdir, f"pseudobulk_{safe_celltype}.csv")
        if not os.path.exists(pseudobulk_csv):
            logger.warning(f"Pseudo-bulk matrix for {cell_type} not found, skipping.")
            continue
        counts = pd.read_csv(pseudobulk_csv, index_col=0)
        meta = sample_metadata_df[
            sample_metadata_df["celltype"] == cell_type
        ].set_index("donor")
        # Ensure meta and counts columns are aligned
        meta = meta.loc[[c for c in counts.columns if c in meta.index]]
        counts = counts[[c for c in counts.columns if c in meta.index]]
        counts_T = counts.T
        if meta["disease"].nunique() < 2:
            logger.warning(f"Not enough disease groups for {cell_type}, skipping DE.")
            continue
        logger.info(f"Running DESeq2 for cell type: {cell_type}")
        # Use the two most frequent disease groups as contrast
        disease_counts = meta["disease"].value_counts().index.tolist()
        if len(disease_counts) < 2:
            logger.warning(f"Not enough disease groups for {cell_type}, skipping DE.")
            continue
        group1, group2 = disease_counts[:2]
        try:
            dds = DeseqDataSet(
                counts=counts_T,
                metadata=meta,
                design="~disease",
                refit_cooks=True,
            )
            dds.fit_size_factors()
            dds.fit_genewise_dispersions()
            dds.fit_dispersion_trend()
            dds.fit_dispersion_prior()
            dds.fit_MAP_dispersions()
            dds.fit_LFC()
            dds.calculate_cooks()
            if dds.refit_cooks:
                dds.refit()
            from pydeseq2.ds import DeseqStats
        except ValueError as e:
            logger.warning(
                f"Skipping {cell_type} due to insufficient replicates or other DESeq2 error: {e}"
            )
            continue

        ds = DeseqStats(
            dds,
            contrast=["disease", group1, group2],
            alpha=0.05,
            cooks_filter=True,
            independent_filter=True,
        )
        ds.run_wald_test()
        ds.summary()
        results = ds.results_df
        out_de_csv = os.path.join(outdir, f"pseudobulk_{safe_celltype}_DESeq2_py.csv")
        results.to_csv(out_de_csv)
        logger.info(f"    Saved DE results to {out_de_csv}")
    logger.info(
        "PyDESeq2 DE analysis complete. You can now use these results for downstream plotting and interpretation."
    )


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    logger.info(f"Loading AnnData from: {DATA_PATH}")
    adata = sc.read_h5ad(DATA_PATH)
    logger.info(f"AnnData loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")
    required_obs = ["disease", "cell_type", "sample_id"]
    check_metadata(adata, required_obs)
    sample_metadata_df, cell_types = aggregate_pseudobulk(adata, OUTDIR)
    run_deseq2(cell_types, OUTDIR, sample_metadata_df)


if __name__ == "__main__":
    main()
