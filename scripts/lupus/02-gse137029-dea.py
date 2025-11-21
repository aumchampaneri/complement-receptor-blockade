#!/usr/bin/env python3

"""
02-gse137029-dea.py

Pseudo-bulk Differential Expression Analysis for GSE137029 (Lupus PBMC, Harmony-corrected)
==========================================================================================
- Aggregates raw counts per gene per cell type per sample (pseudo-bulk)
- Runs DE using PyDESeq2 (models biological replicates)
- Outputs per-cell-type DE results and sample metadata

INPUT:
    - AnnData (processed & annotated): data/lupus-gse137029/gse137029_qc_harmony.h5ad

OUTPUT:
    - Pseudo-bulk count matrices (one per cell type): outputs/lupus-gse137029/dea/pseudobulk_{celltype}.csv
    - Sample metadata table: outputs/lupus-gse137029/dea/pseudobulk_sample_metadata.csv
    - Per-cell-type DE results: outputs/lupus-gse137029/dea/pseudobulk_{celltype}_DESeq2_py.csv

ENVIRONMENT:
    - Python >=3.8
    - scanpy >=1.9
    - anndata, pandas, numpy, pydeseq2

AUTHOR: Expert Engineer
DATE: 2024-06
"""

import logging
import os
import re
import sys

import numpy as np
import pandas as pd
import scanpy as sc



def make_unique(names):
    counts = {}
    result = []
    for name in names:
        if name not in counts:
            counts[name] = 0
            result.append(name)
        else:
            counts[name] += 1
            result.append(f"{name}.{counts[name]}")
    return result


# ------------------- Configurable Parameters -------------------
DATA_PATH = "/Users/aumchampaneri/complement-receptor-blockade/data/lupus-gse137029/gse137029_qc_harmony.h5ad"
OUTDIR = (
    "/Users/aumchampaneri/complement-receptor-blockade/outputs/lupus-gse137029/dea/"
)
os.makedirs(OUTDIR, exist_ok=True)

# Required metadata columns in .obs
REQUIRED_OBS = ["disease", "cell_type", "sample_uuid"]


# ------------------- Logging Setup -------------------
def setup_logging():
    logger = logging.getLogger("gse137029_dea")
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
    samples = adata.obs["sample_uuid"].unique()
    logger.info(
        f"Found {len(cell_types)} cell types and {len(samples)} samples (sample_uuid)."
    )

    # DEBUG: Print adata.var columns and head to inspect gene identifiers
    print("\n[DEBUG] adata.var columns:", list(adata.var.columns))
    print("[DEBUG] adata.var.head():")
    print(adata.var.head())

    sample_metadata = []
    # Use 'feature_name' as gene symbol (from adata.var)
    if "feature_name" in adata.var.columns:
        gene_symbols = adata.var["feature_name"].astype(str)
        gene_symbols.index = adata.var_names
    else:
        raise ValueError(
            "feature_name column not found in adata.var; cannot assign gene symbols."
        )

    for cell_type in cell_types:
        logger.info(f"Aggregating pseudo-bulk counts for cell type: {cell_type}")
        adata_ct = adata[adata.obs["cell_type"] == cell_type]
        samples_in_ct = adata_ct.obs["sample_uuid"].unique()
        # Use mapped gene symbols for this cell type if available, else fallback to gene_symbols
        gene_symbol_index = (
            mapped_gene_symbols if "mapped_gene_symbols" in locals() else gene_symbols
        )
        gene_names = list(gene_symbol_index.loc[adata_ct.var_names])
        pseudobulk_matrix = pd.DataFrame(
            index=gene_names, columns=samples_in_ct, dtype=float
        )
        for sample in samples_in_ct:
            mask = adata_ct.obs["sample_uuid"] == sample
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
            # Map gene symbols for this subset if available, else fallback to gene_symbols
            sub_gene_symbol_index = (
                mapped_gene_symbols
                if "mapped_gene_symbols" in locals()
                else gene_symbols
            )
            sub_gene_names = list(sub_gene_symbol_index.loc[adata_sub.var_names])
            gene_idx_map = {g: i for i, g in enumerate(sub_gene_names)}
            for gene in gene_names:
                idx = gene_idx_map[gene]
                pseudobulk_matrix.at[gene, sample] = summed[idx]
            disease = adata_sub.obs["disease"].iloc[0]
            sample_metadata.append(
                {
                    "sample": f"{cell_type}__{sample}",
                    "sample_uuid": sample,
                    "celltype": cell_type,
                    "disease": disease,
                }
            )
        # Ensure unique gene names in the index (append .1, .2, etc. to duplicates)
        pseudobulk_matrix.index = pd.Index(pseudobulk_matrix.index).astype(str)
        pseudobulk_matrix.index = pseudobulk_matrix.index.map(str)
        pseudobulk_matrix.index = pd.Index(make_unique(list(pseudobulk_matrix.index)))
        safe_celltype = re.sub(r"[^a-zA-Z0-9_]", "_", str(cell_type))
        out_csv = os.path.join(outdir, f"pseudobulk_{safe_celltype}.csv")
        pseudobulk_matrix = pseudobulk_matrix.fillna(0)
        # Enforce non-negative integer counts
        pseudobulk_matrix[pseudobulk_matrix < 0] = 0
        pseudobulk_matrix = pseudobulk_matrix.astype(int)
        pseudobulk_matrix.index.name = "gene"
        pseudobulk_matrix.to_csv(out_csv, index=True)
        logger.info(f"  Saved pseudo-bulk count matrix: {out_csv}")
        # Print first few gene names for diagnostics
        logger.info(
            f"First 10 gene names in pseudobulk matrix for {cell_type}: {list(pseudobulk_matrix.index[:10])}"
        )
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
        ].set_index("sample_uuid")
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
    logger.info(f"Loading AnnData from: {DATA_PATH}")
    adata = sc.read_h5ad(DATA_PATH)
    logger.info(f"AnnData loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")
    check_metadata(adata, REQUIRED_OBS)
    sample_metadata_df, cell_types = aggregate_pseudobulk(adata, OUTDIR)
    run_deseq2(cell_types, OUTDIR, sample_metadata_df)


if __name__ == "__main__":
    main()
