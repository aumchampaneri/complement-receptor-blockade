#!/usr/bin/env python3

"""
Pseudo-bulk Differential Expression Analysis for Sjogren's CellxGene Data
=========================================================================
Refactored for logging, modularity, versioning, and consistent metadata.

This script implements best practices for single-cell DE by:
- Aggregating raw counts per gene per cell type per donor (pseudo-bulk)
- Running DE using PyDESeq2 (models biological replicates)
- Outputting per-cell-type DE results and sample metadata
- (Optional) Genome-wide DE, with complement gene focus for reporting/plotting

INPUT:
    - AnnData (processed): /Users/aumchampaneri/complement-receptor-blockade/data/sjogrens-cxg/sjogrens_processed.h5ad
    - Complement gene list: /Users/aumchampaneri/complement-receptor-blockade/resources/complement-genes.txt

OUTPUT:
    - Pseudo-bulk count matrices (one per cell type): /Users/aumchampaneri/complement-receptor-blockade/outputs/sjogrens-cxg/dea/pseudobulk_{celltype}.csv
    - Sample metadata table: /Users/aumchampaneri/complement-receptor-blockade/outputs/sjogrens-cxg/dea/pseudobulk_sample_metadata.csv
    - Per-cell-type DE results: /Users/aumchampaneri/complement-receptor-blockade/outputs/sjogrens-cxg/dea/pseudobulk_{celltype}_DESeq2_py.csv

ENVIRONMENT:
    - Python >=3.8
    - scanpy >=1.9
    - anndata, pandas, numpy, pydeseq2

AUTHOR: Automated Pipeline (refactored for pseudo-bulk DE)
DATE: 2024
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import scanpy as sc

# --- Logging, versioning, and environment info ---
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
try:
    from utils.common import log_environment, save_version_info, setup_logging
except ImportError:
    from sjogrens.utils.common import log_environment, save_version_info, setup_logging

# ------------------- Configurable Parameters -------------------
DATA_PATH = "/Users/aumchampaneri/complement-receptor-blockade/data/sjogrens-cxg/sjogrens_processed.h5ad"
OUTDIR = "/Users/aumchampaneri/complement-receptor-blockade/outputs/sjogrens-cxg/dea/"
GENE_LIST_PATH = (
    "/Users/aumchampaneri/complement-receptor-blockade/resources/complement-genes.txt"
)


def load_data(data_path, logger):
    logger.info(f"Loading AnnData from: {data_path}")
    adata = sc.read_h5ad(data_path)
    logger.info(f"AnnData loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")
    return adata


def load_complement_genes(gene_list_path, logger):
    with open(gene_list_path, "r") as f:
        genes = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(genes)} complement genes from {gene_list_path}")
    return genes


def check_metadata(adata, required_obs, logger):
    for col in required_obs:
        if col not in adata.obs.columns:
            logger.error(f"Missing '{col}' in adata.obs. Please add this metadata.")
            raise ValueError(f"Missing '{col}' in adata.obs. Please add this metadata.")
    logger.info(f"All required metadata columns present: {required_obs}")


def aggregate_pseudobulk(adata, outdir, logger):
    cell_types = adata.obs["cell_type"].unique()
    batches = adata.obs["batch"].unique()
    logger.info(f"Found {len(cell_types)} cell types and {len(batches)} donors.")
    sample_metadata = []
    for cell_type in cell_types:
        logger.info(f"Aggregating pseudo-bulk counts for cell type: {cell_type}")
        adata_ct = adata[adata.obs["cell_type"] == cell_type]
        batches_in_ct = adata_ct.obs["batch"].unique()
        gene_names = list(adata.var_names)
        pseudobulk_matrix = pd.DataFrame(
            index=gene_names, columns=batches_in_ct, dtype=float
        )
        for batch in batches_in_ct:
            mask = adata_ct.obs["batch"] == batch
            adata_sub = adata_ct[mask]
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
                pseudobulk_matrix.at[gene, batch] = summed[idx]
            disease = adata_sub.obs["disease"].iloc[0]
            sample_metadata.append(
                {
                    "sample": f"{cell_type}__{batch}",
                    "donor": batch,
                    "celltype": cell_type,
                    "disease": disease,
                }
            )
        safe_celltype = re.sub(r"[^a-zA-Z0-9_]", "_", cell_type)
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


def run_deseq2(cell_types, outdir, sample_metadata_df, logger):
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
        safe_celltype = re.sub(r"[^a-zA-Z0-9_]", "_", cell_type)
        pseudobulk_csv = os.path.join(outdir, f"pseudobulk_{safe_celltype}.csv")
        if not os.path.exists(pseudobulk_csv):
            logger.warning(f"Pseudo-bulk matrix for {cell_type} not found, skipping.")
            continue
        counts = pd.read_csv(pseudobulk_csv, index_col=0)
        meta = sample_metadata_df[
            sample_metadata_df["celltype"] == cell_type
        ].set_index("donor")
        meta = meta.loc[counts.columns]
        counts_T = counts.T
        if meta["disease"].nunique() < 2:
            logger.warning(f"Not enough disease groups for {cell_type}, skipping DE.")
            continue
        logger.info(f"Running DESeq2 for cell type: {cell_type}")
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
        ds = DeseqStats(
            dds,
            contrast=["disease", "Sjogren syndrome", "normal"],
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
    logger = setup_logging()
    save_version_info(OUTDIR)
    log_environment(logger)
    adata = load_data(DATA_PATH, logger)
    complement_genes = load_complement_genes(GENE_LIST_PATH, logger)
    required_obs = ["disease", "cell_type", "batch"]
    check_metadata(adata, required_obs, logger)
    sample_metadata_df, cell_types = aggregate_pseudobulk(adata, OUTDIR, logger)
    run_deseq2(cell_types, OUTDIR, sample_metadata_df, logger)


if __name__ == "__main__":
    main()
