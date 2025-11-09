#!/usr/bin/env python3
"""
Inspect AnnData .obs columns and show example values
For: sjogrens_processed.h5ad (Sjögren’s CellxGene data)

This script loads the processed AnnData file and prints:
- All column names in adata.obs
- The first few rows of each column (to help identify disease/control labels)

Usage:
    python3 00-inspect-obs.py

Author: Expert Engineer
Date: Auto-generated
"""

import os
import sys

import pandas as pd
import scanpy as sc

# --- Logging, versioning, and environment info ---
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
try:
    from utils.common import log_environment, save_version_info, setup_logging
except ImportError:
    # fallback if running from root or utils not found
    from sjogrens.utils.common import log_environment, save_version_info, setup_logging


def inspect_obs_columns(adata, logger):
    logger.info("Columns in adata.obs:")
    for col in adata.obs.columns:
        logger.info(f"  - {col}")

    logger.info("\nExample values for each column:")
    for col in adata.obs.columns:
        logger.info(f"\nColumn: {col}")
        logger.info(f"{adata.obs[col].value_counts(dropna=False).head(10)}")
        logger.info(f"{adata.obs[col].head(5).to_list()}")


def inspect_var_names(adata, logger):
    logger.info("\nFirst 20 gene names in adata.var_names:")
    logger.info(f"{adata.var_names[:20].to_list()}")


def inspect_celltypist_label(adata, logger):
    if "celltypist_label" in adata.obs.columns:
        logger.info("\nUnique cell types in adata.obs['celltypist_label']:")
        logger.info(f"{adata.obs['celltypist_label'].unique()}")
        logger.info("\nValue counts for each cell type:")
        logger.info(f"{adata.obs['celltypist_label'].value_counts()}")


def main():
    DATA_PATH = "/Users/aumchampaneri/complement-receptor-blockade/data/sjogrens-gse157278/gse157278_processed_celltypist.h5ad"
    OUTDIR = os.path.dirname(DATA_PATH)
    logger = setup_logging()
    save_version_info(OUTDIR)
    log_environment(logger)

    logger.info(f"Loading AnnData from: {DATA_PATH}")
    adata = sc.read_h5ad(DATA_PATH)
    logger.info(f"AnnData loaded: {adata.shape[0]} cells × {adata.shape[1]} genes\n")

    inspect_obs_columns(adata, logger)
    inspect_var_names(adata, logger)
    inspect_celltypist_label(adata, logger)

    logger.info(
        "\nDone. Review the above output to identify the disease/control label column."
    )


if __name__ == "__main__":
    main()
