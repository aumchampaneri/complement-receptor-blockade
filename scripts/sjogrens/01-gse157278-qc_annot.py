import os
import sys

import anndata
import celltypist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import seaborn as sns

# --- Logging, versioning, and environment info ---
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
try:
    from utils.common import log_environment, save_version_info, setup_logging
except ImportError:
    from sjogrens.utils.common import log_environment, save_version_info, setup_logging

# ------------------- Configurable Parameters -------------------
RAW_MATRIX_PATH = "/Users/aumchampaneri/complement-receptor-blockade/raw-data/sjogrens-data/GSE157278/matrix.mtx"
RAW_BARCODES_PATH = "/Users/aumchampaneri/complement-receptor-blockade/raw-data/sjogrens-data/GSE157278/barcodes.tsv"
RAW_FEATURES_PATH = "/Users/aumchampaneri/complement-receptor-blockade/raw-data/sjogrens-data/GSE157278/features.tsv"
DATA_PATH = "/Users/aumchampaneri/complement-receptor-blockade/data/sjogrens-gse157278/gse157278_processed.h5ad"
OUTDIR_DATA = (
    "/Users/aumchampaneri/complement-receptor-blockade/data/sjogrens-gse157278/"
)
OUTDIR_PLOTS = (
    "/Users/aumchampaneri/complement-receptor-blockade/plots/sjogrens-gse157278/"
)
OUTDIR_OUTPUTS = (
    "/Users/aumchampaneri/complement-receptor-blockade/outputs/sjogrens-gse157278/"
)
ANNOTATED_PATH = os.path.join(OUTDIR_DATA, "gse157278_processed_celltypist.h5ad")


def main():
    os.makedirs(OUTDIR_DATA, exist_ok=True)
    os.makedirs(OUTDIR_PLOTS, exist_ok=True)
    os.makedirs(OUTDIR_OUTPUTS, exist_ok=True)
    logger = setup_logging()
    save_version_info(OUTDIR_OUTPUTS)
    log_environment(logger)

    # 1. Always reload raw counts and process for CellTypist annotation
    logger.info(
        "Loading raw matrix and running QC pipeline for CellTypist annotation..."
    )
    mtx = scipy.io.mmread(RAW_MATRIX_PATH).tocsc()
    with open(RAW_BARCODES_PATH) as f:
        barcodes = [line.strip() for line in f]
    with open(RAW_FEATURES_PATH) as f:
        features = [line.strip().split("\t")[0] for line in f]
    adata = anndata.AnnData(mtx.transpose())
    adata.obs_names = barcodes
    adata.var_names = features

    logger.info(f"AnnData loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")
    logger.info(f"Obs columns: {list(adata.obs.columns)}")
    logger.info(f"Var columns: {list(adata.var.columns)}")

    # Add disease/control annotation from cell_batch.tsv
    logger.info("Annotating disease/control status using cell_batch.tsv...")
    batch_map = pd.read_csv(
        "/Users/aumchampaneri/complement-receptor-blockade/raw-data/sjogrens-data/GSE157278/cell_batch.tsv",
        sep="\t",
    )
    batch_dict = dict(zip(batch_map["Cell"], batch_map["batch"]))

    def batch_to_disease(batch):
        if batch.startswith("HC-"):
            return "normal"
        elif batch.startswith("pSS-"):
            return "Sjogren syndrome"
        else:
            return "unknown"

    adata.obs["batch"] = adata.obs_names.map(lambda x: batch_dict.get(x, "unknown"))
    adata.obs["disease"] = adata.obs["batch"].map(batch_to_disease)
    logger.info("Disease/control annotation complete. Value counts:")
    logger.info(f"{adata.obs['disease'].value_counts()}")

    # 2. QC: Mitochondrial Genes
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    # 3. Filter Cells & Genes
    MIN_GENES_PER_CELL = 200
    MIN_CELLS_PER_GENE = 3
    MAX_MITO_PCT = 10.0
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
    TARGET_SUM = 1e4
    logger.info(f"Normalizing to {int(TARGET_SUM)} counts/cell and log1p...")
    sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
    sc.pp.log1p(adata)

    # 5. Highly Variable Genes
    N_TOP_GENES = 2000
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

    # 6. PCA, Neighbors, UMAP
    N_PCS = 20
    N_NEIGHBORS = 15
    RANDOM_STATE = 42
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

    # Save processed AnnData for future use
    adata.write(DATA_PATH)
    logger.info(f"Saved processed AnnData to {DATA_PATH}")

    # Ensure log1p normalization for CellTypist
    logger.info(
        "Ensuring log1p normalization to 10,000 counts per cell for CellTypist..."
    )

    # 2. Map Ensembl IDs to gene symbols using mygene
    import mygene

    logger.info("Mapping Ensembl IDs to gene symbols using mygene...")
    mg = mygene.MyGeneInfo()
    result = mg.querymany(
        list(adata.var_names), scopes="ensembl.gene", fields="symbol", species="human"
    )
    mapping = {r["query"]: r.get("symbol", r["query"]) for r in result}
    # Subset to only genes with valid mapped symbols
    symbol_map = {r["query"]: r.get("symbol") for r in result if "symbol" in r}
    valid_genes = [
        g for g in adata.var_names if g in symbol_map and symbol_map[g] is not None
    ]
    adata = adata[:, valid_genes]
    adata.var_names = [symbol_map[g] for g in valid_genes]
    logger.info(
        f"Gene symbol mapping complete. Example gene names: {adata.var_names[:10]}"
    )

    # Remove duplicated gene symbols
    import numpy as np

    logger.info("Removing duplicated gene symbols...")
    _, unique_idx = np.unique(adata.var_names, return_index=True)
    adata = adata[:, unique_idx]
    logger.info(f"Remaining genes after removing duplicates: {adata.shape[1]}")

    # Force .X to dense and fill NaNs with zeros before CellTypist annotation
    logger.info("Filling NaN values in .X with zeros before CellTypist annotation...")
    adata.X = np.nan_to_num(adata.X)
    logger.info(f"Remaining genes after filling NaNs: {adata.shape[1]}")

    # 3. Run CellTypist Automated Annotation
    import numpy as np
    from celltypist.models import Model

    if hasattr(adata, "raw") and adata.raw is not None:
        logger.info("Restoring raw counts from adata.raw for normalization...")
        adata.X = adata.raw.X.copy()

    logger.info("Final normalization and log1p for CellTypist...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    logger.info(
        "Final check: filling any remaining NaNs in .X with zeros before CellTypist annotation..."
    )
    adata.X = np.nan_to_num(adata.X)

    model_dir = os.path.expanduser("~/.celltypist/data/models")
    model_path = os.path.join(model_dir, "Immune_All_Low.pkl")

    if os.path.exists(model_path):
        logger.info("CellTypist model found locally. Loading...")
        model = Model.load(model_path)
    else:
        logger.info("CellTypist model not found locally. Downloading...")
        model = celltypist.models.download_models("Immune_All_Low.pkl")

    logger.info("Running CellTypist annotation...")
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

    logger.info("CellTypist annotation complete. Labels saved to outputs.")

    # 4. UMAP Plotting
    logger.info("Plotting UMAP colored by CellTypist label...")
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
        logger.warning(f"CellTypist UMAP plot not found at {umap_fig_path}")

    # 5. Save Annotated AnnData
    adata.write(ANNOTATED_PATH)
    logger.info(f"Saved annotated AnnData to {ANNOTATED_PATH}")

    logger.info("Automated annotation pipeline complete. Outputs written to:")
    logger.info(f"  Data:    {OUTDIR_DATA}")
    logger.info(f"  Plots:   {OUTDIR_PLOTS}")
    logger.info(f"  Outputs: {OUTDIR_OUTPUTS}")


if __name__ == "__main__":
    main()
