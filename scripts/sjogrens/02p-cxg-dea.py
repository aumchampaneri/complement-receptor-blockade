#!/usr/bin/env python3

"""
Best Practices Plotting Script for DESeq2 Pseudo-bulk Results
=============================================================
- Refactored for logging, modularity, versioning, and consistent metadata.

This script visualizes differential expression (DE) results from the pseudo-bulk DE pipeline
for the CellxGene Sjogren's dataset. It generates:
    - Volcano plots for each cell type (highlighting complement genes)
    - Heatmap and clustermap of log2 fold changes (complement genes × cell types)
    - Seurat-style dotplot of -log10(padj) and log2FC for complement genes
    - Side-by-side UMAPs for alternative pathway genes (log1p expression, disease vs control)

Inputs:
    - DESeq2 pseudo-bulk results: pseudobulk_{celltype}_DESeq2_py.csv (one per cell type)
    - Complement gene list: complement-genes.txt
    - (Optional) AnnData object for UMAP plots: sjogrens_processed.h5ad

Outputs:
    - Plots and summary tables in the specified output directory

Author: Automated Pipeline (2024)
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Logging, versioning, and environment info ---
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
try:
    from utils.common import log_environment, save_version_info, setup_logging
except ImportError:
    from sjogrens.utils.common import log_environment, save_version_info, setup_logging


def load_complement_genes(gene_list_path, logger):
    with open(gene_list_path, "r") as f:
        genes = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(genes)} complement genes.")
    return genes


def load_deseq2_results(outdir, logger):
    de_files = sorted(glob.glob(os.path.join(outdir, "pseudobulk_*_DESeq2_py.csv")))
    celltype_to_df = {}
    celltypes = []
    for fpath in de_files:
        celltype = (
            os.path.basename(fpath)
            .replace("pseudobulk_", "")
            .replace("_DESeq2_py.csv", "")
            .replace("_", " ")
        )
        df = pd.read_csv(fpath, index_col=0)
        celltype_to_df[celltype] = df
        celltypes.append(celltype)
    logger.info(f"Loaded DESeq2 results for {len(celltype_to_df)} cell types.")
    return celltype_to_df, celltypes


def plot_volcano(celltype_to_df, complement_genes, plotdir, logger):
    logger.info("Generating volcano plots for each cell type...")
    for celltype, df in celltype_to_df.items():
        plt.figure(figsize=(7, 6))
        df = df.copy()
        df["padj"] = df["padj"].replace(0, np.nan)
        df["-log10_padj"] = -np.log10(df["padj"])
        df["-log10_padj_clipped"] = df["-log10_padj"].clip(upper=10)
        df["log2FoldChange_clipped"] = df["log2FoldChange"].clip(lower=-6, upper=6)
        df["is_complement"] = df.index.isin(complement_genes)
        plt.scatter(
            df["log2FoldChange_clipped"],
            df["-log10_padj_clipped"],
            c=df["is_complement"].map({True: "red", False: "gray"}),
            s=40,
            alpha=0.7,
            edgecolor="k",
            linewidth=0.5,
        )
        sig = (df["padj"] < 0.05) & df["is_complement"] & (~df["padj"].isna())
        top = df[sig].sort_values("padj").head(3)
        for gene, row in top.iterrows():
            plt.text(
                row["log2FoldChange_clipped"],
                row["-log10_padj_clipped"] + 0.1,
                gene,
                fontsize=9,
                ha="center",
                va="bottom",
                color="red",
                fontweight="bold",
            )
        plt.axhline(-np.log10(0.05), color="blue", linestyle="--", lw=1)
        plt.xlabel("log2 Fold Change (Sjogren vs Normal)")
        plt.ylabel("-log10(adjusted p-value)")
        plt.title(f"Volcano Plot: {celltype}")
        plt.tight_layout()
        plt.savefig(os.path.join(plotdir, f"volcano_{celltype.replace(' ', '_')}.png"))
        plt.close()


def plot_heatmap(celltype_to_df, complement_genes, celltypes, plotdir, logger):
    logger.info(
        "Generating heatmap of log2 fold changes (complement genes × cell types)..."
    )
    log2fc_matrix = pd.DataFrame(index=complement_genes, columns=celltypes)
    for celltype, df in celltype_to_df.items():
        log2fc_matrix[celltype] = df["log2FoldChange"].reindex(complement_genes)
    plt.figure(
        figsize=(max(12, len(celltypes) * 0.6), max(8, len(complement_genes) * 0.25))
    )
    sns.heatmap(
        log2fc_matrix.astype(float),
        cmap="vlag",
        center=0,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "log2 Fold Change"},
        vmin=-3,
        vmax=3,
    )
    plt.title("Complement Genes: log2 Fold Change (Sjogren vs Normal) per Cell Type")
    plt.xlabel("Cell Type")
    plt.ylabel("Complement Gene")
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir, "heatmap_log2fc_complement_genes.png"))
    plt.close()

    logger.info("Generating clustermap (dendrogram) for log2 fold changes...")
    import scipy

    sns.clustermap(
        log2fc_matrix.astype(float).fillna(0),
        cmap="vlag",
        center=0,
        linewidths=0.5,
        linecolor="gray",
        figsize=(max(12, len(celltypes) * 0.6), max(8, len(complement_genes) * 0.25)),
        cbar_kws={"label": "log2 Fold Change"},
        vmin=-3,
        vmax=3,
        metric="euclidean",
        method="average",
    )
    plt.suptitle("Clustermap: Complement Genes log2 Fold Change (Sjogren vs Normal)")
    plt.savefig(os.path.join(plotdir, "clustermap_log2fc_complement_genes.png"))
    plt.close()


def plot_dotplot(celltype_to_df, complement_genes, celltypes, plotdir, logger):
    logger.info(
        "Generating Seurat-style dotplot for complement genes across cell types..."
    )
    dotplot_data = []
    for gene in complement_genes:
        for celltype in celltypes:
            df = celltype_to_df[celltype]
            if gene in df.index:
                padj = df.loc[gene, "padj"]
                log2fc = df.loc[gene, "log2FoldChange"]
                sig = int((padj < 0.05) if not pd.isna(padj) else 0)
            else:
                padj = np.nan
                log2fc = np.nan
                sig = 0
            dotplot_data.append(
                {
                    "gene": gene,
                    "celltype": celltype,
                    "log2fc": log2fc,
                    "padj": padj,
                    "sig": sig,
                }
            )
    dotplot_df = pd.DataFrame(dotplot_data)
    dotplot_df["log2fc_capped"] = dotplot_df["log2fc"].clip(lower=-3, upper=3)
    plt.figure(
        figsize=(max(12, len(celltypes) * 0.6), max(8, len(complement_genes) * 0.25))
    )
    sc = plt.scatter(
        x=pd.Categorical(dotplot_df["celltype"], categories=celltypes, ordered=True),
        y=pd.Categorical(dotplot_df["gene"], categories=complement_genes, ordered=True),
        s=dotplot_df["sig"] * 200 + 10,
        c=dotplot_df["log2fc_capped"],
        cmap="vlag",
        vmin=-3,
        vmax=3,
        edgecolor="k",
        alpha=0.8,
    )
    plt.colorbar(sc, label="log2 Fold Change")
    plt.title(
        "Seurat-style Dotplot: Complement Genes × Cell Types\nDot size = significant (padj<0.05), color = log2FC"
    )
    plt.xlabel("Cell Type")
    plt.ylabel("Complement Gene")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir, "dotplot_seuratstyle_complement_genes.png"))
    plt.close()


def build_summary_table(celltype_to_df, plotdir, logger):
    logger.info("Building summary table of top DE complement genes per cell type...")
    summary_rows = []
    for celltype, df in celltype_to_df.items():
        df_valid = df[df["padj"].notna()]
        if not df_valid.empty:
            top_gene = df_valid.loc[df_valid["padj"].idxmin()]
            summary_rows.append(
                {
                    "cell_type": celltype,
                    "gene": df_valid["padj"].idxmin(),
                    "log2FoldChange": top_gene["log2FoldChange"],
                    "padj": top_gene["padj"],
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("padj")
    summary_csv = os.path.join(plotdir, "summary_top_de_complement_genes.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Saved summary table to {summary_csv}")


def plot_umaps_alt_pathway(plotdir, logger):
    logger.info(
        "Generating side-by-side UMAPs for alternative pathway complement genes..."
    )
    import anndata
    import scanpy as sc

    ALT_PATHWAY_GENES = [
        "C3",
        "CFB",
        "CFD",
        "CFH",
        "CFP",
        "C3AR1",
        "C5",
        "C5AR1",
        "C5AR2",
        "ITGAM",
        "ITGAX",
        "VSIG4",
        "CR1",
        "CR2",
    ]
    ANNDATA_PATH = "/Users/aumchampaneri/complement-receptor-blockade/data/sjogrens-gse157278/gse157278_processed_celltypist.h5ad"
    logger.info(f"Loading AnnData from: {ANNDATA_PATH}")
    adata = sc.read_h5ad(ANNDATA_PATH)

    for gene in ALT_PATHWAY_GENES:
        if gene not in adata.var_names:
            logger.warning(f"Gene {gene} not found in AnnData, skipping.")
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        vmax = None
        vmin = None
        expr_vals = []
        for group in ["Sjogren syndrome", "normal"]:
            adata_group = adata[adata.obs["disease"] == group]
            arr = adata_group[:, gene].X
            if hasattr(arr, "toarray"):
                arr = arr.toarray().flatten()
            else:
                arr = np.asarray(arr).flatten()
            arr = arr[arr > 0]
            if arr.size > 0:
                arr = np.log1p(arr)
                expr_vals.append(arr)
        if expr_vals:
            vmin = min([np.min(ev) for ev in expr_vals])
            vmax = max([np.max(ev) for ev in expr_vals])
        for idx, group in enumerate(["Sjogren syndrome", "normal"]):
            adata_group = adata[adata.obs["disease"] == group]
            arr = adata_group[:, gene].X
            if hasattr(arr, "toarray"):
                arr = arr.toarray().flatten()
            else:
                arr = np.asarray(arr).flatten()
            arr = np.log1p(arr)
            adata_group.obs[f"{gene}_log1p"] = arr
            sc.pl.umap(
                adata_group,
                color=f"{gene}_log1p",
                use_raw=False,
                ax=axes[idx],
                show=False,
                vmin=vmin,
                vmax=vmax,
            )
            axes[idx].set_title(f"{gene} (log1p) - {group}")
        plt.suptitle(f"{gene} expression (log1p) UMAP: Sjogren syndrome vs Normal")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(plotdir, f"umap_{gene}_sidebyside_log.png"))
        plt.close()
        logger.info(f"Saved side-by-side log-scaled UMAP for {gene}")


def main():
    OUTDIR = "/Users/aumchampaneri/complement-receptor-blockade/outputs/sjogrens-gse157278/dea/"
    PLOTDIR = "/Users/aumchampaneri/complement-receptor-blockade/plots/sjogrens-gse157278/dea/"
    GENE_LIST_PATH = "/Users/aumchampaneri/complement-receptor-blockade/resources/complement-genes.txt"
    os.makedirs(PLOTDIR, exist_ok=True)
    logger = setup_logging()
    save_version_info(PLOTDIR)
    log_environment(logger)

    complement_genes = load_complement_genes(GENE_LIST_PATH, logger)
    celltype_to_df, celltypes = load_deseq2_results(OUTDIR, logger)
    plot_volcano(celltype_to_df, complement_genes, PLOTDIR, logger)
    plot_heatmap(celltype_to_df, complement_genes, celltypes, PLOTDIR, logger)
    plot_dotplot(celltype_to_df, complement_genes, celltypes, PLOTDIR, logger)
    build_summary_table(celltype_to_df, PLOTDIR, logger)
    plot_umaps_alt_pathway(PLOTDIR, logger)
    logger.info(f"All plots and summary tables generated in: {PLOTDIR}")


if __name__ == "__main__":
    main()
