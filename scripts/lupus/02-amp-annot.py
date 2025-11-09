"""
02-amp-annot.py

Purpose:
    Annotate clusters in the processed AMP Phase 1 merged AnnData file using marker gene enrichment,
    assign cell type labels, and generate annotated UMAP plots for publication and downstream analysis.

Workflow:
    - Load merged AnnData object (output from 01-amp-merge.py)
    - Assign cell type labels to clusters (manual mapping or marker-based scoring)
    - Save annotated AnnData object
    - Generate and save UMAP plots colored by cell type and key marker genes

Inputs:
    - amp_phase1_merged.h5ad (output from 01-amp-merge.py)

Outputs:
    - amp_phase1_annotated.h5ad (AnnData with cell type labels)
    - figures/umap_cell_type.png (UMAP colored by cell type)
    - figures/umap_markers_<celltype>.png (UMAPs for marker genes)

Author: [Your Name]
Date: [YYYY-MM-DD]
"""

import scanpy as sc
import numpy as np
import pandas as pd
import os

# Paths
input_h5ad = "/Users/aumchampaneri/complement-receptor-blockade/data/lupus-amp/amp_phase1_merged.h5ad"
output_h5ad = "/Users/aumchampaneri/complement-receptor-blockade/data/lupus-amp/amp_phase1_annotated.h5ad"
figures_dir = "/Users/aumchampaneri/complement-receptor-blockade/plots/lupus-amp"
os.makedirs(figures_dir, exist_ok=True)

# Load processed AnnData
print(f"Loading AnnData from {input_h5ad} ...")
adata = sc.read_h5ad(input_h5ad)

# Marker genes for annotation (renal and immune cell types)
marker_genes = {
    "T_cells": ["CD3D", "CD3E", "CD2", "CD4", "CD8A"],
    "B_cells": ["MS4A1", "CD79A", "CD19"],
    "Plasma_cells": ["MZB1", "SDC1", "IGJ"],
    "Monocytes_Macrophages": ["CD14", "LYZ", "CD68", "CD163", "MRC1"],
    "Dendritic_cells": ["FCER1A", "ITGAX", "CLEC9A"],
    "NK_cells": ["GNLY", "NKG7", "KLRD1"],
    "Neutrophils": ["S100A8", "S100A9", "FCGR3B"],
    "Podocytes": ["NPHS1", "WT1", "SYNPO", "PODXL"],
    "Proximal_tubule": ["SLC5A12", "SLC5A2", "CUBN", "LRP2", "ALDOB"],
    "Distal_tubule": ["SLC12A3", "PVALB", "CALB1"],
    "Collecting_duct": ["AQP2", "AVPR2", "SCNN1G", "SLC14A2"],
    "Endothelial_cells": ["PECAM1", "VWF", "CDH5"],
    "Fibroblasts": ["COL1A1", "DCN", "PDGFRA"],
    "Parietal_epithelial": ["CLDN1", "PAX8"],
    "Intercalated_cells": ["ATP6V1B1", "SLC4A1"],
    "Principal_cells": ["AQP2", "AVPR2"]
}

# --- Manual annotation block ---
# After inspecting UMAPs for marker genes, assign clusters to cell types.
# Example mapping (replace with your assignments after reviewing marker plots):
cluster_to_celltype = {
    '0': 'Podocyte',
    '1': 'Proximal_tubule',
    '2': 'Distal_tubule',
    '3': 'T_cells',
    '4': 'B_cells',
    '5': 'Monocytes_Macrophages',
    '6': 'Endothelial_cells',
    '7': 'Fibroblasts',
    '8': 'Proximal_tubule',
    '9': 'Parietal_epithelial',
    '10': 'Fibroblasts',
    '11': 'Fibroblasts',
    '12': 'Parietal_epithelial',
    '13': 'Fibroblasts',
    '14': 'Parietal_epithelial',
    '15': 'Parietal_epithelial',
    '16': 'Plasma_cells',
    '17': 'Distal_tubule',
    '18': 'Fibroblasts',
    '19': 'Parietal_epithelial',
    '20': 'Collecting_duct',
    '21': 'Distal_tubule'
}

# Diagnostics: Print all unique cluster labels and mapping keys
print("Unique cluster labels in adata.obs['leiden']:", adata.obs['leiden'].unique())
print("Mapping keys:", list(cluster_to_celltype.keys()))

# Assign cell type labels (ensure mapping uses string keys)
print("Assigning cell type labels to clusters...")
adata.obs['cell_type'] = adata.obs['leiden'].astype(str).map(cluster_to_celltype).astype("category")

# Print clusters still NA after assignment
na_cells = adata.obs[adata.obs['cell_type'].isna()]
print("Clusters with NA cell types after assignment:", na_cells['leiden'].unique())

# Save annotated AnnData object
print(f"Saving annotated AnnData to {output_h5ad} ...")
adata.write(output_h5ad)

# Diagnostics: Identify clusters with NA cell types
na_cells = adata.obs[adata.obs['cell_type'].isna()]
na_clusters = na_cells['leiden'].unique()
print("Clusters with NA cell types:", na_clusters)
print("Number of NA cells per cluster:")
print(na_cells['leiden'].value_counts())

# Print marker gene expression for NA clusters
if len(na_clusters) > 0:
    print("Marker gene expression for clusters with NA cell types:")
    for cluster in na_clusters:
        print(f"\nCluster {cluster}:")
        cluster_cells = adata[adata.obs['leiden'] == cluster]
        for celltype, genes in marker_genes.items():
            present_genes = [gene for gene in genes if gene in adata.var_names]
            if present_genes:
                mean_expr = cluster_cells[:, present_genes].X.mean(axis=0)
                print(f"  {celltype}:")
                for gene, expr in zip(present_genes, mean_expr):
                    print(f"    {gene}: {expr:.2f}")

# Plot UMAP colored by cell type
print("Plotting UMAP colored by cell type...")
sc.pl.umap(adata, color='cell_type', save="_cell_type.png", show=False)
if os.path.exists("figures/umap_cell_type.png"):
    os.rename("figures/umap_cell_type.png", os.path.join(figures_dir, "umap_cell_type.png"))
elif os.path.exists("umap_cell_type.png"):
    os.rename("umap_cell_type.png", os.path.join(figures_dir, "umap_cell_type.png"))

# Plot UMAP colored by cohort (dataset origin)
print("Plotting UMAP colored by cohort (dataset origin)...")
sc.pl.umap(adata, color='cohort', save="_cohort.png", show=False)
if os.path.exists("figures/umap_cohort.png"):
    os.rename("figures/umap_cohort.png", os.path.join(figures_dir, "umap_cohort.png"))
elif os.path.exists("umap_cohort.png"):
    os.rename("umap_cohort.png", os.path.join(figures_dir, "umap_cohort.png"))

# Plot UMAPs for marker genes per cell type
print("Plotting UMAPs for marker genes...")
for celltype, genes in marker_genes.items():
    present_genes = [gene for gene in genes if gene in adata.var_names]
    if present_genes:
        sc.pl.umap(adata, color=present_genes, save=f"_markers_{celltype}.png", show=False)
        # Move plot to figures directory
        plot_name = f"figures/umap_markers_{celltype}.png"
        if os.path.exists(plot_name):
            os.rename(plot_name, os.path.join(figures_dir, f"umap_markers_{celltype}.png"))
        elif os.path.exists(f"umap_markers_{celltype}.png"):
            os.rename(f"umap_markers_{celltype}.png", os.path.join(figures_dir, f"umap_markers_{celltype}.png"))
    else:
        print(f"Warning: No marker genes found for {celltype} in dataset.")

print("Annotation and plotting complete. Review figures in the figures/ directory.")
