"""
02-emtab-annot.py

Purpose:
    Annotate clusters in the processed E-MTAB-13596 merged AnnData file using marker gene enrichment,
    assign cell type labels, and generate annotated UMAP plots for publication and downstream analysis.

Workflow:
    - Load merged AnnData object (output from 01-emtab-merge_qc.py)
    - Plot UMAPs for marker genes to guide annotation
    - Assign cell type labels to clusters (manual mapping or marker-based scoring)
    - Save annotated AnnData object
    - Generate and save UMAP plots colored by cell type and key marker genes

Inputs:
    - emtab_merged.h5ad (output from 01-emtab-merge_qc.py)

Outputs:
    - emtab_annotated.h5ad (AnnData with cell type labels)
    - UMAP plots: /Users/aumchampaneri/complement-receptor-blockade/plots/lupus-emtab/
    - Marker plots: /Users/aumchampaneri/complement-receptor-blockade/plots/lupus-emtab/

Author: [Your Name]
Date: [YYYY-MM-DD]
"""

import scanpy as sc
import numpy as np
import pandas as pd
import os

# Paths
input_h5ad = "/Users/aumchampaneri/complement-receptor-blockade/data/lupus-emtab/emtab_merged.h5ad"
output_h5ad = "/Users/aumchampaneri/complement-receptor-blockade/data/lupus-emtab/emtab_annotated.h5ad"
figures_dir = "/Users/aumchampaneri/complement-receptor-blockade/plots/lupus-emtab"
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

print("Review the marker UMAPs in the plots directory to manually assign cell types to clusters.")

# --- Manual annotation block ---
# After inspecting UMAPs for marker genes, assign clusters to cell types.
# Example mapping (replace with your assignments after reviewing marker plots):
cluster_to_celltype = {
    '0': 'T_cells',
    '1': 'NK_cells',
    '2': 'NK_cells',
    '3': 'T_cells',
    '4': 'T_cells',
    '5': 'Monocytes_Macrophages',
    '6': 'Monocytes_Macrophages',
    '7': 'T_cells',
    '8': 'Monocytes_Macrophages',
    '9': 'B_cells',
    '10': 'NK_cells',
    '11': 'Neutrophils',
    '12': 'Proximal_tubule',
    '13': 'Plasma_cells',
    '14': 'Monocytes_Macrophages',
    '15': 'Plasma_cells',
    '16': 'Neutrophils',
    '17': 'Dendritic_cells',
    '18': 'Collecting_duct'
}

# Assign cell type labels
print("Assigning cell type labels to clusters...")
adata.obs['cell_type'] = adata.obs['leiden'].astype(str).map(cluster_to_celltype).astype("category")

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

# Save annotated AnnData object
print(f"Saving annotated AnnData to {output_h5ad} ...")
adata.write(output_h5ad)

# Plot UMAP colored by cell type
print("Plotting UMAP colored by cell type...")
sc.pl.umap(adata, color='cell_type', save="_cell_type.png", show=False)
if os.path.exists("figures/umap_cell_type.png"):
    os.rename("figures/umap_cell_type.png", os.path.join(figures_dir, "umap_cell_type.png"))
elif os.path.exists("umap_cell_type.png"):
    os.rename("umap_cell_type.png", os.path.join(figures_dir, "umap_cell_type.png"))

# Plot UMAP colored by batch (dataset origin)
print("Plotting UMAP colored by batch (dataset origin)...")
sc.pl.umap(adata, color='batch', save="_batch.png", show=False)
if os.path.exists("figures/umap_batch.png"):
    os.rename("figures/umap_batch.png", os.path.join(figures_dir, "umap_batch.png"))
elif os.path.exists("umap_batch.png"):
    os.rename("umap_batch.png", os.path.join(figures_dir, "umap_batch.png"))

print("Annotation and plotting complete. Review figures in the plots directory.")
