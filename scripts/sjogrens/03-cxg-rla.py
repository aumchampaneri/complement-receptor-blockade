#!/usr/bin/env python3
"""
Receptor-Ligand Analysis for Complement Cascade Genes in Sjögren’s Syndrome
Uses LIANA to infer cell–cell communication events involving complement receptors and ligands.

INPUT:
    - AnnData: /Users/aumchampaneri/complement-receptor-blockade/data/sjogrens-cxg/sjogrens_processed.h5ad
    - Complement gene list: /Users/aumchampaneri/complement-receptor-blockade/resources/complement-genes.txt

OUTPUT:
    - Filtered LIANA results (complement pairs): /Users/aumchampaneri/complement-receptor-blockade/outputs/sjogrens-cxg/rla/liana_complement_pairs.csv
    - Top interaction plot: /Users/aumchampaneri/complement-receptor-blockade/plots/sjogrens-cxg/rla/liana_complement_top_pairs.png

ENVIRONMENT:
    - Python >=3.8
    - liana, scanpy, pandas, matplotlib

AUTHOR: Automated Pipeline
DATE: 2024-06
"""

import os

import matplotlib
import pandas as pd
import scanpy as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import liana as li
except ImportError:
    raise ImportError("Please install liana: pip install liana")

# ------------------- Configurable Paths -------------------
DATA_PATH = "/Users/aumchampaneri/complement-receptor-blockade/data/sjogrens-cxg/sjogrens_processed.h5ad"
GENE_LIST_PATH = (
    "/Users/aumchampaneri/complement-receptor-blockade/resources/complement-genes.txt"
)
OUTDIR = "/Users/aumchampaneri/complement-receptor-blockade/outputs/sjogrens-cxg/rla/"
PLOT_DIR = "/Users/aumchampaneri/complement-receptor-blockade/plots/sjogrens-cxg/rla/"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ------------------- Load Data -------------------
print("Loading AnnData from:", DATA_PATH)
adata = sc.read_h5ad(DATA_PATH)
if "cell_type" not in adata.obs.columns:
    raise ValueError("AnnData must have 'cell_type' annotation in .obs.")

print("Loading complement gene list from:", GENE_LIST_PATH)
complement_genes = pd.read_csv(GENE_LIST_PATH, header=None)[0].str.strip().tolist()
print(f"Loaded {len(complement_genes)} complement genes.")

# ------------------- Use all cell types for broad LIANA resource analysis -------------------
print("Using all cell types for LIANA resource analysis.")

# ------------------- Map Ensembl IDs to Gene Symbols -------------------
import mygene

print("Mapping Ensembl IDs to HGNC gene symbols using mygene...")
mg = mygene.MyGeneInfo()
ensembl_ids = adata.var_names.tolist()
query_results = mg.querymany(
    ensembl_ids,
    scopes="ensembl.gene",
    fields="symbol",
    species="human",
    as_dataframe=True,
)
id_to_symbol = query_results["symbol"].to_dict()
adata.var["gene_symbol"] = [id_to_symbol.get(eid, "") for eid in adata.var_names]

# Prepare complement gene list for filtering after LIANA
present_complement_genes = [
    g.upper().strip()
    for g in complement_genes
    if g.upper().strip() in [str(gs).upper().strip() for gs in adata.var["gene_symbol"]]
]
print(
    f"Will filter for {len(present_complement_genes)} present complement genes after LIANA."
)

# Remap AnnData var_names to gene symbols for LIANA built-in resource compatibility
adata_symbol = adata.copy()
adata_symbol.var_names = adata_symbol.var["gene_symbol"]
adata_symbol = adata_symbol[:, pd.Series(adata_symbol.var_names).notnull()]
adata_symbol = adata_symbol[:, pd.Series(adata_symbol.var_names) != ""]
adata_symbol = adata_symbol[:, ~pd.Series(adata_symbol.var_names).duplicated()]

# ------------------- Run LIANA with built-in resource -------------------
resource_name = "cellphonedb"
print(f"Using LIANA resource: {resource_name}")
li_resource = li.resource.select_resource(resource_name)

print("Running LIANA for receptor-ligand analysis (broad resource)...")
li.mt.rank_aggregate(
    adata_symbol,
    groupby="cell_type",
    resource=li_resource,
    n_perms=100,
    verbose=True,
    use_raw=False,
)
if "liana_res" not in adata_symbol.uns:
    raise RuntimeError("LIANA results not found in adata_symbol.uns['liana_res'].")

results = adata_symbol.uns["liana_res"]
print(f"LIANA returned {len(results)} interactions.")

# ------------------- Filter for Complement Cascade -------------------
complement_pairs = results[
    results["ligand_complex"].str.upper().isin(present_complement_genes)
    | results["receptor_complex"].str.upper().isin(present_complement_genes)
].copy()
print(f"Found {len(complement_pairs)} complement-related interactions.")

# Save filtered results
out_csv = os.path.join(OUTDIR, "liana_complement_pairs.csv")
complement_pairs.to_csv(out_csv, index=False)
print(f"Saved filtered complement pairs to {out_csv}")


ligand_strength = (
    complement_pairs.groupby("ligand_complex")["lrscore"]
    .mean()
    .sort_values(ascending=False)
)
receptor_strength = (
    complement_pairs.groupby("receptor_complex")["lrscore"]
    .mean()
    .sort_values(ascending=False)
)

# Define top_sender and top_receiver for reporting and plotting
# Use the most frequent sender and receiver cell types in complement_pairs
top_sender = (
    complement_pairs["source"].value_counts().idxmax()
    if "source" in complement_pairs.columns
    else "unknown_sender"
)
top_receiver = (
    complement_pairs["target"].value_counts().idxmax()
    if "target" in complement_pairs.columns
    else "unknown_receiver"
)

print(f"\nTop ligand genes driving interactions from {top_sender} to {top_receiver}:")
print(ligand_strength.head(10))
print(f"\nTop receptor genes driving interactions from {top_sender} to {top_receiver}:")
print(receptor_strength.head(10))

# --- Optional: Barplot of top driving genes ---
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
ligand_strength.head(10).plot(kind="barh", color="teal")
plt.xlabel("Mean LIANA Score")
plt.title(f"Top Ligand Genes: {top_sender} → {top_receiver}")
plt.gca().invert_yaxis()
plt.tight_layout()
ligand_bar_path = os.path.join(
    PLOT_DIR, f"micro_ligand_{top_sender}_to_{top_receiver}.png"
)
plt.savefig(ligand_bar_path)
plt.close()
print(f"Saved ligand gene barplot to {ligand_bar_path}")

plt.figure(figsize=(8, 4))
receptor_strength.head(10).plot(kind="barh", color="purple")
plt.xlabel("Mean LIANA Score")
plt.title(f"Top Receptor Genes: {top_sender} → {top_receiver}")
plt.gca().invert_yaxis()
plt.tight_layout()
receptor_bar_path = os.path.join(
    PLOT_DIR, f"micro_receptor_{top_sender}_to_{top_receiver}.png"
)
plt.savefig(receptor_bar_path)
plt.close()
print(f"Saved receptor gene barplot to {receptor_bar_path}")

# --- Publication-ready heatmap for top complement cell types ---
import seaborn as sns

# Focused cell types (from summary and biology)
focused_cell_types = [
    "myoepithelial cell",
    "duct epithelial cell",
    "fibroblast",
    "dendritic cell",
    "ionocyte",
    "IgG plasma cell",
]

print("Receptor-ligand analysis complete.")
