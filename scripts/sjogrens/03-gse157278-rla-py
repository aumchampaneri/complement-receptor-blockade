#!/usr/bin/env python3
"""
Multi-Resource LIANA Receptor-Ligand Analysis & Consensus Aggregation for GSE157278
===================================================================================

- Runs LIANA for all major resources (cellphonedb, cellchatdb, icellnet, connectomedb2020)
- Filters for complement-related interactions and saves per-resource CSVs
- Aggregates all per-resource results into a consensus table (CSV only, no plots)

INPUT:
    - AnnData: /Users/aumchampaneri/complement-receptor-blockade/data/sjogrens-gse157278/gse157278_processed_celltypist.h5ad
    - Complement gene list: /Users/aumchampaneri/complement-receptor-blockade/resources/complement-genes.txt

OUTPUT:
    - Per-resource filtered LIANA results:
        /Users/aumchampaneri/complement-receptor-blockade/outputs/sjogrens-gse157278/rla/liana_complement_pairs_{resource}.csv
    - Consensus table:
        /Users/aumchampaneri/complement-receptor-blockade/outputs/sjogrens-gse157278/rla/consensus_complement_interactions.csv

ENVIRONMENT:
    - Python >=3.8
    - liana, scanpy, pandas, mygene

AUTHOR: Automated Pipeline (multi-resource, GSE157278)
DATE: 2024-06
"""

import os
import pandas as pd
import scanpy as sc

try:
    import liana as li
except ImportError:
    raise ImportError("Please install liana: pip install liana")

# ------------------- Configurable Paths -------------------
DATA_PATH = "/Users/aumchampaneri/complement-receptor-blockade/data/sjogrens-gse157278/gse157278_processed_celltypist.h5ad"
GENE_LIST_PATH = "/Users/aumchampaneri/complement-receptor-blockade/resources/complement-genes.txt"
OUTDIR = "/Users/aumchampaneri/complement-receptor-blockade/outputs/sjogrens-gse157278/rla/"
os.makedirs(OUTDIR, exist_ok=True)

RESOURCES = ["cellphonedb", "cellchatdb", "icellnet", "connectomedb2020"]

# ------------------- Load Data -------------------
print("Loading AnnData from:", DATA_PATH)
adata = sc.read_h5ad(DATA_PATH)
if "celltypist_label" not in adata.obs.columns:
    raise ValueError("AnnData must have 'celltypist_label' annotation in .obs.")

print("Loading complement gene list from:", GENE_LIST_PATH)
complement_genes = pd.read_csv(GENE_LIST_PATH, header=None)[0].str.strip().tolist()
print(f"Loaded {len(complement_genes)} complement genes.")

# ------------------- Diagnostics and Fallback for Gene Symbol Mapping -------------------
print("First 10 genes in AnnData:", list(adata.var_names[:10]))
print("First 10 complement genes:", complement_genes[:10])

# Check overlap
overlap = set(g.upper() for g in adata.var_names) & set(g.upper() for g in complement_genes)
print(f"Number of complement genes present in AnnData: {len(overlap)}")

if len(overlap) > 0:
    print("Gene symbols in AnnData match complement gene list. Using gene symbols directly.")
    adata.var["gene_symbol"] = adata.var_names
else:
    print("Gene symbols in AnnData do not match complement gene list. Attempting Ensembl-to-symbol mapping with mygene...")
    import mygene
    mg = mygene.MyGeneInfo()
    ensembl_ids = adata.var_names.tolist()
    query_results = mg.querymany(
        ensembl_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        as_dataframe=True,
    )
    # Robustly handle missing 'symbol' column
    if "symbol" in query_results.columns:
        id_to_symbol = query_results["symbol"].to_dict()
    else:
        # Fallback: use index as keys, empty string as value
        id_to_symbol = {eid: "" for eid in ensembl_ids}
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

# ------------------- Run LIANA for Each Resource -------------------
for resource_name in RESOURCES:
    print(f"\n=== Running LIANA for resource: {resource_name} ===")
    li_resource = li.resource.select_resource(resource_name)
    adata_resource = adata_symbol.copy()
    try:
        li.mt.rank_aggregate(
            adata_resource,
            groupby="celltypist_label",
            resource=li_resource,
            n_perms=100,
            verbose=True,
            use_raw=False,
        )
    except Exception as e:
        print(f"Error running LIANA for resource {resource_name}: {e}")
        continue

    if "liana_res" not in adata_resource.uns:
        print(
            f"LIANA results not found in adata_resource.uns['liana_res'] for {resource_name}. Skipping."
        )
        continue

    results = adata_resource.uns["liana_res"]
    print(f"LIANA returned {len(results)} interactions for {resource_name}.")

    # Filter for complement genes
    complement_pairs = results[
        results["ligand_complex"].str.upper().isin(present_complement_genes)
        | results["receptor_complex"].str.upper().isin(present_complement_genes)
    ].copy()
    print(
        f"Found {len(complement_pairs)} complement-related interactions for {resource_name}."
    )

    # Save filtered results
    out_csv = os.path.join(OUTDIR, f"liana_complement_pairs_{resource_name}.csv")
    complement_pairs.to_csv(out_csv, index=False)
    print(f"Saved filtered complement pairs to {out_csv}")

print("\nAll LIANA runs complete. Aggregating consensus table...")

# ------------------- Aggregate: Consensus Table (No Plots) -------------------
dfs = []
for res in RESOURCES:
    fpath = os.path.join(OUTDIR, f"liana_complement_pairs_{res}.csv")
    if not os.path.exists(fpath):
        print(f"WARNING: File not found for resource {res}: {fpath}")
        continue
    df = pd.read_csv(fpath)
    df["resource"] = res
    dfs.append(df)
    print(f"Loaded {len(df)} interactions for {res}")

if not dfs:
    raise RuntimeError("No LIANA complement results found for any resource.")

all_df = pd.concat(dfs, ignore_index=True)
print(f"Total interactions (all resources, with duplicates): {len(all_df)}")

# Define a unique interaction by ligand, receptor, source, target
all_df["interaction_id"] = (
    all_df["ligand_complex"].astype(str)
    + "|"
    + all_df["receptor_complex"].astype(str)
    + "|"
    + all_df["source"].astype(str)
    + "|"
    + all_df["target"].astype(str)
)

# For each interaction, list resources supporting it and aggregate scores
grouped = all_df.groupby("interaction_id")
agg_rows = []
for iid, group in grouped:
    row = {
        "ligand_complex": group["ligand_complex"].iloc[0],
        "receptor_complex": group["receptor_complex"].iloc[0],
        "source": group["source"].iloc[0],
        "target": group["target"].iloc[0],
        "resources": sorted(group["resource"].unique()),
        "n_resources": group["resource"].nunique(),
        "mean_lrscore": group["lrscore"].mean(),
        "max_lrscore": group["lrscore"].max(),
        "min_lrscore": group["lrscore"].min(),
    }
    agg_rows.append(row)
agg_df = pd.DataFrame(agg_rows)
agg_df = agg_df.sort_values(["n_resources", "mean_lrscore"], ascending=[False, False])
agg_df.to_csv(
    os.path.join(OUTDIR, "consensus_complement_interactions.csv"), index=False
)
print(
    f"Saved consensus table to {os.path.join(OUTDIR, 'consensus_complement_interactions.csv')}"
)

print("Multi-resource LIANA analysis and consensus aggregation for GSE157278 complete.")
