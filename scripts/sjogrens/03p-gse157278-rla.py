#!/usr/bin/env python3
"""
Aggregate LIANA Complement Results Across Resources & Generate Consensus/Overlap Plots (GSE157278)
==================================================================================================

This script aggregates LIANA receptor-ligand analysis results for complement genes
across multiple resources (cellphonedb, cellchatdb, icellnet, connectomedb2020)
and generates consensus/overlap plots and summary tables for the GSE157278 dataset.

INPUT:
    - Filtered LIANA results per resource:
        /Users/aumchampaneri/complement-receptor-blockade/outputs/sjogrens-gse157278/rla/liana_complement_pairs_{resource}.csv

OUTPUT:
    - Aggregated consensus CSV
    - Barplot: number of resources supporting each interaction
    - UpSet plot: overlap of interactions across resources
    - Consensus heatmap: mean LR score for consensus interactions
    - Table: top consensus interactions

ENVIRONMENT:
    - Python >=3.8
    - pandas, matplotlib, seaborn, upsetplot

AUTHOR: Automated Pipeline (adapted for GSE157278)
DATE: 2024-06
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from upsetplot import UpSet, from_memberships
except ImportError:
    print("upsetplot not found. Install with: pip install upsetplot")
    UpSet = None

# ------------------- Configurable -------------------
RESOURCES = ["cellphonedb", "cellchatdb", "icellnet", "connectomedb2020"]
IN_DIR = (
    "/Users/aumchampaneri/complement-receptor-blockade/outputs/sjogrens-gse157278/rla/"
)
PLOT_DIR = (
    "/Users/aumchampaneri/complement-receptor-blockade/plots/sjogrens-gse157278/rla/"
)
os.makedirs(PLOT_DIR, exist_ok=True)

# ------------------- 1. Load All Results -------------------
dfs = []
for res in RESOURCES:
    fpath = os.path.join(IN_DIR, f"liana_complement_pairs_{res}.csv")
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

# ------------------- 2. Aggregate: Consensus Table -------------------
# Define a unique interaction by ligand, receptor, source, target
interaction_cols = ["ligand_complex", "receptor_complex", "source", "target"]

# For UpSet plot: build membership sets
all_df["interaction_id"] = (
    all_df["ligand_complex"].astype(str)
    + "|"
    + all_df["receptor_complex"].astype(str)
    + "|"
    + all_df["source"].astype(str)
    + "|"
    + all_df["target"].astype(str)
)

# For each interaction, list resources supporting it
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
    os.path.join(PLOT_DIR, "consensus_complement_interactions.csv"), index=False
)
print(
    f"Saved consensus table to {os.path.join(PLOT_DIR, 'consensus_complement_interactions.csv')}"
)

# ------------------- 3. Barplot: Number of Resources Supporting Each Interaction -------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="n_resources", data=agg_df, palette="viridis")
plt.title("Number of Resources Supporting Each Complement Interaction")
plt.xlabel("Number of Resources")
plt.ylabel("Number of Interactions")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "consensus_barplot_n_resources.png"))
plt.close()
print("Saved barplot of resource support.")

# ------------------- 4. UpSet Plot: Overlap of Interactions -------------------
if UpSet is not None:
    # Build membership list for each interaction
    memberships = [tuple(row["resources"]) for _, row in agg_df.iterrows()]
    upset_data = from_memberships(memberships)
    plt.figure(figsize=(10, 6))
    UpSet(upset_data, subset_size="count", show_counts=True).plot()
    plt.suptitle("UpSet Plot: Overlap of Complement Interactions Across Resources")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "consensus_upsetplot.png"))
    plt.close()
    print("Saved UpSet plot of resource overlap.")
else:
    print("upsetplot not installed; skipping UpSet plot.")

# ------------------- 5. Consensus Heatmap: Mean LR Score for Consensus Interactions -------------------
# Only show interactions supported by >=2 resources
consensus_df = agg_df[agg_df["n_resources"] >= 2]
if not consensus_df.empty:
    # Pivot: source x target, value = mean_lrscore (for top N interactions)
    topN = 30
    top_consensus = consensus_df.head(topN)
    heatmap_data = top_consensus.pivot_table(
        index="source", columns="target", values="mean_lrscore", aggfunc="max"
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap="vlag", annot=True, fmt=".2f", linewidths=0.5)
    plt.title(
        f"Consensus Heatmap: Mean LR Score (Top {topN} Interactions, ≥2 Resources)"
    )
    plt.xlabel("Receiver Cell Type")
    plt.ylabel("Sender Cell Type")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "consensus_heatmap_mean_lrscore.png"))
    plt.close()
    print("Saved consensus heatmap.")
else:
    print("No consensus interactions (≥2 resources) for heatmap.")

# ------------------- 6. Table: Top Consensus Interactions -------------------
top_consensus = agg_df[agg_df["n_resources"] >= 2].head(20)
top_consensus.to_csv(
    os.path.join(PLOT_DIR, "consensus_top_consensus_interactions.csv"), index=False
)

print("All consensus plots and tables for GSE157278 have been generated.")
