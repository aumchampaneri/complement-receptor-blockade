#!/usr/bin/env python3
"""
Aggregate LIANA Complement Results Across Resources & Generate Consensus/Overlap Plots (GSE137029)
==================================================================================================

This script aggregates LIANA receptor-ligand analysis results for complement genes
across multiple resources (cellphonedb, cellchatdb, icellnet, connectomedb2020)
and generates consensus/overlap plots and summary tables for the GSE137029 dataset.

INPUT:
    - Filtered LIANA results per resource:
        /Users/aumchampaneri/complement-receptor-blockade/outputs/lupus-gse137029/rla/liana_complement_pairs_{resource}.csv

OUTPUT:
    - Aggregated consensus CSV
    - Barplot: number of resources supporting each interaction
    - UpSet plot: overlap of interactions across resources
    - Consensus heatmap: mean LR score for consensus interactions
    - Table: top consensus interactions

ENVIRONMENT:
    - Python >=3.8
    - pandas, matplotlib, seaborn, upsetplot

AUTHOR: Expert Engineer (adapted for GSE137029)
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
    "/Users/aumchampaneri/complement-receptor-blockade/outputs/lupus-gse137029/rla/"
)
PLOT_DIR = (
    "/Users/aumchampaneri/complement-receptor-blockade/plots/lupus-gse137029/rla/"
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
interaction_cols = ["ligand_complex", "receptor_complex", "source", "target"]
all_df["interaction_id"] = (
    all_df["ligand_complex"].astype(str)
    + "|"
    + all_df["receptor_complex"].astype(str)
    + "|"
    + all_df["source"].astype(str)
    + "|"
    + all_df["target"].astype(str)
)
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
        "mean_lrscore": group["lrscore"].mean() if "lrscore" in group else np.nan,
        "max_lrscore": group["lrscore"].max() if "lrscore" in group else np.nan,
        "min_lrscore": group["lrscore"].min() if "lrscore" in group else np.nan,
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
consensus_df = agg_df[agg_df["n_resources"] >= 2]
if not consensus_df.empty:
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

# ------------------- 7. Extra Plots: Complement Gene Focus -------------------

# 7.1 Barplot: Top Complement Genes as Ligands and as Receptors
complement_genes = set()
for df in dfs:
    complement_genes.update(df["ligand_complex"].unique())
    complement_genes.update(df["receptor_complex"].unique())
complement_genes = {
    g
    for g in complement_genes
    if isinstance(g, str)
    and (
        g.startswith("C")
        or g.startswith("CF")
        or g.startswith("CR")
        or g.startswith("MASP")
        or g
        in [
            "CLU",
            "FGA",
            "FGB",
            "FGG",
            "HRG",
            "ITGAM",
            "ITGAX",
            "KLKB1",
            "KNG1",
            "MBL2",
            "PLG",
            "PROS1",
            "SERPING1",
            "THBD",
            "VTN",
            "VSIG4",
        ]
    )
}

# Ligand barplot
ligand_counts = agg_df[agg_df["ligand_complex"].isin(complement_genes)][
    "ligand_complex"
].value_counts()
plt.figure(figsize=(10, 5))
ligand_counts.head(20).plot(kind="bar", color="dodgerblue")
plt.title("Top Complement Genes as Ligands (Top 20)")
plt.xlabel("Ligand Gene")
plt.ylabel("Number of Interactions")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "complement_ligand_barplot.png"))
plt.close()
print("Saved barplot of top complement ligands.")

# Receptor barplot
receptor_counts = agg_df[agg_df["receptor_complex"].isin(complement_genes)][
    "receptor_complex"
].value_counts()
plt.figure(figsize=(10, 5))
receptor_counts.head(20).plot(kind="bar", color="darkorange")
plt.title("Top Complement Genes as Receptors (Top 20)")
plt.xlabel("Receptor Gene")
plt.ylabel("Number of Interactions")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "complement_receptor_barplot.png"))
plt.close()
print("Saved barplot of top complement receptors.")

# 7.2 Heatmap: Complement Gene × Cell Type (as ligand or receptor)
celltypes = sorted(set(agg_df["source"]).union(set(agg_df["target"])))
gene_celltype_matrix = pd.DataFrame(
    0, index=sorted(complement_genes), columns=celltypes
)
for _, row in agg_df.iterrows():
    if (
        row["ligand_complex"] in gene_celltype_matrix.index
        and row["source"] in gene_celltype_matrix.columns
    ):
        gene_celltype_matrix.loc[row["ligand_complex"], row["source"]] += 1
    if (
        row["receptor_complex"] in gene_celltype_matrix.index
        and row["target"] in gene_celltype_matrix.columns
    ):
        gene_celltype_matrix.loc[row["receptor_complex"], row["target"]] += 1
plt.figure(figsize=(14, max(6, 0.25 * len(gene_celltype_matrix))))
sns.heatmap(gene_celltype_matrix, cmap="Blues", linewidths=0.5, annot=False)
plt.title("Complement Gene × Cell Type Interaction Count")
plt.xlabel("Cell Type")
plt.ylabel("Complement Gene")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "complement_gene_celltype_heatmap.png"))
plt.close()
print("Saved heatmap of complement gene × cell type interactions.")

# 7.3 Distribution Plot: LR Scores for Complement Interactions
comp_lr_scores = agg_df[
    (agg_df["ligand_complex"].isin(complement_genes))
    | (agg_df["receptor_complex"].isin(complement_genes))
]["mean_lrscore"].dropna()
if not comp_lr_scores.empty:
    plt.figure(figsize=(7, 4))
    sns.histplot(comp_lr_scores, bins=30, kde=True, color="purple")
    plt.title("Distribution of Mean LR Scores for Complement Interactions")
    plt.xlabel("Mean LR Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "complement_lrscore_distribution.png"))
    plt.close()
    print("Saved LR score distribution plot for complement interactions.")
else:
    print("No complement LR scores found for distribution plot.")

# 7.4 Network Plot: Top Complement Interactions (by mean LR score)
try:
    import networkx as nx

    topN_network = 30
    top_net = (
        agg_df[
            (agg_df["ligand_complex"].isin(complement_genes))
            | (agg_df["receptor_complex"].isin(complement_genes))
        ]
        .sort_values("mean_lrscore", ascending=False)
        .head(topN_network)
    )
    G = nx.DiGraph()
    for _, row in top_net.iterrows():
        G.add_edge(
            f"{row['ligand_complex']} ({row['source']})",
            f"{row['receptor_complex']} ({row['target']})",
            weight=row["mean_lrscore"],
        )
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue")
    nx.draw_networkx_edges(
        G,
        pos,
        width=[2 + 3 * (w / np.nanmax(edge_weights)) for w in edge_weights],
        alpha=0.7,
        edge_color="gray",
    )
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(f"Top {topN_network} Complement Interactions (Network)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "complement_network_top_interactions.png"))
    plt.close()
    print("Saved network plot of top complement interactions.")
except ImportError:
    print("networkx not installed; skipping network plot.")
except Exception as e:
    print(f"Error generating network plot: {e}")

print("All consensus plots and tables for GSE137029 have been generated.")
