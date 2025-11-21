#!/usr/bin/env python3
"""
00-fix-gse137029-varnames.py

Utility script to fix AnnData .var_names for GSE137029 (lupus PBMC) so that gene symbols or feature names
are used as the index, enabling downstream compatibility with LIANA and other tools.

- Loads the processed AnnData file.
- Inspects .var columns for gene symbol or feature name.
- Sets .var_names to the appropriate column (gene symbol or feature_name).
- Makes .var_names unique.
- Saves a new AnnData file with corrected .var_names.

INPUT:
    - /Users/aumchampaneri/complement-receptor-blockade/data/lupus-gse137029/gse137029_qc_harmony.h5ad

OUTPUT:
    - /Users/aumchampaneri/complement-receptor-blockade/data/lupus-gse137029/gse137029_qc_harmony_genesymbols.h5ad

AUTHOR: Expert Engineer
DATE: 2024-06
"""

import scanpy as sc

INFILE = "/Users/aumchampaneri/complement-receptor-blockade/data/lupus-gse137029/gse137029_qc_harmony.h5ad"
OUTFILE = "/Users/aumchampaneri/complement-receptor-blockade/data/lupus-gse137029/gse137029_qc_harmony_genesymbols.h5ad"

print(f"Loading AnnData: {INFILE}")
adata = sc.read_h5ad(INFILE)

print("adata.var columns:", list(adata.var.columns))
print("First 5 rows of adata.var:")
print(adata.var.head())

# Try to find a column with gene symbols or feature names
candidate_columns = ["gene_symbol", "feature_name", "symbol", "Gene", "gene", "name"]
selected_col = None
for col in candidate_columns:
    if col in adata.var.columns:
        selected_col = col
        break

if selected_col is None:
    raise ValueError(
        f"Could not find a gene symbol or feature name column in adata.var! "
        f"Available columns: {list(adata.var.columns)}"
    )

print(f"Setting adata.var index and var_names to column: '{selected_col}'")
# Convert to string to avoid CategoricalIndex issues
adata.var[selected_col] = adata.var[selected_col].astype(str)
adata.var.set_index(selected_col, inplace=True)
adata.var.index = adata.var.index.astype(str)  # Ensure plain string Index
adata.var_names_make_unique()
# Drop the column if it still exists to avoid AnnData index/column conflict
if selected_col in adata.var.columns:
    adata.var.drop(columns=[selected_col], inplace=True)

print("First 10 new var_names:", list(adata.var_names[:10]))

print(f"Saving AnnData with fixed var_names to: {OUTFILE}")
adata.write(OUTFILE)
print("Done.")
