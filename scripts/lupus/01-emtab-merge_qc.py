"""
01-emtab-merge_qc.py

Purpose:
    Full pipeline for E-MTAB-13596 lupus single-cell RNA-seq data:
    - Parse SDRF metadata
    - Load, annotate, and merge all samples
    - Perform QC, normalization, HVG selection, PCA, batch correction (Harmony), clustering, UMAP
    - Save processed AnnData, QC metrics, and plots to organized directories

Inputs:
    - SDRF metadata: /Users/aumchampaneri/complement-receptor-blockade/raw-data/lupus-data/E-MTAB-13596/E-MTAB-13596.sdrf.txt
    - Matrix/barcode/feature files for each sample in /Users/aumchampaneri/complement-receptor-blockade/raw-data/lupus-data/E-MTAB-13596/

Outputs:
    - Merged AnnData: /Users/aumchampaneri/complement-receptor-blockade/data/lupus-amp/emtab_merged.h5ad
    - QC metrics: /Users/aumchampaneri/complement-receptor-blockade/outputs/lupus-amp/qc_metrics_emtab.csv
    - Plots: /Users/aumchampaneri/complement-receptor-blockade/plots/lupus-amp/

Author: [Your Name]
Date: [YYYY-MM-DD]
"""

import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

# Directories
data_dir = "/Users/aumchampaneri/complement-receptor-blockade/raw-data/lupus-data/E-MTAB-13596"
output_h5ad = "/Users/aumchampaneri/complement-receptor-blockade/data/lupus-emtab/emtab_merged.h5ad"
qc_metrics_path = "/Users/aumchampaneri/complement-receptor-blockade/outputs/lupus-emtab/qc_metrics_emtab.csv"
plots_dir = "/Users/aumchampaneri/complement-receptor-blockade/plots/lupus-emtab"
os.makedirs(plots_dir, exist_ok=True)

# SDRF metadata path
sdrf_path = os.path.join(data_dir, "E-MTAB-13596.sdrf.txt")

print("Parsing SDRF metadata...")
sdrf = pd.read_csv(sdrf_path, sep="\t")

# Extract relevant columns
samples = []
for idx, row in sdrf.iterrows():
    sample_id = row['Source Name']
    disease = row['Characteristics[disease]']
    cell_type = row['Characteristics[cell type]']
    disease_stage = row.get('Characteristics[disease staging]', '')
    # Find corresponding matrix/barcode/features files in SDRF columns
    matrix_file = None
    barcode_file = None
    feature_file = None
    for col in sdrf.columns:
        if isinstance(row[col], str) and row[col].endswith("_matrix.mtx"):
            matrix_file = row[col]
        if isinstance(row[col], str) and row[col].endswith("_barcodes.tsv"):
            barcode_file = row[col]
        if isinstance(row[col], str) and row[col].endswith("_features.tsv"):
            feature_file = row[col]
    # Fallback: infer from sample_id
    if matrix_file is None:
        matrix_file = f"{sample_id}_matrix.mtx"
    if barcode_file is None:
        barcode_file = f"{sample_id}_barcodes.tsv"
    if feature_file is None:
        feature_file = f"{sample_id}_features.tsv"
    samples.append({
        "sample_id": sample_id,
        "disease": disease,
        "cell_type": cell_type,
        "disease_stage": disease_stage,
        "matrix_file": matrix_file,
        "barcode_file": barcode_file,
        "feature_file": feature_file
    })

print(f"Found {len(samples)} samples in SDRF.")

# Load all samples
adatas = []
for sample in samples:
    print(f"Loading sample {sample['sample_id']} ...")
    matrix_path = os.path.join(data_dir, sample['matrix_file'])
    barcode_path = os.path.join(data_dir, sample['barcode_file'])
    feature_path = os.path.join(data_dir, sample['feature_file'])
    print(f"Loading matrix from: {matrix_path}")
    # Manual loading of mtx, barcodes, and features
    import scipy.io
    # Load matrix
    matrix = scipy.io.mmread(matrix_path)
    # Load barcodes
    barcodes = pd.read_csv(barcode_path, header=None)[0].tolist()
    # Load features
    features = pd.read_csv(feature_path, sep="\t", header=None)
    gene_names = features[1].tolist()  # column 1 is usually gene symbol
    # Check shape and transpose if needed
    if matrix.shape[0] == len(barcodes):
        matrix = matrix.tocsc()
    elif matrix.shape[1] == len(barcodes):
        matrix = matrix.T.tocsc()
    else:
        raise ValueError(f"Matrix dimensions {matrix.shape} do not match barcodes length {len(barcodes)} for sample {sample['sample_id']}.")
    # Diagnostic: check for unique barcodes
    prefixed_barcodes = [f"{sample['sample_id']}_{bc}" for bc in barcodes]
    unique_barcodes = set(prefixed_barcodes)
    print(f"Sample {sample['sample_id']} - Total barcodes: {len(barcodes)}, Unique barcodes: {len(unique_barcodes)}")
    # Create AnnData
    ad = sc.AnnData(matrix)
    ad.obs_names = prefixed_barcodes
    # Ensure gene names are unique
    from anndata.utils import make_index_unique
    ad.var_names = make_index_unique(pd.Index(gene_names))
    # Annotate sample metadata
    ad.obs['sample_id'] = sample['sample_id']
    ad.obs['disease'] = sample['disease']
    ad.obs['cell_type_sort'] = sample['cell_type']
    ad.obs['disease_stage'] = sample['disease_stage']
    ad.obs['batch'] = sample['sample_id']  # For batch correction
    adatas.append(ad)

print("Merging all samples into a single AnnData object...")
adata = adatas[0].concatenate(
    adatas[1:],
    batch_key="batch",
    batch_categories=[s['sample_id'] for s in samples]
)

print(f"Merged AnnData: {adata.n_obs} cells, {adata.n_vars} genes.")
print("AnnData .obs columns:", adata.obs.columns.tolist())
print("Preview of .obs (first 5 rows):")
print(adata.obs.head())

# --- QC ---
print("Running quality control...")
sc.pp.filter_cells(adata, min_genes=200)
print(f"After min_genes filter: {adata.n_obs} cells")
sc.pp.filter_genes(adata, min_cells=3)
print(f"After min_cells filter: {adata.n_vars} genes")

# Mitochondrial filtering
adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
print(f"Cells with <10% mito: {(adata.obs['pct_counts_mt'] < 10).sum()} / {adata.n_obs}")
adata = adata[adata.obs['pct_counts_mt'] < 10, :]
print(f"After mito filter: {adata.n_obs} cells")

# Save QC metrics
qc_metrics = adata.obs[['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'sample_id', 'disease', 'disease_stage']]
qc_metrics.to_csv(qc_metrics_path)
print(f"QC metrics saved to {qc_metrics_path}")

# Remove cells with zero total counts before normalization
zero_count_cells = (adata.X.sum(axis=1) == 0)
print(f"Cells with zero total counts before normalization: {zero_count_cells.sum()}")
adata = adata[~zero_count_cells, :]

# Remove cells with any NaN values before normalization
import scipy.sparse
if scipy.sparse.issparse(adata.X):
    nan_cells = np.isnan(adata.X.toarray()).any(axis=1)
else:
    nan_cells = np.isnan(adata.X).any(axis=1)
print(f"Cells with NaN values before normalization: {nan_cells.sum()}")
adata = adata[~nan_cells, :]

# Diagnostics: Check for negative values before normalization/log1p
print("Min value in adata.X before normalization:", adata.X.min())
print("Max value in adata.X before normalization:", adata.X.max())
if adata.X.min() < 0:
    print("Data appears to be already log-transformed or scaled (contains negative values). Skipping normalization/log1p.")
else:
    print("Normalizing and log-transforming...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

# Remove cells with any NaN values after normalization/log1p
if scipy.sparse.issparse(adata.X):
    nan_cells_post = np.isnan(adata.X.toarray()).any(axis=1)
else:
    nan_cells_post = np.isnan(adata.X).any(axis=1)
print(f"Cells with NaN values after normalization/log1p: {nan_cells_post.sum()}")
adata = adata[~nan_cells_post, :]

# --- HVG selection ---
print("Selecting highly variable genes...")
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# --- Scaling and PCA ---
print("Scaling and running PCA...")
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')

# --- Harmony batch correction ---
print("Running Harmony batch correction...")
import scanpy.external as sce
# Confirm batch harmonization is using the correct column
print("Harmony will use 'batch' column for integration. Unique batch values:", adata.obs['batch'].unique())
sce.pp.harmony_integrate(adata, 'batch')

if 'X_pca_harmony' in adata.obsm:
    print("Harmony-corrected PCA found. Using for downstream analysis.")
    print("Shape of X_pca_harmony:", adata.obsm['X_pca_harmony'].shape)
    print("Any NaNs in X_pca_harmony?", np.isnan(adata.obsm['X_pca_harmony']).any())
    print("Min/Max in X_pca_harmony:", adata.obsm['X_pca_harmony'].min(), adata.obsm['X_pca_harmony'].max())
    adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
else:
    print("WARNING: Harmony-corrected PCA not found. Using default PCA.")

# --- UMAP and clustering ---
print("Computing neighbors, UMAP, and clustering after Harmony...")
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
sc.tl.umap(adata)
sc.tl.leiden(adata)

# --- Save diagnostic plots ---
print("Saving UMAP and PCA plots...")
sc.pl.pca(adata, color='batch', save="_emtab_pca.png", show=False)
sc.pl.umap(adata, color=['batch', 'disease', 'disease_stage', 'leiden'], save="_emtab_umap.png", show=False)

# Move plots to correct directory if needed
for plot_name in ["figures/pca_emtab_pca.png", "figures/umap_emtab_umap.png", "pca_emtab_pca.png", "umap_emtab_umap.png"]:
    src = plot_name
    dst = os.path.join(plots_dir, os.path.basename(plot_name))
    if os.path.exists(src):
        os.rename(src, dst)

# --- Save processed AnnData object ---
print(f"Saving processed AnnData object to {output_h5ad} ...")
adata.write(output_h5ad)

print("Processing complete. Output written to", output_h5ad)
print(f"QC metrics written to {qc_metrics_path}")
print(f"Plots written to {plots_dir}")
