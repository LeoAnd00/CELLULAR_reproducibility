# Import packages
import scanpy as sc
import pandas as pd
import numpy as np
from functions import data_preprocessing as dp
from scipy.io import mmread
import argparse

# ScRNA-Seq from [Zheng68k](https://www.nature.com/articles/ncomms14049#Sec34)
### Github: https://github.com/10XGenomics/single-cell-3prime-paper/tree/master 

# cd CELLULAR_reproducibility/code/data_preprocessing/data_for_evaluating_cell_type_annotation/Zheng68k/
# sbatch jobscript_Zheng68k_preprocess.sh

def main(main_path: str='../../../../data/raw/data_for_evaluating_cell_type_annotation/Zheng68k/filtered_matrices_mex/hg19/',
         out_path: str='../../../../data/processed/data_for_evaluating_cell_type_annotation/Zheng68k'):

    ### Read data

    path = f'{main_path}matrix.mtx'

    # Read the .mtx file
    matrix = mmread(path).toarray()

    # Convert the matrix to a NumPy array
    matrix_array = np.array(matrix)

    # Read labels
    labels_path = f'{main_path}68k_pbmc_barcodes_annotation.tsv'
    labels = pd.read_csv(labels_path, sep='\t')[["barcodes","celltype"]]

    # Read barcodes
    barcodes_path = f'{main_path}barcodes.tsv'
    barcodes = pd.read_csv(barcodes_path, sep='\t', header=None)
    barcodes.columns = ["barcodes"]

    merged_barcodes_labels = pd.merge(barcodes, labels, on='barcodes', how='left')

    # Read genes
    genes_path = f'{main_path}genes.tsv'
    genes = pd.read_csv(genes_path, sep='\t', header=None)

    ### Make anndata object
    adata = sc.AnnData(X=matrix_array.T)
    adata.index = genes.iloc[:,1].to_list()
    adata.var_names = genes.iloc[:,1].to_list()
    adata.var["gene_ENSG"] = genes.iloc[:,0].to_list()
    adata.obs["barcodes"] = merged_barcodes_labels["barcodes"].to_list()
    adata.obs["cell_type"] = merged_barcodes_labels["celltype"].to_list()

    patient_ids = []
    for barcode in adata.obs["barcodes"]:
        patient_ids.append(f"Patient_{str(barcode).split('-')[-1]}")
    adata.obs["patientID"] = patient_ids

    # Checking for duplicate genes
    duplicate_genes = adata.var_names[adata.var_names.duplicated()]
    if not duplicate_genes.empty:
        print(f"Duplicate genes found: {duplicate_genes}")
        print(f"Found {len(duplicate_genes)} genes")

        # Find the indexes of duplicate genes
        duplicate_gene_indexes = adata.var_names[adata.var_names.duplicated()]

        # Keep one index for each gene
        indexes_to_keep = []
        count_tracker = {}
        for idx, gene in enumerate(adata.var_names):
            if gene in duplicate_gene_indexes:
                try:
                    if count_tracker[gene] == 1:
                        continue
                except:
                    count_tracker[gene] = 1

            indexes_to_keep.append(idx)

        # Filter adata to remove duplicates, and only keep one of them
        adata = adata[:, indexes_to_keep]

    ### Preprocessing
    adata = dp.QC().QC_metric_calc(adata)

    #Filter genes:
    print('Number of genes before filtering: {:d}'.format(adata.n_vars))

    # Min "expression_limit" cells - filters out 0 count genes
    sc.pp.filter_genes(adata, min_cells=20)
    print(f'Number of genes after filtering so theres min {20} unique cells per gene: {adata.n_vars}')

    ### Normalize
    norm_qc_adata = dp.log1p_normalize(adata)
    del norm_qc_adata.layers['pp_counts']

    # Change data type
    norm_qc_adata.X = norm_qc_adata.X.astype(np.float32)

    ### Download normalized count matrix
    norm_qc_adata.write(f'{out_path}.h5ad')

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Runs the preprocessing of the Zheng68k dataset.')
    parser.add_argument('main_path', type=str, help='Path to folder containing all files related to Zheng68k.')
    parser.add_argument('out_path', type=str, help='Path to save preprocessed Anndata.')
    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(args.main_path, args.out_path)

