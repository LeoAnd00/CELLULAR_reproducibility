# data
All data used in this project are located in this folder, consisting of scRNA-Seq data, pathway information and gene2vec embeddings.

## Structur
- **processed:** Contains all preprocessed scRNA-Seq data.
    - **data_for_evaluating_cell_type_annotation:** Data used for evaluating cell type annotation performance.
        - **Baron:** Preprocessed scRNA-Seq data from [Baron](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133)
        - **MacParland:** Preprocessed scRNA-Seq data from [MacParland](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115469)
        - **Segerstolpe:** Preprocessed scRNA-Seq data from [Segerstolpe](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-5061)
        - **Zheng68k:** Preprocessed scRNA-Seq data from [Zheng68k](https://www.nature.com/articles/ncomms14049#Sec34)
    - **immune_cells:** Data related to immune cells.
        - **merged:** Merged AnnData objects from the bone marrow and PBMC dataset.
		-**Oetjen_merged.h5ad:** Is the bone marrow dataset.
		-**PBMC_merged_all.h5ad:** Is the PBMC dataset.
    - **pancreas_cells:** Preprocessed data of the pancreas dataset.
	-**pancreas_1_adata.h5ad:** Is the pancreas dataset.
    - **kidney_cells:** Preprocessed data of the kidney dataset.
	-**Muto_merged.h5ad:** Is the kidney dataset.
    - **merged:** AnnData object where all bone marrow, PBMC, pancreas and kidney datasets have been merged.
	-**merged_all.h5ad:** Is the merged dataset.