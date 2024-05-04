# data
All data used in this project are located in this folder, consisting of scRNA-Seq data, pathway information and gene2vec embeddings.

## Structur
- **raw:** Contains all raw scRNA-Seq data.
    - **data_for_evaluating_cell_type_annotation:** Data used for evaluating cell type annotation performance.
        - **Baron:** scRNA-Seq data from [Baron](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133)
        - **MacParland:** scRNA-Seq data from [MacParland](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115469)
        - **Segerstolpe:** scRNA-Seq data from [Segerstolpe](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-5061)
        - **Zheng68k:** scRNA-Seq data from [Zheng68k](https://www.nature.com/articles/ncomms14049#Sec34)
    - **immune_cells:** Data related to immune cells.
        - **bone_marrow_human:** scRNA-Seq data samples taken from bone marrow of human subjects.
        - **pbmcs_human:** scRNA-Seq data samples taken from peripheral blood of human subjects.
    - **pancreas_human:** Data related to pancreas cells.
    - **kidney_cells:** Data related to kidney cells.
- **processed:** Contains all preprocessed scRNA-Seq data.
    - **data_for_evaluating_cell_type_annotation:** Data used for evaluating cell type annotation performance.
        - **Baron:** Preprocessed scRNA-Seq data from [Baron](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133)
        - **MacParland:** Preprocessed scRNA-Seq data from [MacParland](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115469)
        - **Segerstolpe:** Preprocessed scRNA-Seq data from [Segerstolpe](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-5061)
        - **Zheng68k:** Preprocessed scRNA-Seq data from [Zheng68k](https://www.nature.com/articles/ncomms14049#Sec34)
    - **immune_cells:** Data related to immune cells.
        - **bone_marrow_human:** Preprocessed scRNA-Seq data of the bone marrow dataset.
        - **pbmcs_human:** Preprocessed scRNA-Seq data of the PBMC dataset.
        - **merged:** Merged AnnData objects from the bone marrow and PBMC dataset.
    - **pancreas_cells:** Preprocessed data of the pancreas dataset.
    - **kidney_cells:** Preprocessed data of the kidney dataset.
    - **merged:** AnnData object where all bone marrow, PBMC, pancreas and kidney datasets have been merged.