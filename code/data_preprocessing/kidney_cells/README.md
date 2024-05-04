# kidney_cells
Contains all code for preprocessing the scRNA-seq data of the kidney dataset.

## How to use notebooks in the subfolder *kidney_1*, *kidney_2* and *kidney_3*
- **Step 1:** Start by runing *X_preprocess.ipynb* to perform QC and normalization.
- **Step 2:** Data is then annotated with cell types using *X_ScType_CellType_Labeling.Rmd* using the [ScType](https://github.com/IanevskiAleksandr/sc-type/tree/master) library.
- **Step 3:** Run *X_apply_labels.ipynb* to add the cell type annotations to the AnnData object.

## How to use notebooks in the subfolder *merge*
Ones all notebooks in *kidney_1*, *kidney_2* and *kidney_3*, run *Muto_merge.ipynb* in merge to merge all kidney cells related data.