# immune_cells
Contains all code for preprocessing the scRNA-seq data of the bone marrow and PBMC dataset.

## How to use notebooks in the subfolder *bone_marrow_human* and *pbmcs_human*
- **Step 1:** Start by runing *X_preprocess.ipynb* to perform QC and normalization.
- **Step 2:** Data is then annotated with cell types using *X_ScType_CellType_Labeling.Rmd* using the [ScType](https://github.com/IanevskiAleksandr/sc-type/tree/master) library.
- **Step 3:** Run *X_apply_labels.ipynb* to add the cell type annotations to the AnnData object.

## How to use notebooks in the subfolder *merge*
Ones all notebooks in *bone_marrow_human* and *pbmcs_human*, run all *X_merge.ipynb* and finally run *immune_cells_merge_all.ipynb*
to merge all immune cells related data.

## Automated processing
Code that will preprocess and annotate all data without visualizing each step. <br>
Run:
```
cd code\data_preprocessing\immune_cells\automated_preprocess
```
and then:
```
python auto_pp.py --resolution 0.8
```