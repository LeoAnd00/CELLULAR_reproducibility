**cell_type_annotation:** Contains code for performing the cell type annotation benchmark.
1. Start by running *Make_data_splits.ipynb* to generate the training and test data folds used by the R based methods. This is done to make sure the data is split the same for both python and R based methods.
2. Run all codes in */Baron*, */MacParland*, */Segerstolpe*, and */Zheng68k* for scNym, Seurat, SciBet, and CellID.
3. In */benchmark_CELLULAR_and_TOSICA*, run all commands in *CELLULAR_annotation.py* and *TOSICA_annotation.py* to train CELLULAR and TOSICA.
4. Run *Calc_metrics.ipynb* to calculate accuracy, balanced accuracy and F1 score on the test data of each fold.
5. Run *visualize_metrics.ipynb* to make visualization plots of results. 