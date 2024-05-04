# code
Contains code used in this project, including preprocessing, visualization, machine learning models and more.

## Structur
- **data_preprocessing:** Contains preprocessing code and visualization of scRNA-seq data. All processed data is stored in the *data/processed/* folder.
    - **automated_preprocess:** Code one can run in cmd that automatically preprocess the bone marrow, PBMC, pancreas, kidney and merged datasets.
    - **immune_cells:** Code for preprocessing data related to immune cells.
        - **bone_marrow_human:** Preprocessing code for scRNA-seq data of the bone marrow dataset.
        - **pbmcs_human:** Preprocessing code for scRNA-seq data of the PBMC dataset.
        - **merge:** Code to merge all AnnData objects from *bone_marrow_human* and *pbmcs_human*.
        - **automated_preprocess:** Code one can run in cmd that automatically preprocess all immune cells data.
    - **pancreas_cells:** Code for preprocessing data of the pancreas dataset.
    - **kidney_cells:** Code for preprocessing data of the kidney dataset.
    - **pathway_information:** Contains preprocessing code for human pathway information (gene sets). Run *gene_sets_filtering.ipynb* to process the gene sets.
    - **data_for_evaluating_cell_type_annotation:** Code for processing the Baron, MacParland, Segerstolpe, and Zheng68k datasets.
    - **visualizations:** Code for making UMAP visualizations of data.

- **cell_type_representation:** Contains training code and visualizations from the scRNA-Seq embedding space benchmark. 
    - In *benchmark_generalizability.py* the commands needed to run the benchmark on all datasets can be found as a comment.
    - **visualizations:** Visualizations for report. Both for the generalizable embedding space and for invetigating the validity of the cell type centroid loss. But also to visualize the embedding space using UMAP for the first test fold of the kidney dataset.

- **cell_type_annotation:** Contains code for performing the cell type annotation benchmark.
    1. Start by running *Make_data_splits.ipynb* to generate the training and test data folds used by the R based methods. This is done to make sure the data is split the same for both python and R based methods.
    2. Run all codes in */Baron*, */MacParland*, */Segerstolpe*, and */Zheng68k* for scNym, Seurat, SciBet, CellID, TOSICA, and CELLULAR.
    3. In */benchmark_CELLULAR_and_TOSICA*, run all commands in *CELLULAR_annotation.py* and *TOSICA_annotation.py*.
    4. Run *Calc_metrics.ipynb* to calculate accuracy, balanced accuracy and F1 score on the test data of each fold.
    5. Run *visualize_metrics.ipynb* to make visualization plots of results. 

- **cell_type_annotation_loss_comp:** Contains code for visualizing the comparison between different loss functions.
    1. Code for training the model on the different loss functions can be found in */train_code*.
    2. Run *Calc_metrics.ipynb* to concatenate all cell type annotation results into */results_and_visualizations*
    3. Run *visualize_metrics.ipynb* to make visualizations.

- **novel_cell_type_detection:** Contains code for visualizing novel cell type detection function of the model.
    1. In *benchmark_annotation.py* the commands needed to run the novel cell type detection can be found as a comment. In this code we train CELLULAR for each fold when leaving out every type of cell in the MacParland dataset. 
    2. In *benchmark_annotation_Segerstolpe.py* the commands needed to run the novel cell type detection can be found as a comment. In this code we train CELLULAR for each fold when leaving out every type of cell in the Segerstolpe dataset.
    3. In *benchmark_annotation_Baron.py* the commands needed to run the novel cell type detection can be found as a comment. In this code we train CELLULAR for each fold when leaving out every type of cell in the Baron dataset. 
    4. In *benchmark_annotation_Zheng68k.py* the commands needed to run the novel cell type detection can be found as a comment. In this code we train CELLULAR for each fold when leaving out every type of cell in the Zheng68k dataset. 
    5. In *visualization/* there's code to create data for visualization of min confidence for novel and non-novel cells of each fold for each dataset and each cell type dropout event. Go to *novel_cell_type_confidence.py* and execute the command given as a comment in that file. This will create a file called *likelihood.json* where a minimum likelihoods are saved. Run *novel_cell_type_confidence.ipynb* to make figures.
