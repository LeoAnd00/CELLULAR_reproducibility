# Load packages
import numpy as np
import pandas as pd
import scanpy as sc
import torch.nn as nn
import torch
import random
import tensorflow as tf
import warnings
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
import CELLULAR

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class classifier_train():
    """
    A class for benchmarking single-cell RNA-seq data annotation methods.

    Parameters
    ----------
    data_path : str 
        The path to the single-cell RNA-seq Anndata file in h5ad format.
    dataset_name : str 
        Name of dataset.
    batch_key : str, optional
        The batch key to use for batch effect information (default is "patientID").
    label_key : str, optional
        The label key containing the cell type information (default is "cell_type").
    HVG : bool, optional 
        Whether to select highly variable genes (HVGs) (default is True).
    HVGs : int, optional
        The number of highly variable genes to select if HVG is enabled (default is 2000).
    num_folds : int, optional
        Number of folds for cross testing
    fold : int, optional
        Which fold to use.
    seed : int, optional
        Which random seed to use (default is 42).
    """

    def __init__(self, 
                 data_path: str, 
                 dataset_name: str,
                 batch_key: str="patientID", 
                 label_key: str="cell_type", 
                 HVG: bool=True, 
                 HVGs: int=2000, 
                 num_folds: int=5,
                 fold: int=1,
                 seed: int=42):
        
        adata = sc.read(data_path, cache=True)

        adata.obs["batch"] = adata.obs[batch_key]

        # Delete adata.layers['log1p_counts'] to free up more memory if needed. (The MacParland dataset doesn't have the log1p_counts layer)
        if dataset_name != "MacParland":
            del adata.layers['log1p_counts']

        self.adata = adata

        self.label_key = label_key
        self.seed = seed
        self.HVGs = HVGs
        self.fold = fold
        self.dataset_name = dataset_name

        self.metrics = None
        self.metrics_CELLULAR = None
        self.metrics_TOSICA = None

        # Ensure reproducibility
        def rep_seed(seed):
            # Check if a GPU is available
            if torch.cuda.is_available():
                # Set the random seed for PyTorch CUDA (GPU) operations
                torch.cuda.manual_seed(seed)
                # Set the random seed for all CUDA devices (if multiple GPUs are available)
                torch.cuda.manual_seed_all(seed)
            
            # Set the random seed for CPU-based PyTorch operations
            torch.manual_seed(seed)
            
            # Set the random seed for NumPy
            np.random.seed(seed)
            
            # Set the random seed for Python's built-in 'random' module
            random.seed(seed)
            
            # Set the random seed for TensorFlow
            tf.random.set_seed(seed)
            
            # Set CuDNN to deterministic mode for PyTorch (GPU)
            torch.backends.cudnn.deterministic = True
            
            # Disable CuDNN's benchmarking mode for deterministic behavior
            torch.backends.cudnn.benchmark = False

        rep_seed(self.seed)

        # Initialize Stratified K-Fold
        stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

        # Iterate through the folds
        self.adata = self.adata.copy()
        self.test_adata = self.adata.copy()
        fold_counter = 0
        for train_index, test_index in stratified_kfold.split(self.adata.X, self.adata.obs[self.label_key]):
            fold_counter += 1
            if fold_counter == fold:
                self.adata = self.adata[train_index, :].copy()
                self.test_adata = self.test_adata[test_index, :].copy()
                break

        self.original_adata = self.adata.copy()
        self.original_test_adata = self.test_adata.copy()

        if HVG:
            sc.pp.highly_variable_genes(self.adata, n_top_genes=HVGs, flavor="cell_ranger")
            self.test_adata = self.test_adata[:, self.adata.var["highly_variable"]].copy()
            self.test_adata.var["highly_variable"] = self.adata.var["highly_variable"].copy()
            self.adata = self.adata[:, self.adata.var["highly_variable"]].copy()

        # Settings for visualizations
        sc.settings.set_figure_params(dpi_save=600,  frameon=False, transparent=True, fontsize=12)
        self.celltype_title = 'Cell type'
        self.batcheffect_title = 'Batch effect'

    def tosica(self):
        """
        Evaluate and visualization on performance of TOSICA (https://github.com/JackieHanLab/TOSICA/tree/main) on single-cell RNA-seq data.

        Parameters
        ----------

        Returns
        -------
        None

        Notes
        -----
        This method computes various metrics to evaluate performance.

        If umap_plot is True, UMAP plots are generated to visualize the distribution of cell types and batch effects in the latent space.
        The UMAP plots can be saved as SVG files if save_figure is True.
        """
        import TOSICA.TOSICA as TOSICA

        adata_tosica = self.adata.copy()
        TOSICA.train(adata_tosica, gmt_path='human_gobp', label_name=self.label_key,project='hGOBP_TOSICA')

        model_weight_path = './hGOBP_TOSICA/model-9.pth'
        adata_tosica = self.test_adata.copy()
        new_adata = TOSICA.pre(adata_tosica, model_weight_path = model_weight_path,project='hGOBP_TOSICA')

        adata_tosica.obs[f"{self.label_key}_prediction"] = new_adata.obs['Prediction'].copy()

        self.metrics = pd.DataFrame({"pred": adata_tosica.obs[f"{self.label_key}_prediction"].to_list(), 
                                    "true_label": adata_tosica.obs["cell_type"].to_list(), 
                                    "fold": [self.fold]*adata_tosica.obs[f"{self.label_key}_prediction"].shape[0]})

        self.metrics.reset_index(drop=True, inplace=True)

    def CELLULAR_classifier(self, save_path: str="trained_models/", train: bool=True):
        """
        Evaluate and visualization on performance of CELLULAR on single-cell RNA-seq data.

        Parameters
        ----------
        save_path : str
            Path at which the model will be saved.
        train : bool, optional
            Whether to train the model (True) or use a existing model (False) (default: True).

        Returns
        -------
        None

        Notes
        -----
        This method computes various metrics to evaluate performance.

        If umap_plot is True, UMAP plots are generated to visualize the distribution of cell types and batch effects in the latent space.
        The UMAP plots can be saved as SVG files if save_figure is True.
        """
        save_path = f"{save_path}Fold_{self.fold}/"

        adata_in_house = self.original_adata.copy()
        
        if train:
            CELLULAR.train(adata=adata_in_house, model_path=save_path, train_classifier=True, target_key=self.label_key, batch_key="batch")
        
        adata_in_house_test = self.original_test_adata#.copy()
        predictions = CELLULAR.predict(adata=adata_in_house_test, model_path=save_path)
        adata_in_house_test.obsm["latent_space"] = predictions

        predictions = CELLULAR.predict(adata=adata_in_house_test, model_path=save_path, use_classifier=True)
        adata_in_house_test.obs[f"{self.label_key}_prediction"] = predictions

        self.metrics = pd.DataFrame({"pred": adata_in_house_test.obs[f"{self.label_key}_prediction"].to_list(), 
                                    "true_label": adata_in_house_test.obs["cell_type"].to_list(), 
                                    "fold": [self.fold]*adata_in_house_test.obs[f"{self.label_key}_prediction"].shape[0]})

        del adata_in_house_test, predictions

        self.metrics.reset_index(drop=True, inplace=True)

    def save_results_as_csv(self, name: str='benchmarks/results/Benchmark_results'):
        """
        Saves the performance metrics dataframe as a CSV file.

        Parameters
        ----------
        name : str, optional
            The file path and name for the CSV file (default is 'benchmarks/results/Benchmark_results' (file name will then be Benchmark_results.csv)).

        Returns
        -------
        None

        Notes
        -----
        This method exports the performance metrics dataframe to a CSV file.
        """
        self.metrics.to_csv(f'{name}.csv', index=True, header=True)
        self.metrics = None

    def read_csv(self, name: str='benchmarks/results/Benchmark_results'):
        """
        Reads a CSV file and updates the performance metrics dataframe.

        Parameters
        ----------
        name : str, optional
            The file path and name of the CSV file to read (default is 'benchmarks/results/Benchmark_results').

        Returns
        -------
        None

        Notes
        -----
        This method reads a CSV file containing performance metrics and updates the metrics dataframe.
        """
        if self.metrics is not None:
            metrics = pd.read_csv(f'{name}.csv', index_col=0)
            self.metrics = pd.concat([metrics, self.metrics], axis="rows")
        else:
            self.metrics = pd.read_csv(f'{name}.csv', index_col=0)



