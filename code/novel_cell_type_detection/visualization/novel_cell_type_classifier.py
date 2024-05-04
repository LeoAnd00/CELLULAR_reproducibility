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
import os
import matplotlib.pyplot as plt
import seaborn as sns
import anndata

import sys
sys.path.append('../') # Add parent directory to path, since it contains the CELLULAR package
import CELLULAR

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class classifier_train():
    """
    A class for doing inference for novel cell type detection and calculating likelihood/confidence scores of predictions.

    Parameters
    ----------
    exclude_cell_types : list
        A list containing cell types to exclude.
    data_path : str 
        The path to the single-cell RNA-seq Anndata file in h5ad format.
    dataset_name : str 
        Name of dataset.
    image_path : str, optional
        The path to save UMAP images.
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
                 exclude_cell_types,
                 data_path: str, 
                 dataset_name: str,
                 image_path: str='',
                 batch_key: str="patientID", 
                 label_key: str="cell_type", 
                 HVG: bool=True, 
                 HVGs: int=2000, 
                 num_folds: int=5,
                 fold: int=1,
                 seed: int=42):
        
        adata = sc.read(data_path, cache=True)

        adata.obs["batch"] = adata.obs[batch_key]

        if dataset_name != "MacParland":
            del adata.layers['log1p_counts']

        self.adata = adata

        self.label_key = label_key

        if not os.path.exists(image_path):
            os.makedirs(image_path)
        if not os.path.exists(f'figures/dendrogram{image_path}'):
            os.makedirs(f'figures/dendrogram{image_path}')
        
        self.image_path = image_path

        self.seed = seed
        self.HVGs = HVGs
        self.fold = fold
        self.dataset_name = dataset_name

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

        # Define cell types to exclude
        self.exclude_cell_types = exclude_cell_types

        # Iterate through the folds
        self.adata = self.adata#.copy()
        self.test_adata = self.adata#.copy()
        fold_counter = 0
        for train_index, test_index in stratified_kfold.split(self.adata.X, self.adata.obs[self.label_key]):
            fold_counter += 1
            if fold_counter == fold:
                self.adata = self.adata[train_index, :]#.copy()
                self.test_adata = self.test_adata[test_index, :]#.copy()

                # Create a boolean mask to select cells that are not in the exclude list
                mask = ~self.adata.obs['cell_type'].isin(self.exclude_cell_types)

                # Add the excluded data to test data so we utilize it better
                adata_novel = self.adata[~mask]
                self.test_adata = anndata.concat([self.test_adata, adata_novel])

                # Apply the mask to AnnData object
                self.adata = self.adata[mask]

                break

        #self.original_adata = self.adata#.copy()
        self.original_test_adata = self.test_adata#.copy()

        if HVG:
            sc.pp.highly_variable_genes(self.adata, n_top_genes=HVGs, flavor="cell_ranger")
            self.test_adata = self.test_adata[:, self.adata.var["highly_variable"]].copy()
            self.test_adata.var["highly_variable"] = self.adata.var["highly_variable"].copy()
            self.adata = self.adata[:, self.adata.var["highly_variable"]].copy()

        del self.adata, self.test_adata

        # Settings for visualizations
        sc.settings.set_figure_params(dpi_save=600,  frameon=False, transparent=True, fontsize=12)
        self.celltype_title = 'Cell Type'
        self.batcheffect_title = 'Batch Effect'

    def threshold_investigation(self, save_path: str="trained_models/"):
        """
        Calculates min likelihood of novel and non-novel cell types of current fold.

        Parameters
        ----------
        save_path : str
            Path at which the model is saved.

        Returns
        -------
        None
        """
        save_path = f"{save_path}Fold_{self.fold}/"

        adata_in_house_test = self.original_test_adata#.copy()
        predictions = CELLULAR.predict(adata=adata_in_house_test, model_path=save_path)
        adata_in_house_test.obsm["latent_space"] = predictions

        predictions, pred_prob = CELLULAR.predict(adata=adata_in_house_test, model_path=save_path, use_classifier=True, return_pred_probs=True)
        adata_in_house_test.obs[f"{self.label_key}_prediction"] = predictions
        adata_in_house_test.obs[f"{self.label_key}_probability"] = pred_prob

        del predictions, pred_prob

        if "Novel" not in adata_in_house_test.obs[self.label_key].cat.categories:
            adata_in_house_test.obs[self.label_key] = adata_in_house_test.obs[self.label_key].cat.add_categories(["Novel"])
        for i, cell_type_ in enumerate(adata_in_house_test.obs[self.label_key]):
            if cell_type_ in self.exclude_cell_types:
                adata_in_house_test.obs[self.label_key].iloc[i] = "Novel"

        ## Calculate the minimum confidence of non-novel cell types and of novel cell types
        # Sort the DataFrame by probability column
        sorted_indices = adata_in_house_test.obs[f"{self.label_key}_probability"].argsort()

        # Iterate through sorted DataFrame
        min_non_novel_confidence = 0
        min_novel_confidence = 0
        count_tracker = 0
        count_tracker2 = 0
        for index in sorted_indices:
            
            if (count_tracker == 0) and (adata_in_house_test.obs[self.label_key].iloc[index] != "Novel"):
                min_non_novel_confidence = adata_in_house_test.obs[f"{self.label_key}_probability"][index]
                count_tracker += 1

            if (count_tracker2 == 0) and (adata_in_house_test.obs[self.label_key].iloc[index] == "Novel"):
                min_novel_confidence = adata_in_house_test.obs[f"{self.label_key}_probability"][index]
                count_tracker2 += 1

            if (count_tracker != 0) and (count_tracker2 != 0):
                break
        
        return min_non_novel_confidence, min_novel_confidence
        


