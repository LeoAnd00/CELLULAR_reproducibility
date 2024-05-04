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
    A class for training CELLULAR for novel cell type detection.

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
                 num_folds: int=5,
                 fold: int=1,
                 seed: int=42):
        
        adata = sc.read(data_path, cache=True)

        adata.obs["batch"] = adata.obs[batch_key]

        if dataset_name != "MacParland":
            del adata.layers['log1p_counts']

        self.adata = adata

        self.label_key = label_key
        self.image_path = image_path
        self.seed = seed
        self.fold = fold
        self.dataset_name = dataset_name

        self.metrics = None
        self.metrics_CELLULAR = None

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
        self.adata = self.adata.copy()
        self.test_adata = self.adata.copy()
        fold_counter = 0
        for train_index, test_index in stratified_kfold.split(self.adata.X, self.adata.obs[self.label_key]):
            fold_counter += 1
            if fold_counter == fold:
                self.adata = self.adata[train_index, :].copy()
                self.test_adata = self.test_adata[test_index, :].copy()

                # Create a boolean mask to select cells that are not in the exclude list
                mask = ~self.adata.obs['cell_type'].isin(self.exclude_cell_types)

                # Apply the mask to AnnData object
                self.adata = self.adata[mask]

                break

        self.original_adata = self.adata

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
        """
        save_path = f"{save_path}Fold_{self.fold}/"

        adata_in_house = self.original_adata.copy()

        if train:
            CELLULAR.train(adata=adata_in_house, model_path=save_path, train_classifier=True, target_key=self.label_key, batch_key="batch")


