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
import scib
import CELLULAR
import CELLULAR_cl
import CELLULAR_cent

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class classifier_train():
    """
    A class for benchmarking single-cell RNA-seq data embedding space generation of CELLULAR using different loss functions.

    Parameters
    ----------
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

        self.adata = adata

        self.label_key = label_key
        self.image_path = image_path
        self.seed = seed
        self.HVGs = HVGs
        self.fold = fold
        self.dataset_name = dataset_name

        self.metrics = None
        self.metrics_CELLULAR = None
        self.metrics_CELLULAR_centroid_loss = None
        self.metrics_CELLULAR_CL_loss = None

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

    def CELLULAR_full_loss_classifier(self, save_path: str="trained_models/", umap_plot: bool=True, train: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of CELLULAR on single-cell RNA-seq data when using the full loss.

        Parameters
        ----------
        save_path : str
            Path at which the model will be saved.
        umap_plot : bool, optional
            Whether to plot resulting latent space using UMAP (default: True).
        train : bool, optional
            Whether to train the model (True) or use a existing model (False) (default: True).
        save_figure : bool, optional
            If True, save UMAP plots as SVG files (default is False).

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
            CELLULAR.train(adata=adata_in_house, model_path=save_path, target_key=self.label_key, batch_key="batch")
        
        adata_in_house_test = self.original_test_adata.copy()
        predictions = CELLULAR.predict(adata=adata_in_house_test, model_path=save_path)
        adata_in_house_test.obsm["latent_space"] = predictions

        del predictions

        sc.pp.neighbors(adata_in_house_test, use_rep="latent_space")

        self.metrics_CELLULAR = scib.metrics.metrics(
            self.original_test_adata,
            adata_in_house_test,
            "batch", 
            self.label_key,
            embed="latent_space",
            isolated_labels_asw_=True,
            silhouette_=True,
            hvg_score_=True,
            graph_conn_=True,
            pcr_=True,
            isolated_labels_f1_=True,
            trajectory_=False,
            nmi_=True,
            ari_=True,
            cell_cycle_=True,
            kBET_=False,
            ilisi_=False,
            clisi_=False,
            organism="human",
        )

        del adata_in_house_test

    def CELLULAR_CL_loss_classifier(self, save_path: str="trained_models/", umap_plot: bool=True, train: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of CELLULAR on single-cell RNA-seq data when using the CL loss part.

        Parameters
        ----------
        save_path : str
            Path at which the model will be saved.
        umap_plot : bool, optional
            Whether to plot resulting latent space using UMAP (default: True).
        train : bool, optional
            Whether to train the model (True) or use a existing model (False) (default: True).
        save_figure : bool, optional
            If True, save UMAP plots as SVG files (default is False).

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
            CELLULAR_cl.train(adata=adata_in_house, model_path=save_path, target_key=self.label_key, batch_key="batch")
        
        adata_in_house_test = self.original_test_adata.copy()
        predictions = CELLULAR_cl.predict(adata=adata_in_house_test, model_path=save_path)
        adata_in_house_test.obsm["latent_space"] = predictions

        del predictions

        sc.pp.neighbors(adata_in_house_test, use_rep="latent_space")

        self.metrics_CELLULAR_CL_loss = scib.metrics.metrics(
            self.original_test_adata,
            adata_in_house_test,
            "batch", 
            self.label_key,
            embed="latent_space",
            isolated_labels_asw_=True,
            silhouette_=True,
            hvg_score_=True,
            graph_conn_=True,
            pcr_=True,
            isolated_labels_f1_=True,
            trajectory_=False,
            nmi_=True,
            ari_=True,
            cell_cycle_=True,
            kBET_=False,
            ilisi_=False,
            clisi_=False,
            organism="human",
        )

        del adata_in_house_test

    def CELLULAR_centroid_loss_classifier(self, save_path: str="trained_models/", umap_plot: bool=True, train: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of CELLULAR on single-cell RNA-seq data when using the centroid loss part.

        Parameters
        ----------
        save_path : str
            Path at which the model will be saved.
        umap_plot : bool, optional
            Whether to plot resulting latent space using UMAP (default: True).
        train : bool, optional
            Whether to train the model (True) or use a existing model (False) (default: True).
        save_figure : bool, optional
            If True, save UMAP plots as SVG files (default is False).

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
            CELLULAR_cent.train(adata=adata_in_house, model_path=save_path, target_key=self.label_key, batch_key="batch")
        
        adata_in_house_test = self.original_test_adata.copy()
        predictions = CELLULAR_cent.predict(adata=adata_in_house_test, model_path=save_path)
        adata_in_house_test.obsm["latent_space"] = predictions

        del predictions

        sc.pp.neighbors(adata_in_house_test, use_rep="latent_space")

        self.metrics_CELLULAR_centroid_loss = scib.metrics.metrics(
            self.original_test_adata,
            adata_in_house_test,
            "batch", 
            self.label_key,
            embed="latent_space",
            isolated_labels_asw_=True,
            silhouette_=True,
            hvg_score_=True,
            graph_conn_=True,
            pcr_=True,
            isolated_labels_f1_=True,
            trajectory_=False,
            nmi_=True,
            ari_=True,
            cell_cycle_=True,
            kBET_=False,
            ilisi_=False,
            clisi_=False,
            organism="human",
        )

        del adata_in_house_test

    def make_benchamrk_results_dataframe(self):
        """
        Generates a dataframe named 'metrics' containing the performance metrics of different methods.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This method consolidates performance metrics from various methods into a single dataframe.
        """

        calculated_metrics = []
        calculated_metrics_names = []
        if self.metrics_CELLULAR is not None:
            calculated_metrics.append(self.metrics_CELLULAR)
            calculated_metrics_names.append("CELLULAR CL + Centroid Loss")
        if self.metrics_CELLULAR_centroid_loss is not None:
            calculated_metrics.append(self.metrics_CELLULAR_centroid_loss)
            calculated_metrics_names.append("CELLULAR Centroid Loss")
        if self.metrics_CELLULAR_CL_loss is not None:
            calculated_metrics.append(self.metrics_CELLULAR_CL_loss)
            calculated_metrics_names.append("CELLULAR CL Loss")

        if len(calculated_metrics_names) != 0:
            metrics = pd.concat(calculated_metrics, axis="columns")

            metrics = metrics.set_axis(calculated_metrics_names, axis="columns")

            metrics = metrics.loc[
                [
                    "ASW_label",
                    "ASW_label/batch",
                    "PCR_batch",
                    "isolated_label_silhouette",
                    "graph_conn",
                    "hvg_overlap",
                    "NMI_cluster/label",
                    "ARI_cluster/label",
                    "cell_cycle_conservation",
                    "isolated_label_F1"
                ],
                :,
            ]

            metrics = metrics.T
            metrics = metrics.drop(columns=["hvg_overlap"])

            if self.metrics is None:
                self.metrics = metrics#.sort_values(by='Overall', ascending=False)
            else:
                self.metrics = pd.concat([self.metrics, metrics], axis="rows").drop_duplicates()
        
        self.metrics["Overall Batch"] = self.metrics[["ASW_label/batch", "PCR_batch", "graph_conn"]].mean(axis=1)
        self.metrics["Overall Bio"] = self.metrics[["ASW_label", 
                                        "isolated_label_silhouette", 
                                        "NMI_cluster/label", 
                                        "ARI_cluster/label",
                                        "isolated_label_F1",
                                        "cell_cycle_conservation"]].mean(axis=1)
        self.metrics["Overall"] = 0.4 * self.metrics["Overall Batch"] + 0.6 * self.metrics["Overall Bio"] # priorities biology slightly more
        self.metrics = self.metrics.sort_values(by='Overall', ascending=False)

        self.metrics = self.metrics.sort_values(by='Overall', ascending=False)

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
            self.metrics = pd.concat([self.metrics, metrics], axis="rows")
        else:
            self.metrics = pd.read_csv(f'{name}.csv', index_col=0)



