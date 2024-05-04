# Load packages
import numpy as np
import pandas as pd
import scanpy as sc
import scib
import torch.nn as nn
import torch
import random
import tensorflow as tf
import warnings
from IPython.display import display
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import CELLULAR

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class benchmark():
    """
    A class for benchmarking single-cell RNA-seq data integration methods.

    Parameters
    ----------
    data_path : str 
        The path to the single-cell RNA-seq Anndata file in h5ad format.
    image_path : str, optional
        The path to save UMAP images.
    batch_key : str, optional
        The batch key to use for batch effect information (default is "patientID").
    label_key : str, optional
        The label key containing the cell type information (default is "cell_type").
    HVG : bool, optional 
        Whether to select highly variable genes (HVGs) (default is True).
    HVGs : int, optional
        The number of highly variable genes to select if HVG is enabled (default is 4000).
    num_folds : int, optional
        Number of folds for cross testing
    fold : int, optional
        Which fold to use.
    pct_for_training : float, optional
        The percentage of data used for training.
    seed : int, optional
        Which random seed to use (default is 42).
    """

    def __init__(self, 
                 data_path: str, 
                 image_path: str='',
                 batch_key: str="patientID", 
                 label_key: str="cell_type", 
                 HVG: bool=True, 
                 HVGs: int=2000, 
                 num_folds: int=5,
                 fold: int=1,
                 pct_for_training: float=0.8,
                 seed: int=42):
        
        adata = sc.read(data_path, cache=True)

        adata.obs["batch"] = adata.obs[batch_key]

        self.adata = adata
        self.label_key = label_key
        self.image_path = image_path
        self.seed = seed
        self.HVGs = HVGs
        self.fold = fold

        # Initialize variables
        self.metrics = None
        self.metrics_pca = None
        self.metrics_unscaled = None
        self.metrics_scvi = None
        self.metrics_scanvi = None
        self.metrics_scgen = None
        self.metrics_tosica_attention = None
        self.CELLULAR = None

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

    def unintegrated(self, umap_plot: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of unintegrated (Not going through any model) version of single-cell RNA-seq data.

        Parameters
        ----------
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
        save_figure : bool, optional
            If True, save UMAP plots as SVG files (default is False).

        Returns
        -------
        None

        Notes
        -----
        This method computes various metrics to evaluate the quality of an unintegrated version of single-cell RNA-seq data.

        If umap_plot is True, UMAP plots are generated to visualize the distribution of cell types and batch effects in the unintegrated data.
        The UMAP plots can be saved as SVG files if save_figure is True.
        """

        adata_unscaled = self.test_adata.copy()

        adata_unscaled.obsm["Unscaled"] = adata_unscaled.X
        sc.pp.neighbors(adata_unscaled, use_rep="Unscaled")

        self.metrics_unscaled = scib.metrics.metrics(
            self.original_test_adata,
            adata_unscaled,
            "batch", 
            self.label_key,
            embed="Unscaled",
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

        del adata_unscaled


    def pca(self, n_comps: int=50, umap_plot: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of PCA on single-cell RNA-seq data.

        Parameters
        ----------
        n_comps : int, optional
            Number of components to retrieve from PCA.
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
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

        adata_pca = self.test_adata.copy()

        sc.tl.pca(adata_pca, n_comps=n_comps, use_highly_variable=True)
        adata_pca.obsm["PCA"] = adata_pca.obsm["X_pca"]
        sc.pp.neighbors(adata_pca, use_rep="PCA")

        self.metrics_pca = scib.metrics.metrics(
            self.original_test_adata,
            adata_pca,
            "batch", 
            self.label_key,
            embed="PCA",
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

        del adata_pca

    def scvi(self, umap_plot: bool=True, save_figure: bool=False):
        """
        scVI version 1.0.4: https://github.com/scverse/scvi-tools
        
        Parameters
        ----------
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
        save_figure : bool, optional
            If True, save UMAP plots as SVG files (default is False).
        """
        # import package
        import scvi

        adata_scvi_train = self.adata.copy()

        scvi.model.SCVI.setup_anndata(adata_scvi_train, layer="pp_counts", batch_key="batch")
        vae = scvi.model.SCVI(adata_scvi_train, gene_likelihood="nb", n_layers=2, n_latent=30)
        vae.train()
        del adata_scvi_train

        adata_scvi = self.test_adata.copy()
        adata_scvi.obsm["scVI"] = vae.get_latent_representation(adata_scvi)

        sc.pp.neighbors(adata_scvi, use_rep="scVI")

        self.metrics_scvi = scib.metrics.metrics(
            self.original_test_adata,
            adata_scvi,
            "batch", 
            self.label_key,
            embed="scVI",
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

        random_order = np.random.permutation(adata_scvi.n_obs)
        adata_scvi = adata_scvi[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_scvi)
            sc.pl.umap(adata_scvi, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_scvi, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_scvi)
            sc.pl.umap(adata_scvi, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save="scVI_cell_type.svg")
            sc.pl.umap(adata_scvi, color="batch", ncols=1, title=self.batcheffect_title, show=False, save="scVI_batch_effect.svg")

        del adata_scvi

        return vae

    def scanvi(self, umap_plot: bool=True, vae=None, save_figure: bool=False):
        """
        scANVI version 1.0.4: https://github.com/scverse/scvi-tools
        
        Parameters
        ----------
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
        save_figure : bool, optional
            If True, save UMAP plots as SVG files (default is False).
        """
        # import package
        import scvi

        adata_scANVI_train = self.adata.copy()

        if vae is None:
            scvi.model.SCVI.setup_anndata(adata_scANVI_train, layer="pp_counts", batch_key="batch")
            vae = scvi.model.SCVI(adata_scANVI_train, gene_likelihood="nb", n_layers=2, n_latent=30)
            vae.train()

        lvae = scvi.model.SCANVI.from_scvi_model(
            vae,
            adata=adata_scANVI_train,
            labels_key=self.label_key,
            unlabeled_category="UnknownUnknown",
        )
        lvae.train(max_epochs=20, n_samples_per_label=100)
        del adata_scANVI_train

        adata_scANVI = self.test_adata.copy()
        adata_scANVI.obsm["scANVI"] = lvae.get_latent_representation(adata_scANVI)

        del lvae, vae
        sc.pp.neighbors(adata_scANVI, use_rep="scANVI")

        self.metrics_scanvi = scib.metrics.metrics(
            self.original_test_adata,
            adata_scANVI,
            "batch", 
            self.label_key,
            embed="scANVI",
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

        random_order = np.random.permutation(adata_scANVI.n_obs)
        adata_scANVI = adata_scANVI[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_scANVI)
            sc.pl.umap(adata_scANVI, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_scANVI, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_scANVI)
            sc.pl.umap(adata_scANVI, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save="scANVI_cell_type.svg")
            sc.pl.umap(adata_scANVI, color="batch", ncols=1, title=self.batcheffect_title, show=False, save="scANVI_batch_effect.svg")

        del adata_scANVI

    def scgen(self, umap_plot: bool=True, save_figure: bool=False):
        """
        scGen version 2.1.1: https://github.com/theislab/scgen 
        
        Parameters
        ----------
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
        save_figure : bool, optional
            If True, save UMAP plots as SVG files (default is False).
        """
        from scgen import SCGEN

        adata_scgen_train = self.adata.copy()

        SCGEN.setup_anndata(adata_scgen_train, batch_key="batch", labels_key=self.label_key)
        model = SCGEN(adata_scgen_train)
        model.train(
            max_epochs=100,
            batch_size=128,
            early_stopping=True,
            early_stopping_patience=10,
        )
        del adata_scgen_train

        adata_scgen = self.test_adata.copy()
        corrected_adata = model.batch_removal(adata_scgen)#model.batch_removal()

        adata_scgen.obsm["scGen"] = corrected_adata.obsm["corrected_latent"]

        del corrected_adata
        sc.pp.neighbors(adata_scgen, use_rep="scGen")

        self.metrics_scgen = scib.metrics.metrics(
            self.original_test_adata,
            adata_scgen,
            "batch", 
            self.label_key,
            embed="scGen",
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

        random_order = np.random.permutation(adata_scgen.n_obs)
        adata_scgen = adata_scgen[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_scgen)
            sc.pl.umap(adata_scgen, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_scgen, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_scgen)
            sc.pl.umap(adata_scgen, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save="scGen_cell_type.svg")
            sc.pl.umap(adata_scgen, color="batch", ncols=1, title=self.batcheffect_title, show=False, save="scGen_batch_effect.svg")

        del adata_scgen

    def tosica(self, umap_plot: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of TOSICA (https://github.com/JackieHanLab/TOSICA/tree/main) on single-cell RNA-seq data.

        Parameters
        ----------
        umap_plot : bool, optional
            If True, generate UMAP plots for cell type and batch effect visualization (default is True).
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
        import TOSICA.TOSICA as TOSICA

        adata_tosica = self.adata.copy()
        TOSICA.train(adata_tosica, gmt_path='human_gobp', label_name=self.label_key, project='hGOBP_TOSICA')

        model_weight_path = './hGOBP_TOSICA/model-9.pth'

        # Attention embedding
        adata_tosica = self.test_adata.copy()
        new_adata = TOSICA.pre(adata_tosica, model_weight_path = model_weight_path, project='hGOBP_TOSICA', laten=False)

        adata_tosica.obsm["TOSICA"] = new_adata.X.copy()

        del new_adata
        sc.pp.neighbors(adata_tosica, use_rep="TOSICA")

        self.metrics_tosica_attention = scib.metrics.metrics(
            self.original_test_adata,
            adata_tosica,
            "batch", 
            self.label_key,
            embed="TOSICA",
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

        random_order = np.random.permutation(adata_tosica.n_obs)
        adata_tosica = adata_tosica[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_tosica)
            sc.pl.umap(adata_tosica, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_tosica, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_tosica)
            sc.pl.umap(adata_tosica, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}TOSICA_cell_type.svg")
            sc.pl.umap(adata_tosica, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}TOSICA_batch_effect.svg")

        del adata_tosica


    def CELLULAR_benchmark(self, save_path: str="trained_models/", umap_plot: bool=True, train: bool=True, save_figure: bool=False):
        """
        Evaluate and visualization on performance of CELLULAR on single-cell RNA-seq data.

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
        adata_in_house_test.obsm["In_house"] = predictions

        del predictions
        sc.pp.neighbors(adata_in_house_test, use_rep="In_house")

        self.CELLULAR = scib.metrics.metrics(
            self.original_test_adata,
            adata_in_house_test,
            "batch", 
            self.label_key,
            embed="In_house",
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

        random_order = np.random.permutation(adata_in_house_test.n_obs)
        adata_in_house_test = adata_in_house_test[random_order, :]

        if umap_plot:
            sc.tl.umap(adata_in_house_test)
            sc.pl.umap(adata_in_house_test, color=self.label_key, ncols=1, title=self.celltype_title)
            sc.pl.umap(adata_in_house_test, color="batch", ncols=1, title=self.batcheffect_title)
        if save_figure:
            sc.tl.umap(adata_in_house_test)
            sc.pl.umap(adata_in_house_test, color=self.label_key, ncols=1, title=self.celltype_title, show=False, save=f"{self.image_path}CELLULAR_cell_type.svg")
            sc.pl.umap(adata_in_house_test, color="batch", ncols=1, title=self.batcheffect_title, show=False, save=f"{self.image_path}CELLULAR_batch_effect.svg")

        del adata_in_house_test

    def make_benchamrk_results_dataframe(self, min_max_normalize: bool=False):
        """
        Generates a dataframe named 'metrics' containing the performance metrics of different methods.

        Parameters
        ----------
        min_max_normalize : bool, optional
            If True, performs min-max normalization on the metrics dataframe (default is False).

        Returns
        -------
        None

        Notes
        -----
        This method consolidates performance metrics from various methods into a single dataframe.
        If min_max_normalize is True, the metrics dataframe is normalized between 0 and 1.
        """

        calculated_metrics = []
        calculated_metrics_names = []
        if self.metrics_pca is not None:
            calculated_metrics.append(self.metrics_pca)
            calculated_metrics_names.append("PCA")
        if self.metrics_unscaled is not None:
            calculated_metrics.append(self.metrics_unscaled)
            calculated_metrics_names.append("Unintegrated")
        if self.metrics_scvi is not None:
            calculated_metrics.append(self.metrics_scvi)
            calculated_metrics_names.append("scVI")
        if self.metrics_scanvi is not None:
            calculated_metrics.append(self.metrics_scanvi)
            calculated_metrics_names.append("scANVI")
        if self.metrics_scgen is not None:
            calculated_metrics.append(self.metrics_scgen)
            calculated_metrics_names.append("scGen")
        if self.metrics_tosica_attention is not None:
            calculated_metrics.append(self.metrics_tosica_attention)
            calculated_metrics_names.append("TOSICA")
        if self.CELLULAR is not None:
            calculated_metrics.append(self.CELLULAR)
            calculated_metrics_names.append("CELLULAR")

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

        if min_max_normalize:
            self.metrics = (self.metrics - self.metrics.min()) / (self.metrics.max() - self.metrics.min())
        
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

    def visualize_results(self, bg_color: str="Blues"):
        """
        Visualizes the performance metrics dataframe using a colored heatmap.

        Parameters
        ----------
        bg_color : str, optional
            The colormap for the heatmap (default is "Blues").

        Returns
        -------
        None

        Notes
        -----
        This method creates a styled heatmap of the performance metrics dataframe for visual inspection.
        """
        styled_metrics = self.metrics.style.background_gradient(cmap=bg_color)
        display(styled_metrics)

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



