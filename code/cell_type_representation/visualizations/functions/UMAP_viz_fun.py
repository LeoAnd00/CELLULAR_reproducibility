import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import math
import numpy as np
import torch.nn as nn
import torch
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import CELLULAR

class UMAP_Kidney_Viz():
    """
    A class for visualizing the first fold of the kidney dataset for PCA and CELLULAR.

    Parameters
    ----------
    data_path : str 
        The path to the single-cell RNA-seq Anndata file in h5ad format.
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
    seed : int, optional
        Which random seed to use (default is 42).
    """

    def __init__(self, 
                 data_path: str, 
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
        self.seed = seed
        self.HVGs = HVGs
        self.fold = fold

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

    def pca(self):
        """
        Performce PCA on single-cell RNA-seq data.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        adata_pca = self.test_adata.copy()

        pca = PCA(n_components=60, svd_solver='arpack', random_state=42)
        adata_pca.obsm["embedding"]  = pca.fit_transform(adata_pca.X)

        return adata_pca
    
    def CELLULAR(self, save_path: str="trained_models/"):
        """
        Use of CELLULAR on single-cell RNA-seq data.

        Parameters
        ----------
        save_path : str
            Path at which the model is saved.

        Returns
        -------
        None
        """
        
        save_path = f"{save_path}Fold_{self.fold}/"

        adata_in_house_test = self.original_test_adata.copy()
        predictions = CELLULAR.predict(adata=adata_in_house_test, model_path=save_path)
        adata_in_house_test.obsm["embedding"] = predictions

        return adata_in_house_test

    def umap_viz(self, save_path: str="trained_models/", image_path: str=None):
        """
        Visualizes single-cell RNA sequencing data using UMAP embedding.

        Parameters:
        - save_path (str): Path at which the model is saved.
        - image_path (str): The file path to save the generated visualization. If None, the plot will not be saved.

        Returns:
        None
        """

        np.random.seed(42)

        adata_pca = self.pca()
        adata_CELLULAR = self.CELLULAR(save_path=save_path)

        # Create subplots
        fig, axs = plt.subplots(4, 1, figsize=(7.08, (7.08*2)))

        ### Visualize PCA
        vis_adata = adata_pca

        umap = UMAP(n_components=2, n_neighbors=15, random_state=42)
        vis_adata.obsm["X_umap"] = umap.fit_transform(vis_adata.obsm["embedding"] )

        random_order = np.random.permutation(vis_adata.n_obs)
        vis_adata = vis_adata[random_order, :]

        # Convert categorical variables to numerical labels
        le_cell_type = LabelEncoder()
        vis_adata.obs["cell_type_encoded"] = le_cell_type.fit_transform(vis_adata.obs["cell_type"])

        le_patientID = LabelEncoder()
        vis_adata.obs["patientID_encoded"] = le_patientID.fit_transform(vis_adata.obs["patientID"])

        # Define color palette
        palette = plt.cm.tab20.colors  # You can choose any other color map

        # Plot UMAP colored by cell_type
        color_dict = {}
        for idx, _ in enumerate(vis_adata.obs.index):
            label = vis_adata.obs["cell_type"][idx]
            color_idx = le_cell_type.transform([label])[0]  # Get index of label in encoded classes
            color = palette[color_idx % len(palette)]  # Cycle through the palette colors
            if label not in color_dict:
                color_dict[label] = color
            axs[0].scatter(vis_adata.obsm["X_umap"][idx, 0], vis_adata.obsm["X_umap"][idx, 1], color=color, s=2)
        axs[0].set_title('Cell Type', fontsize=7)

        num_columns = max(1, math.ceil(len(color_dict) / 23))  # Each column for every 20 legend entries
        # Add legend
        # Create legend entries
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=2.5, markerfacecolor=color_dict[label], label=label) for label in color_dict]

        # Add legend outside of the loop
        axs[0].legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=6, markerscale=2.5, ncol=num_columns, columnspacing=0.5)

        # Remove border around the plot
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['bottom'].set_visible(False)
        axs[0].spines['left'].set_visible(False)
        axs[0].axis('off')

        # Plot UMAP colored by patientID
        color_dict = {}
        for idx, _ in enumerate(vis_adata.obs.index):
            label = vis_adata.obs["patientID"][idx]
            color_idx = le_patientID.transform([label])[0]  # Get index of label in encoded classes
            color = palette[color_idx % len(palette)]  # Cycle through the palette colors
            if label not in color_dict:
                color_dict[label] = color
            axs[1].scatter(vis_adata.obsm["X_umap"][idx, 0], vis_adata.obsm["X_umap"][idx, 1], color=color, s=2)

        axs[1].set_title('Batch Effect', fontsize=7)
        
        num_columns = max(1, math.ceil(len(color_dict) / 23))  # Each column for every 20 legend entries
        # Add legend
        # Create legend entries
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=2.5, markerfacecolor=color_dict[label], label=label) for label in color_dict]

        # Add legend outside of the loop
        axs[1].legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=6, markerscale=2.5, ncol=num_columns, columnspacing=0.5)

        # Remove border around the plot
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['bottom'].set_visible(False)
        axs[1].spines['left'].set_visible(False)
        axs[1].axis('off')

        ### Visualize CELLULAR
        vis_adata = adata_CELLULAR

        umap = UMAP(n_components=2, n_neighbors=15, random_state=42)
        vis_adata.obsm["X_umap"] = umap.fit_transform(vis_adata.obsm["embedding"] )

        #random_order = np.random.permutation(vis_adata.n_obs)
        vis_adata = vis_adata[random_order, :]

        # Convert categorical variables to numerical labels
        le_cell_type = LabelEncoder()
        vis_adata.obs["cell_type_encoded"] = le_cell_type.fit_transform(vis_adata.obs["cell_type"])

        le_patientID = LabelEncoder()
        vis_adata.obs["patientID_encoded"] = le_patientID.fit_transform(vis_adata.obs["patientID"])

        # Define color palette
        palette = plt.cm.tab20.colors  # You can choose any other color map

        # Plot UMAP colored by cell_type
        color_dict = {}
        for idx, _ in enumerate(vis_adata.obs.index):
            label = vis_adata.obs["cell_type"][idx]
            color_idx = le_cell_type.transform([label])[0]  # Get index of label in encoded classes
            color = palette[color_idx % len(palette)]  # Cycle through the palette colors
            if label not in color_dict:
                color_dict[label] = color
            axs[2].scatter(vis_adata.obsm["X_umap"][idx, 0], vis_adata.obsm["X_umap"][idx, 1], color=color, s=2)
        axs[2].set_title('Cell Type', fontsize=7)

        num_columns = max(1, math.ceil(len(color_dict) / 23))  # Each column for every 20 legend entries
        # Add legend
        # Create legend entries
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=2.5, markerfacecolor=color_dict[label], label=label) for label in color_dict]

        # Add legend outside of the loop
        axs[2].legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=6, markerscale=2.5, ncol=num_columns, columnspacing=0.5)

        # Remove border around the plot
        axs[2].spines['top'].set_visible(False)
        axs[2].spines['right'].set_visible(False)
        axs[2].spines['bottom'].set_visible(False)
        axs[2].spines['left'].set_visible(False)
        axs[2].axis('off')

        # Plot UMAP colored by patientID
        color_dict = {}
        for idx, _ in enumerate(vis_adata.obs.index):
            label = vis_adata.obs["patientID"][idx]
            color_idx = le_patientID.transform([label])[0]  # Get index of label in encoded classes
            color = palette[color_idx % len(palette)]  # Cycle through the palette colors
            if label not in color_dict:
                color_dict[label] = color
            axs[3].scatter(vis_adata.obsm["X_umap"][idx, 0], vis_adata.obsm["X_umap"][idx, 1], color=color, s=2)

        axs[3].set_title('Batch Effect', fontsize=7)
        
        num_columns = max(1, math.ceil(len(color_dict) / 23))  # Each column for every 20 legend entries
        # Add legend
        # Create legend entries
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=2.5, markerfacecolor=color_dict[label], label=label) for label in color_dict]

        # Add legend outside of the loop
        axs[3].legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=6, markerscale=2.5, ncol=num_columns, columnspacing=0.5)

        # Remove border around the plot
        axs[3].spines['top'].set_visible(False)
        axs[3].spines['right'].set_visible(False)
        axs[3].spines['bottom'].set_visible(False)
        axs[3].spines['left'].set_visible(False)
        axs[3].axis('off')

        # Annotate subplots with letters
        for ax, letter in zip(axs.ravel(), ['a', 'b', 'c', 'd']):
            ax.text(0, 1.1, letter, transform=ax.transAxes, fontsize=7, fontweight='bold', va='top')

        plt.tight_layout()

        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}.svg', format='svg', dpi=300)
            
        plt.show()
