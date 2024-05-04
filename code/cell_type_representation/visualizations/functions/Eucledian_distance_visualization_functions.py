
import scanpy as sc
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

class VisualizeEnv():
    """
    A class for visualizing eucledian distance between cell type clusters when accounting for batch effect.

    Methods:
    - MakePredictions: Make predictions using a trained model on a separate dataset.
    - PCA_cell_type_centroid_distances: Calculate the centroid distances between cell types in PCA space.
    - CalculateDistanceMatrix: Calculate the distance matrix based on PCA space.
    - DownloadDistanceMatrix: Save the distance matrix to a file.
    - LoadDistanceMatrix: Load a precomputed distance matrix.
    - VisualizeCellTypeCorrelations: Visualize the distance matrices and their statistics.
    """

    def __init__(self):
        pass

    def ReadscData(self, 
                    path: str, 
                    target_key: str,
                    batch_key: str):
        """
        Read Anndata .h5ad file.

        Parameters:
        - path (str): File path to the dataset in AnnData format.
        - target_key (str): Key for cell type labels.
        - batch_key (str): Key for batch information.

        Returns:
        None
        """
        
        self.label_key = target_key
        self.pred_adata = sc.read(path, cache=True)
        self.pred_adata.obs["batch"] = self.pred_adata.obs[batch_key]


    def PCA_cell_type_centroid_distances(self, n_components: int=100):
        """
        Calculate the average centroid distances between cell types across batch effects in PCA space.

        Parameters:
        - n_components (int): Number of principal components for PCA.

        Returns:
        - average_distance_df (pd.DataFrame): DataFrame of average centroid distances.
        - distance_std_df (pd.DataFrame): DataFrame of standard deviations of centroid distances.
        """

        # Step 1: Perform PCA on AnnData.X
        adata = self.pred_adata.copy()  # Make a copy of the original AnnData object
        pca = PCA(n_components=n_components)
        adata_pca = pca.fit_transform(adata.X)

        # Step 2: Calculate centroids for each cell type cluster of each batch effect
        centroids = {}
        for batch_effect in adata.obs['batch'].unique():
            for cell_type in adata.obs['cell_type'].unique():
                mask = (adata.obs['batch'] == batch_effect) & (adata.obs['cell_type'] == cell_type)
                centroid = np.mean(adata_pca[mask], axis=0)
                centroids[(batch_effect, cell_type)] = centroid

        # Step 3: Calculate the average centroid distance between all batch effects
        average_distance_matrix = np.zeros((len(adata.obs['cell_type'].unique()), len(adata.obs['cell_type'].unique())))
        distance_std_matrix = np.zeros((len(adata.obs['cell_type'].unique()), len(adata.obs['cell_type'].unique())))
        for i, cell_type_i in enumerate(adata.obs['cell_type'].unique()):
            for j, cell_type_j in enumerate(adata.obs['cell_type'].unique()):
                distances = []
                for batch_effect in adata.obs['batch'].unique():
                    centroid_i = torch.tensor(centroids[(batch_effect, cell_type_i)], dtype=torch.float32, requires_grad=False)
                    centroid_j = torch.tensor(centroids[(batch_effect, cell_type_j)], dtype=torch.float32, requires_grad=False)
                    try:
                        #distance = euclidean(centroids[(batch_effect, cell_type_i)], centroids[(batch_effect, cell_type_j)])
                        distance = torch.norm(centroid_j - centroid_i, p=2)
                        if not torch.isnan(distance).any():
                            distances.append(distance)
                    except: # Continue if centroids[(batch_effect, cell_type_i)] doesn't exist
                        continue
                average_distance = np.mean(distances)
                distance_std = np.std(distances)
                distance_std_matrix[i, j] = distance_std
                average_distance_matrix[i, j] = average_distance

        # Convert average_distance_matrix into a DataFrame
        average_distance_df = pd.DataFrame(average_distance_matrix, index=adata.obs['cell_type'].unique(), columns=adata.obs['cell_type'].unique())
        
        # Replace NaN values with 0
        average_distance_df = average_distance_df.fillna(0)

        #average_distance_df = average_distance_df/average_distance_df.max().max()

        # Convert distance_std_matrix into a DataFrame
        distance_std_df = pd.DataFrame(distance_std_matrix, index=adata.obs['cell_type'].unique(), columns=adata.obs['cell_type'].unique())
        
        # Replace NaN values with 0
        distance_std_df = distance_std_df.fillna(0)

        return average_distance_df, distance_std_df

    def CalculateDistanceMatrix(self, model_output_dim: int=100):
        """
        Calculate the distance matrix based on PCA space.

        Parameters:
        - model_output_dim (int): Dimensionality of the model output.

        Returns:
        None
        """

        cell_type_vector = self.pred_adata.obs["cell_type"]

        # Calculate the avergae centroid distance between cell type clusters of PCA transformed data
        self.pca_cell_type_centroids_distances_matrix, self.pca_distance_std_df = self.PCA_cell_type_centroid_distances(n_components=model_output_dim)

        cell_type_centroids_distances_matrix_filter = self.pca_cell_type_centroids_distances_matrix.loc[cell_type_vector.unique().tolist(),cell_type_vector.unique().tolist()]

        self.cell_type_centroids_distances_matrix_filter = torch.tensor(cell_type_centroids_distances_matrix_filter.values, dtype=torch.float32)
        
        distance_std_df = self.pca_distance_std_df.loc[cell_type_vector.unique().tolist(),cell_type_vector.unique().tolist()]
        self.distance_std_df = torch.tensor(distance_std_df.values, dtype=torch.float32)

    def VisCellTypeCorrelations(self, image_path: str=None):
        """
        Visualize the distance matrix and CV matrix.

        Parameters:
        - image_path (str): Directory path to save the visualizations.

        Returns:
        None
        """

        # Create a heatmap to visualize the relative distance matrix of the PCA reference
        plt.figure(figsize=(7.08, 6))
        heatmap = plt.imshow(self.cell_type_centroids_distances_matrix_filter/torch.max(self.cell_type_centroids_distances_matrix_filter), cmap='viridis', interpolation='nearest')

        colorbar = plt.colorbar(heatmap, label='Normalized Euclidean Distance')
        colorbar.ax.tick_params(axis='y', labelsize=7)
        colorbar.set_label('Euclidean Distance', fontsize=7) 
        plt.xticks(range(len(self.pred_adata.obs['cell_type'].unique())), self.pred_adata.obs['cell_type'].unique(), rotation=60, ha='right', fontsize=5)  # Adjust rotation and alignment
        plt.yticks(range(len(self.pred_adata.obs['cell_type'].unique())), self.pred_adata.obs['cell_type'].unique(), fontsize=5)
        plt.title('Normalized euclidean distance between cell type centorids in PCA latent space', fontsize=7)
        plt.tight_layout()  # Adjust layout for better spacing
        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}_PCA.svg', format='svg', dpi=300)
        plt.show()

        CV_df = (self.distance_std_df / self.cell_type_centroids_distances_matrix_filter)
        nan_mask = torch.isnan(CV_df)
        CV_df = torch.where(nan_mask, torch.tensor(0.0), CV_df)

        # Extract upper triangular part of the matrix
        upper_triangular = torch.triu(CV_df)

        # Find non-zero elements and their indices
        non_zero_indices = torch.nonzero(upper_triangular)

        # Extract non-zero elements
        non_zero_elements = CV_df[non_zero_indices[:, 0], non_zero_indices[:, 1]]
        mean_value = torch.mean(non_zero_elements)
        std_value = torch.std(non_zero_elements)
        
        plt.figure(figsize=(7.08, 6))
        heatmap = plt.imshow(CV_df, cmap='viridis', interpolation='nearest')

        colorbar = plt.colorbar(heatmap, label='Coefficient of variation (CV)')
        colorbar.ax.tick_params(axis='y', labelsize=7)
        colorbar.set_label('Euclidean Distance', fontsize=7) 
        plt.xticks(range(len(self.pred_adata.obs['cell_type'].unique())), self.pred_adata.obs['cell_type'].unique(), rotation=60, ha='right', fontsize=5)  # Adjust rotation and alignment
        plt.yticks(range(len(self.pred_adata.obs['cell_type'].unique())), self.pred_adata.obs['cell_type'].unique(), fontsize=5)
        plt.title('CV of normalized euclidean distance between cell type centorids in PCA latent space', fontsize=7)
        plt.tight_layout()  # Adjust layout for better spacing
        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}_PCA_CV.svg', format='svg', dpi=300)
        plt.show()




        # Set up the subplots
        fig, axs = plt.subplots(2, 1, figsize=(7.08, 12))  # 2 rows

        # Plot 1
        heatmap = axs[0].imshow(self.cell_type_centroids_distances_matrix_filter/torch.max(self.cell_type_centroids_distances_matrix_filter), cmap='viridis', interpolation='nearest')
        colorbar = plt.colorbar(heatmap, ax=axs[0], label='Normalized Euclidean Distance')
        colorbar.ax.tick_params(axis='y', labelsize=7)
        colorbar.set_label('Normalized Euclidean Distance', fontsize=7)
        axs[0].set_xticks(range(len(self.pred_adata.obs['cell_type'].unique())))
        axs[0].set_xticklabels(self.pred_adata.obs['cell_type'].unique(), rotation=60, ha='right', fontsize=5)
        axs[0].set_yticks(range(len(self.pred_adata.obs['cell_type'].unique())))
        axs[0].set_yticklabels(self.pred_adata.obs['cell_type'].unique(), fontsize=5)
        axs[0].set_title('Normalized euclidean distance between cell type centorids in PCA latent space', fontsize=7)

        # Plot 2
        CV_df = (self.distance_std_df / self.cell_type_centroids_distances_matrix_filter)
        nan_mask = torch.isnan(CV_df)
        CV_df = torch.where(nan_mask, torch.tensor(0.0), CV_df)
        upper_triangular = torch.triu(CV_df)
        non_zero_indices = torch.nonzero(upper_triangular)
        non_zero_elements = CV_df[non_zero_indices[:, 0], non_zero_indices[:, 1]]
        mean_value = torch.mean(non_zero_elements)
        std_value = torch.std(non_zero_elements)

        heatmap2 = axs[1].imshow(CV_df, cmap='viridis', interpolation='nearest')
        colorbar2 = plt.colorbar(heatmap2, ax=axs[1], label='Coefficient of variation (CV)')
        colorbar2.ax.tick_params(axis='y', labelsize=7)
        colorbar2.set_label('CV of Euclidean Distance', fontsize=7)
        axs[1].set_xticks(range(len(self.pred_adata.obs['cell_type'].unique())))
        axs[1].set_xticklabels(self.pred_adata.obs['cell_type'].unique(), rotation=60, ha='right', fontsize=5)
        axs[1].set_yticks(range(len(self.pred_adata.obs['cell_type'].unique())))
        axs[1].set_yticklabels(self.pred_adata.obs['cell_type'].unique(), fontsize=5)
        axs[1].set_title('CV of normalized euclidean distance between cell type centorids in PCA latent space', fontsize=7)

        # Annotate subplots with letters
        for ax, letter in zip(axs.ravel(), ['a', 'b']):
            ax.text(-0.1, 1.1, letter, transform=ax.transAxes, fontsize=7, fontweight='bold', va='top')

        plt.tight_layout()

        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}_PCA_combined.svg', format='svg', dpi=300)

        plt.show()





        # Create a violin plot of CV scores
        plt.figure(figsize=((7.08/2), (6/2)))
        non_zero_elements_np = non_zero_elements.numpy()
        sns.violinplot(y=non_zero_elements_np)

        # Add labels and title
        plt.xlabel("Density", fontsize=7)
        plt.ylabel("CV score", fontsize=7)
        plt.title("", fontsize=7)
        plt.tight_layout()  # Adjust layout for better spacing
        if image_path:
            plt.savefig(f'{image_path}_PCA_CV_Violin.svg', format='svg', dpi=300)

        # Show the plot
        plt.show()

        print("mean CV value: ", mean_value)
        print("std CV value: ", std_value)
        print("Number fo cell types: ", len(self.pred_adata.obs['cell_type'].unique()))
        
