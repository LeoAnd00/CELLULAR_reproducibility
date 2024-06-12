import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import math

def umap_viz(adata, image_path: str=None):
    """
    Visualizes single-cell RNA sequencing data using UMAP embedding.

    Parameters:
    - adata (Anndata): An Anndata object containing the single-cell RNA sequencing data.
    - image_path (str): The file path to save the generated visualization. If None, the plot will not be saved.

    Returns:
    None
    """

    np.random.seed(42)

    # Assuming adata is already defined
    vis_adata = adata#.copy()

    # Preprocessing
    sc.pp.highly_variable_genes(vis_adata, n_top_genes=2000, flavor="cell_ranger")
    vis_adata = vis_adata[:, vis_adata.var['highly_variable']]
    pca = PCA(n_components=60, svd_solver='arpack', random_state=42)
    vis_adata.obsm["X_pca"] = pca.fit_transform(vis_adata.X)
    umap = UMAP(n_components=2, n_neighbors=15, random_state=42)
    vis_adata.obsm["X_umap"] = umap.fit_transform(vis_adata.obsm["X_pca"])

    random_order = np.random.permutation(vis_adata.n_obs)
    vis_adata = vis_adata[random_order, :]

    # Convert categorical variables to numerical labels
    le_cell_type = LabelEncoder()
    vis_adata.obs["cell_type_encoded"] = le_cell_type.fit_transform(vis_adata.obs["cell_type"])

    le_patientID = LabelEncoder()
    vis_adata.obs["patientID_encoded"] = le_patientID.fit_transform(vis_adata.obs["patientID"])

    # Define color palette
    palette = plt.cm.tab20.colors  # You can choose any other color map

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(7*(150/180), (7*81/180)))

    # Plot UMAP colored by cell_type
    color_dict = {}
    for idx, _ in enumerate(vis_adata.obs.index):
        label = vis_adata.obs["cell_type"][idx]
        color_idx = le_cell_type.transform([label])[0]  # Get index of label in encoded classes
        color = palette[color_idx % len(palette)]  # Cycle through the palette colors
        if label not in color_dict:
            color_dict[label] = color
        axs[0].scatter(vis_adata.obsm["X_umap"][idx, 0], vis_adata.obsm["X_umap"][idx, 1], color=color, s=0.3)
    #axs[0].set_title('Cell Type', fontsize=7)

    num_columns = max(1, math.ceil(len(color_dict) / 23))  # Each column for every 20 legend entries
    # Add legend
    # Create legend entries
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=1.5, markerfacecolor=color_dict[label], label=label) for label in color_dict]

    # Add legend outside of the loop
    #axs[0].legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=5, markerscale=1.5, ncol=num_columns, columnspacing=0.5)

    # Remove border around the plot
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].spines['left'].set_visible(False)
    axs[0].axis('off')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    # Plot UMAP colored by patientID
    color_dict = {}
    for idx, _ in enumerate(vis_adata.obs.index):
        label = vis_adata.obs["patientID"][idx]
        color_idx = le_patientID.transform([label])[0]  # Get index of label in encoded classes
        color = palette[color_idx % len(palette)]  # Cycle through the palette colors
        if label not in color_dict:
            color_dict[label] = color
        axs[1].scatter(vis_adata.obsm["X_umap"][idx, 0], vis_adata.obsm["X_umap"][idx, 1], color=color, s=0.3)

    #axs[1].set_title('Batch Effect', fontsize=7)
    
    num_columns = max(1, math.ceil(len(color_dict) / 23))  # Each column for every 20 legend entries
    # Add legend
    # Create legend entries
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=1.5, markerfacecolor=color_dict[label], label=label) for label in color_dict]

    # Add legend outside of the loop
    #axs[1].legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=5, markerscale=1.5, ncol=num_columns, columnspacing=0.5)

    # Remove border around the plot
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].axis('off')
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    # Annotate subplots with letters
    #for ax, letter in zip(axs.ravel(), ['a', 'b']):
    #    ax.text(0, 1.05, letter, transform=ax.transAxes, fontsize=7, fontweight='bold', va='top')

    #border_thickness = 0.5  # Set your desired border thickness here
    #for ax in axs.ravel():
    #    for spine in ax.spines.values():
    #        spine.set_linewidth(border_thickness)

    plt.tight_layout()

    # Save the plot as an SVG file
    if image_path:
        plt.savefig(f'{image_path}.svg', format='svg', dpi=300)
        
    plt.show()
