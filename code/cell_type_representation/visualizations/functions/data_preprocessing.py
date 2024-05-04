import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


def read_sc_data(count_data: str, gene_data: str, barcode_data: str):
    """
    Read single-cell RNA sequencing data and associated gene and barcode information.

    Parameters:
    - count_data (str): The path to the count data file in a format compatible with Scanpy.
    - gene_data (str): The path to the gene information file in tab-separated format.
    - barcode_data (str): The path to the barcode information file in tab-separated format.

    Returns:
    - data (AnnData): An AnnData object containing the count data, gene information, and barcode information.

    This function loads single-cell RNA sequencing data, gene information, and barcode information, and organizes them into
    an AnnData object. The count data is expected to be in a format supported by Scanpy, and it is transposed to ensure
    genes are represented as rows and cells as columns. Gene information is used to annotate the genes in the data, and
    barcode information is used to annotate the cells in the data.

    Example usage:
    data = read_sc_data('count_matrix.mtx.gz', 'genes.tsv.gz', 'barcodes.tsv.gz')
    """

    #Load data
    data = sc.read(count_data, cache=True).transpose()
    data.X = data.X.toarray()

    # Load genes and barcodes
    genes = pd.read_csv(gene_data, sep='\t', header=None)
    barcodes = pd.read_csv(barcode_data, sep='\t', header=None)

    # set genes
    genes.rename(columns={0:'gene_id', 1:'gene_symbol'}, inplace=True)
    genes.set_index('gene_symbol', inplace=True)
    data.var = genes

    # set barcodes
    barcodes.rename(columns={0:'barcode'}, inplace=True)
    barcodes.set_index('barcode', inplace=True)
    data.obs = barcodes

    return data

def log1p_normalize(data):
    """
    Perform log1p normalization on single-cell RNA sequencing data.

    Parameters:
    - data (AnnData): An AnnData object containing the count data to be normalized.

    Returns:
    - data (AnnData): An AnnData object with log1p normalized count data.

    This function performs log1p normalization on count data from single-cell RNA sequencing. It calculates size factors,
    applies log1p transformation to the counts, and updates the AnnData object with the log1p normalized counts.
    data.X will be log1p normalized, but also a layer is created called log1p_counts that also contains the normalized counts
    and a layer called counts is also created containing the unnormalized data.

    Example usage:
    data = log1p_normalize(data)
    """

    data.layers["pp_counts"] = data.X.copy()

    # Calculate size factor
    L = data.X.sum() / data.shape[0]
    data.obs["size_factors"] = data.X.sum(1) / L

    # Normalize using shifted logarithm (log1p)
    scaled_counts = data.X / data.obs["size_factors"].values[:,None]
    data.layers["log1p_counts"] = np.log1p(scaled_counts)

    data.X = data.layers["log1p_counts"]

    return data

def scale_data(data, scale_max: int=10, return_mean_and_std: bool=False, feature_means: list=None, feature_stdevs: list=None):
    """
    Perform feature-level scaling on input data.

    Parameters:
        data (numpy.ndarray): Input data with features as columns and samples as rows.
        scale_max (float): Maximum value to which each feature will be scaled. (Default is 10)
        return_mean_and_std (bool): Whether to return the mean and std values. (Default is False)
        feature_means (list): If one wants to provide the mean values manually.
        feature_stdevs (list): If one wants to provide the std values manually.

    Returns:
        numpy.ndarray: Scaled data with mean-centered and scaled features. (If return_mean_and_std=False)
        numpy.ndarray: Scaled data with mean-centered and scaled features, list of mean values and list of std values (If return_mean_and_std=True)
    """
    # Calculate the mean and standard deviation for each feature (column)
    if feature_means is None:
        feature_means = np.mean(data, axis=0)
    if feature_stdevs is None:
        feature_stdevs = np.std(data, axis=0)

    # Clip the scaled values to scale_max
    scaled_data = np.clip((data - feature_means) / feature_stdevs, -scale_max, scale_max)

    if return_mean_and_std:
        return scaled_data, feature_means, feature_stdevs
    else:
        return scaled_data

class QC():
    """
    Quality Control (QC) class for single-cell RNA sequencing data.

    This class provides methods for performing quality control on single-cell RNA sequencing data, including
    Median Absolute Deviation (MAD) based outlier detection and filtering based on various QC metrics.
    """

    def __init__(self):
        pass

    def median_absolute_deviation(self, data):
        """
        Calculate the Median Absolute Deviation (MAD) of a dataset.

        Parameters:
        - data (list or numpy.ndarray): The dataset for which MAD is calculated.

        Returns:
        - float: The Median Absolute Deviation (MAD) of the dataset.
        """
        median = np.median(data)
        
        absolute_differences = np.abs(data - median)

        mad = np.median(absolute_differences)
        
        return mad

    def QC_metric_calc(self, data):
        """
        Calculate various quality control metrics for single-cell RNA sequencing data.

        Parameters:
        - data (AnnData): An AnnData object containing the single-cell RNA sequencing data.

        Returns:
        AnnData: An AnnData object with additional QC metrics added as observations.

        This method calculates the following QC metrics and adds them as observations to the input AnnData object:
        - 'n_counts': Sum of counts per cell.
        - 'log_n_counts': Shifted log of 'n_counts'.
        - 'n_genes': Number of unique genes expressed per cell.
        - 'log_n_genes': Shifted log of 'n_genes'.
        - 'pct_counts_in_top_20_genes': Fraction of total counts among the top 20 genes with the highest counts.
        - 'mt_frac': Fraction of mitochondrial counts.

        Example usage:
        data = QC().QC_metric_calc(data)
        """

        # Sum of counts per cell
        data.obs['n_counts'] = data.X.sum(1)
        # Shifted log of n_counts
        data.obs['log_n_counts'] = np.log(data.obs['n_counts']+1)
        # Number of unique genes per cell
        data.obs['n_genes'] = (data.X > 0).sum(1)
        # Shifted lof og n_genes
        data.obs['log_n_genes'] = np.log(data.obs['n_genes']+1)

        # Fraction of total counts among the top 20 genes with highest counts
        top_20_indices = np.argpartition(data.X, -20, axis=1)[:, -20:]
        top_20_values = np.take_along_axis(data.X, top_20_indices, axis=1)
        data.obs['pct_counts_in_top_20_genes'] = (np.sum(top_20_values, axis=1)/data.obs['n_counts'])

        # Fraction of mitochondial counts
        mt_gene_mask = [gene.startswith('MT-') for gene in data.var_names]
        data.obs['mt_frac'] = data.X[:, mt_gene_mask].sum(1)/data.obs['n_counts']

        # Fraction of ribosomal counts
        ribo_gene_mask = [gene.startswith(("RPS", "RPL")) for gene in data.var_names]
        data.obs['ribo_frac'] = data.X[:, ribo_gene_mask].sum(1)/data.obs['n_counts']

        # Fraction of hemoglobin counts
        ribo_gene_mask = [gene.startswith(("HB")) for gene in data.var_names]
        data.obs['hem_frac'] = data.X[:, ribo_gene_mask].sum(1)/data.obs['n_counts']

        return data

    def MAD_based_outlier(self, data, metric: str, threshold: int = 5):
        """
        Detect outliers based on the Median Absolute Deviation (MAD) of a specific metric.

        Parameters:
        - data (AnnData): An AnnData object containing the single-cell RNA sequencing data.
        - metric (str): The name of the observation metric to use for outlier detection.
        - threshold (int): The threshold in MAD units for outlier detection.

        Returns:
        numpy.ndarray: A boolean array indicating outlier cells.

        This method detects outlier cells in the input AnnData object based on the specified metric and threshold.
        Outliers are identified using the MAD-based approach.

        Example usage:
        outlier_cells = QC().MAD_based_outlier(data, "log_n_counts", threshold=5)
        """

        data_metric = data.obs[metric]
        # calculate indexes where outliers are detected
        outlier = (data_metric < np.median(data_metric) - threshold * self.median_absolute_deviation(data_metric)) | (
                    np.median(data_metric) + threshold * self.median_absolute_deviation(data_metric) < data_metric)
        return outlier

    def QC_filter_outliers(self, data, threshold: list = [5,5,5,5], expression_limit: int = 20):
        """
        Filter outlier cells from the single-cell RNA sequencing data based on QC metrics.

        Parameters:
        - data (AnnData): An AnnData object containing the single-cell RNA sequencing data.
        - threshold (list): A list of threshold values for each QC metric in the following order:
            - log_n_counts threshold
            - log_n_genes threshold
            - pct_counts_in_top_20_genes threshold
            - mt_frac threshold
        - expression_limit (int): Threshold of how many cell must have counts of a gene in order for it to be preserved.

        Returns:
        AnnData: An AnnData object with outlier cells removed.

        This method performs QC filtering on the input AnnData object by removing cells that are identified as outliers
        based on the specified threshold values for each QC metric. Additionally, it filters out genes with fewer than
        expression_limit unique cells expressing them.

        Example usage:
        filtered_data = QC().QC_filter_outliers(data, threshold=[5, 5, 5, 5], expression_limit=20)
        """

        # Ignore FutureWarning
        warnings.filterwarnings("ignore", category=FutureWarning)

        data.obs["outlier"] = (self.MAD_based_outlier(data, "log_n_counts", threshold[0])
            | self.MAD_based_outlier(data, "log_n_genes", threshold[1])
            | self.MAD_based_outlier(data, "pct_counts_in_top_20_genes", threshold[2])
            | self.MAD_based_outlier(data, "mt_frac", threshold[3])
        )

        # Print how many detected outliers by each QC metric 
        outlier1 = (self.MAD_based_outlier(data, "log_n_genes", threshold[1]))
        outlier2 = (self.MAD_based_outlier(data, "log_n_counts", threshold[0]))
        outlier3 = (self.MAD_based_outlier(data, "pct_counts_in_top_20_genes", threshold[2]))
        outlier4 = (self.MAD_based_outlier(data, "mt_frac", threshold[3]))
        print(f"Number of cells before QC filtering: {data.n_obs}")
        print(f"Number of cells removed by log_n_genes filtering: {sum(1 for item in outlier1 if item)}")
        print(f"Number of cells removed by log_n_counts filtering: {sum(1 for item in outlier2 if item)}")
        print(f"Number of cells removed by pct_counts_in_top_20_genes filtering: {sum(1 for item in outlier3 if item)}")
        print(f"Number of cells removed by mt_frac filtering: {sum(1 for item in outlier4 if item)}")
        
        # Filter away outliers
        data = data[(~data.obs.outlier)].copy()
        print(f"Number of cells post QC filtering: {data.n_obs}")

        #Filter genes:
        print('Number of genes before filtering: {:d}'.format(data.n_vars))

        # Min "expression_limit" cells - filters out 0 count genes
        sc.pp.filter_genes(data, min_cells=expression_limit)
        print(f'Number of genes after filtering so theres min {expression_limit} unique cells per gene: {data.n_vars}')

        return data
    
class EDA():
    """
    Exploratory Data Analysis (EDA) class for single-cell RNA sequencing data.

    This class provides methods for visualizing and exploring single-cell RNA sequencing data, including violin plots,
    scatter plots for QC metrics, and visualization of normalization effects.
    """

    def __init__(self):
        pass

    def ViolinJitter(self, data, y_rows: list, title: str = "Violin Plots", subtitle: list = ["Unfiltered", "QC Filtered"]):
        """
        Create violin plots with jitter for specified columns in single-cell RNA sequencing data.

        Parameters:
        - data (list of AnnData): A list of AnnData objects, typically representing unfiltered and QC-filtered data.
        - y_rows (list): A list of column names to plot on the y-axis.
        - title (str): The title for the entire plot.
        - subtitle (list): A list of subtitles corresponding to each dataset in 'data'.

        This method generates violin plots with jitter for the specified columns in the single-cell RNA sequencing data.
        Each dataset in 'data' will be plotted side by side with the specified columns on the y-axis.

        Example usage:
        EDA().ViolinJitter(data=[data_unfiltered, data_QC_filtered], y_rows=["n_counts", "n_genes"], title="QC Metrics")
        """

        # Ignore the specific FutureWarning
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Create a figure and axis for the plot
        num_y_columns = len(y_rows)

        # Get random colors
        colors = sns.color_palette("husl",len(y_rows))#[random.choice(sns.color_palette("husl")) for _ in range(num_y_columns)] #sns.color_palette("husl",2)
        
        # Create subplots for each Y column
        fig, axes = plt.subplots(num_y_columns, len(data), figsize=(10,3*num_y_columns))

        for n, k in enumerate(data):
            for i, (y_column, color) in enumerate(zip(y_rows, colors)):
                ax = axes[i,n]
            
                # Create a violin plot
                sns.violinplot(data=k.obs[y_column], ax=ax, color=color)
                
                # Calculate the scaling factor based on the violin plot width
                scale_factor = 0.5 * (max(ax.collections[0].get_paths()[0].vertices[:, 0]) - min(ax.collections[0].get_paths()[0].vertices[:, 0]))
                
                # Adjust jitter points to the same width as the violin plot distribution
                sns.stripplot(data=k.obs[y_column], color='black', jitter=scale_factor, alpha=1.0, size=1, ax=ax)

                # Set subplot title
                ax.set_title(subtitle[n])
        
        
        # Set overall plot title
        fig.suptitle(title, y=1.0)
        
        # Adjust spacing between subplots
        plt.tight_layout()
        
        plt.show()

    def ScatterForQCMetrics(self, data, title:str = "Scatter Plot", subtitle:list = ["Unfiltered", "QC Filtered"]):
        """
        Create scatter plots for QC metrics in single-cell RNA sequencing data.

        Parameters:
        - data (list of AnnData): A list of AnnData objects, typically representing unfiltered and QC-filtered data.
        - title (str): The title for the entire plot.
        - subtitle (list): A list of subtitles corresponding to each dataset in 'data'.

        This method generates scatter plots for specified QC metrics in the single-cell RNA sequencing data.
        Each dataset in 'data' will be plotted side by side with the specified QC metrics.

        Example usage:
        EDA().ScatterForQCMetrics(data=[data_unfiltered, data_QC_filtered], title="QC Metrics Scatter Plot")
        """

        # Ignore the specific FutureWarning
        warnings.filterwarnings("ignore", category=FutureWarning)

        x_column=['n_counts','n_counts','n_counts','n_counts']
        y_column=['n_genes','n_genes','n_genes','n_genes']
        color_column=['mt_frac','pct_counts_in_top_20_genes','ribo_frac','hem_frac']
        
        # Create a scatter plot with continuous colors using Matplotlib's pyplot
        fig, axes = plt.subplots(len(x_column), len(data), figsize=(16,6*len(x_column)))

        for n, k in enumerate(data):
            for i in range(len(x_column)):
                ax = axes[i, n]
                scatter = ax.scatter(k.obs[x_column[i]], k.obs[y_column[i]], c=k.obs[color_column[i]], cmap='coolwarm')

                # Add a colorbar to display the color scale
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(color_column[i])  # Set the colorbar label

                ax.set_xlabel(x_column[i])
                ax.set_ylabel(y_column[i])

                # Set subplot title
                ax.set_title(subtitle[n])
        
        # Set overall plot title
        fig.suptitle(title, y=1.0)
        
        # Adjust spacing between subplots
        plt.tight_layout()
        
        # Show the plot
        plt.show()

    def VisualizeNormalization(self,data):
        """
        Visualize the effects of normalization on single-cell RNA sequencing data.

        Parameters:
        - data (AnnData): An AnnData object containing the single-cell RNA sequencing data.

        This method generates histograms to visualize the effects of normalization on the single-cell RNA sequencing data.
        It compares the distribution of raw counts and log1p normalized counts.

        Example usage:
        EDA().VisualizeNormalization(data)
        """
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        p1 = sns.histplot(data.obs["n_counts"], bins=100, kde=False, ax=axes[0])
        axes[0].set_title("Raw counts")
        axes[0].set_xlabel("Sum of counts")

        p2 = sns.histplot(data.layers["log1p_counts"].sum(1), bins=100, kde=False, ax=axes[1])
        axes[1].set_xlabel("Sum of Normalized counts")
        axes[1].set_title("log1p normalised counts")

        plt.show()




        
