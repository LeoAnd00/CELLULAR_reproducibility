import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import subprocess
import os


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

def scale_data(data, scale_max=10):
    """
    Perform feature-level scaling on input data.

    Parameters:
        data (numpy.ndarray): Input data with features as columns and samples as rows.
        scale_max (float): Maximum value to which each feature will be scaled.

    Returns:
        numpy.ndarray: Scaled data with mean-centered and scaled features.
    """
    # Calculate the mean and standard deviation for each feature (column)
    feature_means = np.mean(data, axis=0)
    feature_stdevs = np.std(data, axis=0)

    # Clip the scaled values to scale_max
    scaled_data = np.clip((data - feature_means) / feature_stdevs, -scale_max, scale_max)

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
        hem_gene_mask = [(gene == "HBB" or gene == "HBA1" or gene == "HBA2") for gene in data.var_names]
        data.obs['hem_frac'] = data.X[:, hem_gene_mask].sum(1)/data.obs['n_counts']

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

    def QC_filter_outliers(self, data, threshold: list = [5,5,5,5], expression_limit: int = 20, print_: bool = True):
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
        
        # Filter away outliers
        data = data[(~data.obs.outlier)].copy()

        # Min "expression_limit" cells - filters out 0 count genes
        sc.pp.filter_genes(data, min_cells=expression_limit)

        if print_:
            print(f"Number of cells before QC filtering: {data.n_obs}")
            print(f"Number of cells removed by log_n_genes filtering: {sum(1 for item in outlier1 if item)}")
            print(f"Number of cells removed by log_n_counts filtering: {sum(1 for item in outlier2 if item)}")
            print(f"Number of cells removed by pct_counts_in_top_20_genes filtering: {sum(1 for item in outlier3 if item)}")
            print(f"Number of cells removed by mt_frac filtering: {sum(1 for item in outlier4 if item)}")
            print(f"Number of cells post QC filtering: {data.n_obs}")
            #Filter genes:
            print('Number of genes before filtering: {:d}'.format(data.n_vars))
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

### Automate processing and labeling ###

class automatic_preprocessing():
    """
    This class provides functions for automated preprocessing of single-cell RNA-seq data.
    """

    def __init__(self):
        pass

    def QC_filter(self, file_base: str, name: str,count_file_end: str = None, gene_data_end: str = None, barcode_data_end: str = None, thresholds:list = [5,5,5,5], read_option: str = "custom"):
        """
        Performs quality control, normalization, and filtering on the input data.

        Args:
        - file_base (str): Base name for input data files.
        - name (str): Name to be used for output files.
        - count_file_end (str, optional): Suffix for count data file. Default is None.
        - gene_data_end (str, optional): Suffix for gene data file. Default is None.
        - barcode_data_end (str, optional): Suffix for barcode data file. Default is None.
        - thresholds (list, optional): Thresholds for QC filtering. Default is [5, 5, 5, 5].
        - read_option (str, optional): Data read option ("custom", "10X_h5", or "10X_mtx"). Default is "custom".
        """

        # Read data
        if read_option == "custom":
            count_data = file_base+count_file_end
            gene_data = file_base+gene_data_end
            barcode_data = file_base+barcode_data_end
            adata = read_sc_data(count_data, gene_data, barcode_data)
            adata.var_names_make_unique()
        elif read_option == "10X_h5":
            adata = sc.read_10x_h5(filename=file_base)
            adata.var_names_make_unique()
            adata.X = adata.X.toarray()
        elif read_option == "10X_mtx":
            adata = sc.read_10x_mtx(path=file_base, var_names='gene_symbols', make_unique=True, cache=True, gex_only=True )
            adata.var_names_make_unique()
            adata.X = adata.X.toarray()

        # Add QC metrics to adata
        adata = QC().QC_metric_calc(adata)

        # Remove outliers
        qc_adata = QC().QC_filter_outliers(adata, thresholds, print_=False)

        # Normalize
        norm_qc_adata = log1p_normalize(qc_adata)

        # Download normalized count matrix
        sc.pp.highly_variable_genes(norm_qc_adata, n_top_genes=4000, flavor="cell_ranger")
        HVG_data = norm_qc_adata[:, norm_qc_adata.var["highly_variable"]]
        normalized_counts = pd.DataFrame(HVG_data.layers["log1p_counts"])
        normalized_counts.index = HVG_data.obs.index.to_list()
        normalized_counts.columns = HVG_data.var.index.to_list()
        normalized_counts.to_csv(f"{name}.csv")
        norm_qc_adata.write(f"{name}_adata.h5ad")

        del adata, qc_adata, norm_qc_adata

    def cell_labeling(self, RScript_path: str, name: str, args: list, output_path: str):
        """
        Executes an R script to annotate cell types based on the preprocessed data.

        Args:
        - RScript_path (str): Path to the R script for cell labeling.
        - name (str): Name prefix of file.
        - args (list): List of arguments to pass to the R script.
        - output_path (str): Path to the output directory.
        """

        cmd = ['Rscript', RScript_path] + args
        subprocess.call(cmd, shell=True)

        norm_qc_adata = sc.read_h5ad(f"{name}_adata.h5ad")
        labels = pd.read_csv(f"{name}_labels.txt")

        norm_qc_adata.obs["cell_type"] = list(labels.iloc[:,0])

        norm_qc_adata.write(output_path)

        del norm_qc_adata

        # Remove temporary files
        os.remove(f"{name}.csv")
        os.remove(f"{name}_adata.h5ad")
        os.remove(f"{name}_labels.txt")

class BM_1(automatic_preprocessing):

    def __init__(self, res):
        super().__init__()
        self.res = res

    def QC_filter_(self):
        # Path of data
        file_base = '../../../../data/raw/immune_cells/bone_marrow_human/GSM3396161_A/'
        count_file_end = 'GSM3396161_matrix_A.mtx.gz'
        gene_data_end = 'GSM3396161_genes_A.tsv.gz'
        barcode_data_end = 'GSM3396161_barcodes_A.tsv.gz'
        thresholds = [5,5,5,5]
        name = "BM_1"

        self.QC_filter(file_base=file_base, name=name,count_file_end=count_file_end, gene_data_end=gene_data_end, barcode_data_end=barcode_data_end, thresholds=thresholds)
    
    def cell_labeling_(self):
        RScript_path = 'functions\cell_labeling.R'
        name = "BM_1"
        args = ["BM_1", self.res] # File name and resolution
        output_path = "../../../../data/processed/immune_cells/bone_marrow_human/BM_1_adata.h5ad"

        self.cell_labeling(RScript_path, name, args, output_path)

class BM_2(automatic_preprocessing):

    def __init__(self, res):
        super().__init__()
        self.res = res

    def QC_filter_(self):
        # Path of data
        file_base = '../../../../data/raw/immune_cells/bone_marrow_human/GSM3396176_P/'
        count_file_end = 'GSM3396176_matrix_P.mtx.gz'
        gene_data_end = 'GSM3396176_genes_P.tsv.gz'
        barcode_data_end = 'GSM3396176_barcodes_P.tsv.gz'
        thresholds = [5,5,3,5]
        name = "BM_2"

        self.QC_filter(file_base=file_base, name=name, count_file_end=count_file_end, gene_data_end=gene_data_end, barcode_data_end=barcode_data_end, thresholds=thresholds)
    
    def cell_labeling_(self):
        RScript_path = 'functions\cell_labeling.R'
        name = "BM_2"
        args = ["BM_2", self.res] # File name and resolution
        output_path = "../../../../data/processed/immune_cells/bone_marrow_human/BM_2_adata.h5ad"

        self.cell_labeling(RScript_path, name, args, output_path)

class BM_3(automatic_preprocessing):

    def __init__(self, res):
        super().__init__()
        self.res = res

    def QC_filter_(self):
        # Path of data
        file_base = '../../../../data/raw/immune_cells/bone_marrow_human/GSM3396184_U/'
        count_file_end = 'GSM3396184_matrix_U.mtx.gz'
        gene_data_end = 'GSM3396184_genes_U.tsv.gz'
        barcode_data_end = 'GSM3396184_barcodes_U.tsv.gz'
        thresholds = [5,5,5,5]
        name = "BM_3"

        self.QC_filter(file_base=file_base, name=name, count_file_end=count_file_end, gene_data_end=gene_data_end, barcode_data_end=barcode_data_end, thresholds=thresholds)
    
    def cell_labeling_(self):
        RScript_path = 'functions\cell_labeling.R'
        name = "BM_3"
        args = ["BM_3", self.res] # File name and resolution
        output_path = "../../../../data/processed/immune_cells/bone_marrow_human/BM_3_adata.h5ad"

        self.cell_labeling(RScript_path, name, args, output_path)

class BM_4(automatic_preprocessing):

    def __init__(self, res):
        super().__init__()
        self.res = res

    def QC_filter_(self):
        # Path of data
        file_base = '../../../../data/raw/immune_cells/bone_marrow_human/GSM3396183_T/'
        count_file_end = 'GSM3396183_matrix_T.mtx.gz'
        gene_data_end = 'GSM3396183_genes_T.tsv.gz'
        barcode_data_end = 'GSM3396183_barcodes_T.tsv.gz'
        thresholds = [5,5,5,5]
        name = "BM_4"

        self.QC_filter(file_base=file_base, name=name, count_file_end=count_file_end, gene_data_end=gene_data_end, barcode_data_end=barcode_data_end, thresholds=thresholds)
    
    def cell_labeling_(self):
        RScript_path = 'functions\cell_labeling.R'
        name = "BM_4"
        args = ["BM_4", self.res] # File name and resolution
        output_path = "../../../../data/processed/immune_cells/bone_marrow_human/BM_4_adata.h5ad"

        self.cell_labeling(RScript_path, name, args, output_path)

class BM_5(automatic_preprocessing):

    def __init__(self, res):
        super().__init__()
        self.res = res

    def QC_filter_(self):
        # Path of data
        file_base = '../../../../data/raw/immune_cells/bone_marrow_human/GSM3396174_N/'
        count_file_end = 'GSM3396174_matrix_N.mtx.gz'
        gene_data_end = 'GSM3396174_genes_N.tsv.gz'
        barcode_data_end = 'GSM3396174_barcodes_N.tsv.gz'
        thresholds = [5,5,5,5]
        name = "BM_5"

        self.QC_filter(file_base=file_base, name=name, count_file_end=count_file_end, gene_data_end=gene_data_end, barcode_data_end=barcode_data_end, thresholds=thresholds)
    
    def cell_labeling_(self):
        RScript_path = 'functions\cell_labeling.R'
        name = "BM_5"
        args = ["BM_5", self.res] # File name and resolution
        output_path = "../../../../data/processed/immune_cells/bone_marrow_human/BM_5_adata.h5ad"

        self.cell_labeling(RScript_path, name, args, output_path)

class PBMCs_1(automatic_preprocessing):

    def __init__(self, res):
        super().__init__()
        self.res = res

    def QC_filter_(self):
        # Path of data
        file_base = '../../../../data/raw/immune_cells/pbmcs_human/GSE115189/GSM3169075_filtered_gene_bc_matrices_h5.h5'
        thresholds = [5,5,5,5]
        name = "PBMCs_1"

        self.QC_filter(file_base=file_base, name=name, thresholds=thresholds, read_option="10X_h5")
    
    def cell_labeling_(self):
        RScript_path = 'functions\cell_labeling.R'
        name = "PBMCs_1"
        args = ["PBMCs_1", self.res] # File name and resolution
        output_path = "../../../../data/processed/immune_cells/pbmcs_human/PBMCs_1_adata.h5ad"

        self.cell_labeling(RScript_path, name, args, output_path)

class PBMCs_2(automatic_preprocessing):

    def __init__(self, res):
        super().__init__()
        self.res = res

    def QC_filter_(self):
        # Path of data
        file_base = '../../../../data/raw/immune_cells/pbmcs_human/GSM3665016_cs_1/'
        count_file_end = 'GSM3665016_CS_PBL_matrix.mtx.gz'
        gene_data_end = 'GSM3665016_CS_PBL_genes.tsv.gz'
        barcode_data_end = 'GSM3665016_CS_PBL_barcodes.tsv.gz'
        thresholds = [5,5,5,5]
        name = "PBMCs_2"

        self.QC_filter(file_base=file_base, name=name, count_file_end=count_file_end, gene_data_end=gene_data_end, barcode_data_end=barcode_data_end, thresholds=thresholds)
    
    def cell_labeling_(self):
        RScript_path = 'functions\cell_labeling.R'
        name = "PBMCs_2"
        args = ["PBMCs_2", self.res] # File name and resolution
        output_path = "../../../../data/processed/immune_cells/pbmcs_human/PBMCs_2_adata.h5ad"

        self.cell_labeling(RScript_path, name, args, output_path)

class PBMCs_3(automatic_preprocessing):

    def __init__(self, res):
        super().__init__()
        self.res = res

    def QC_filter_(self):
        # Path of data
        file_base = '../../../../data/raw/immune_cells/pbmcs_human/GSM3665017_kc_2/'
        count_file_end = 'GSM3665017_KC_PBL_matrix.mtx.gz'
        gene_data_end = 'GSM3665017_KC_PBL_genes.tsv.gz'
        barcode_data_end = 'GSM3665017_KC_PBL_barcodes.tsv.gz'
        thresholds = [5,5,5,5]
        name = "PBMCs_3"

        self.QC_filter(file_base=file_base, name=name, count_file_end=count_file_end, gene_data_end=gene_data_end, barcode_data_end=barcode_data_end, thresholds=thresholds)
    
    def cell_labeling_(self):
        RScript_path = 'functions\cell_labeling.R'
        name = "PBMCs_3"
        args = ["PBMCs_3", self.res] # File name and resolution
        output_path = "../../../../data/processed/immune_cells/pbmcs_human/PBMCs_3_adata.h5ad"

        self.cell_labeling(RScript_path, name, args, output_path)

class PBMCs_4(automatic_preprocessing):

    def __init__(self, res):
        super().__init__()
        self.res = res

    def QC_filter_(self):
        # Path of data
        file_base = '../../../../data/raw/immune_cells/pbmcs_human/GSM3665018_tb_3/'
        count_file_end = 'GSM3665018_TB_PBL_matrix.mtx.gz'
        gene_data_end = 'GSM3665018_TB_PBL_genes.tsv.gz'
        barcode_data_end = 'GSM3665018_TB_PBL_barcodes.tsv.gz'
        thresholds = [5,5,5,5]
        name = "PBMCs_4"

        self.QC_filter(file_base=file_base, name=name, count_file_end=count_file_end, gene_data_end=gene_data_end, barcode_data_end=barcode_data_end, thresholds=thresholds)
    
    def cell_labeling_(self):
        RScript_path = 'functions\cell_labeling.R'
        name = "PBMCs_4"
        args = ["PBMCs_4", self.res] # File name and resolution
        output_path = "../../../../data/processed/immune_cells/pbmcs_human/PBMCs_4_adata.h5ad"

        self.cell_labeling(RScript_path, name, args, output_path)

class PBMCs_5(automatic_preprocessing):

    def __init__(self, res):
        super().__init__()
        self.res = res

    def QC_filter_(self):
        # Path of data
        file_base = '../../../../data/raw/immune_cells/pbmcs_human/GSM3665019_tc_4/'
        count_file_end = 'GSM3665019_TC_PBL_matrix.mtx.gz'
        gene_data_end = 'GSM3665019_TC_PBL_genes.tsv.gz'
        barcode_data_end = 'GSM3665019_TC_PBL_barcodes.tsv.gz'
        thresholds = [5,5,5,5]
        name = "PBMCs_5"

        self.QC_filter(file_base=file_base, name=name, count_file_end=count_file_end, gene_data_end=gene_data_end, barcode_data_end=barcode_data_end, thresholds=thresholds)
    
    def cell_labeling_(self):
        RScript_path = 'functions\cell_labeling.R'
        name = "PBMCs_5"
        args = ["PBMCs_5", self.res] # File name and resolution
        output_path = "../../../../data/processed/immune_cells/pbmcs_human/PBMCs_5_adata.h5ad"

        self.cell_labeling(RScript_path, name, args, output_path)

class PBMCs_6(automatic_preprocessing):

    def __init__(self, res):
        super().__init__()
        self.res = res

    def QC_filter_(self):
        # Path of data
        file_base = '../../../../data/raw/immune_cells/pbmcs_human/10X_pbmcs_10k/'
        thresholds = [5,5,5,5]
        name = "PBMCs_6"

        self.QC_filter(file_base=file_base, name=name, thresholds=thresholds, read_option="10X_mtx")
    
    def cell_labeling_(self):
        RScript_path = 'functions\cell_labeling.R'
        name = "PBMCs_6"
        args = ["PBMCs_6", self.res] # File name and resolution
        output_path = "../../../../data/processed/immune_cells/pbmcs_human/PBMCs_6_adata.h5ad"

        self.cell_labeling(RScript_path, name, args, output_path)

class Merge():

    def __init__(self):
        pass

    def Oetjen_merge(self):

        file_paths = '../../../../data/processed/immune_cells/bone_marrow_human/'

        filenames = ['BM_1_adata.h5ad', 'BM_2_adata.h5ad', 'BM_3_adata.h5ad', 'BM_4_adata.h5ad', 'BM_5_adata.h5ad']

        adata = []
        samp_let = ['A', 'P', 'U', 'T', 'N']
        for k, name in enumerate(filenames):
            file = file_paths + name
            adata_temp = sc.read(file, cache=True)
            adata_temp.obs['patientID'] = ['Oetjen_'+samp_let[k]]*adata_temp.n_obs
            adata_temp.obs['study'] = ['Oetjen']*adata_temp.n_obs
            adata_temp.obs['chemistry'] = ['v2_10X']*adata_temp.n_obs
            adata_temp.obs['tissue'] = ['Bone_Marrow']*adata_temp.n_obs
            adata_temp.obs['species'] = ['Human']*adata_temp.n_obs
            adata_temp.obs['data_type'] = ['UMI']*adata_temp.n_obs
            adata.append(adata_temp) 

        adata = adata[0].concatenate(adata[1:], batch_key='sample_ID', 
                                               batch_categories=['Oetjen_A','Oetjen_P', 'Oetjen_U', 'Oetjen_T', 'Oetjen_N'])

        adata.obs.index.rename('barcode', inplace=True)
        # Assign adata.X to be the preprocessed unnormalized data
        adata.X = adata.layers['pp_counts']

        # Normalize
        adata = log1p_normalize(adata)

        # Download
        adata.write("../../../../data/processed/immune_cells/merged/Oetjen_merged.h5ad")

    def Freytag_merge(self):
        file_paths = '../../../../data/processed/immune_cells/pbmcs_human/'

        filenames = ['PBMCs_1_adata.h5ad']

        adata = []
        for k, name in enumerate(filenames):
            file = file_paths + name
            adata_temp = sc.read(file, cache=True)
            adata_temp.obs['patientID'] = ['Freytag']*adata_temp.n_obs
            adata_temp.obs['study'] = ['Freytag']*adata_temp.n_obs
            adata_temp.obs['chemistry'] = ['v2_10X']*adata_temp.n_obs
            adata_temp.obs['tissue'] = ['PBMCs']*adata_temp.n_obs
            adata_temp.obs['species'] = ['Human']*adata_temp.n_obs
            adata_temp.obs['data_type'] = ['UMI']*adata_temp.n_obs
            adata.append(adata_temp) 

        adata = adata[0].concatenate(adata[1:], batch_key='sample_ID', 
                                               batch_categories=['Freytag'])

        adata.obs.index.rename('barcode', inplace=True)
        # Assign adata.X to be the preprocessed unnormalized data
        adata.X = adata.layers['pp_counts']

        # Normalize
        adata = log1p_normalize(adata)

        # Download
        adata.write("../../../../data/processed/immune_cells/merged/Freytag_merged.h5ad")

    def Sun_merge(self):
        file_paths = '../../../../data/processed/immune_cells/pbmcs_human/'

        filenames = ['PBMCs_2_adata.h5ad', 'PBMCs_3_adata.h5ad', 'PBMCs_4_adata.h5ad', 'PBMCs_5_adata.h5ad']

        adata = []
        samp_let = ['1', '2', '3', '4']
        for k, name in enumerate(filenames):
            file = file_paths + name
            adata_temp = sc.read(file, cache=True)
            adata_temp.obs['patientID'] = ['Sun_'+samp_let[k]]*adata_temp.n_obs
            adata_temp.obs['study'] = ['Sun']*adata_temp.n_obs
            adata_temp.obs['chemistry'] = ['10X']*adata_temp.n_obs
            adata_temp.obs['tissue'] = ['PBMCs']*adata_temp.n_obs
            adata_temp.obs['species'] = ['Human']*adata_temp.n_obs
            adata_temp.obs['data_type'] = ['UMI']*adata_temp.n_obs
            adata.append(adata_temp) 

        adata = adata[0].concatenate(adata[1:], batch_key='sample_ID', 
                                               batch_categories=['Sun_1','Sun_2', 'Sun_3', 'Sun_4'])

        adata.obs.index.rename('barcode', inplace=True)
        # Assign adata.X to be the preprocessed unnormalized data
        adata.X = adata.layers['pp_counts']

        # Normalize
        adata = log1p_normalize(adata)

        # Download
        adata.write("../../../../data/processed/immune_cells/merged/Sun_merged.h5ad")

    def merge_10X(self):
        file_paths = '../../../../data/processed/immune_cells/pbmcs_human/'

        filenames = ['PBMCs_6_adata.h5ad']

        adata = []
        for k, name in enumerate(filenames):
            file = file_paths + name
            adata_temp = sc.read(file, cache=True)
            adata_temp.obs['patientID'] = ['10X']*adata_temp.n_obs
            adata_temp.obs['study'] = ['10X']*adata_temp.n_obs
            adata_temp.obs['chemistry'] = ['v3_10X']*adata_temp.n_obs
            adata_temp.obs['tissue'] = ['PBMCs']*adata_temp.n_obs
            adata_temp.obs['species'] = ['Human']*adata_temp.n_obs
            adata_temp.obs['data_type'] = ['UMI']*adata_temp.n_obs
            adata.append(adata_temp) 

        adata = adata[0].concatenate(adata[1:], batch_key='sample_ID', 
                                               batch_categories=['10X'])

        adata.obs.index.rename('barcode', inplace=True)
        # Assign adata.X to be the preprocessed unnormalized data
        adata.X = adata.layers['pp_counts']

        # Normalize
        adata = log1p_normalize(adata)

        # Download
        adata.write("../../../../data/processed/immune_cells/merged/10X_merged.h5ad")

    def Immune_cells_merge_all(self):
        file_paths = '../../../../data/processed/immune_cells/merged/'

        filenames = ['Oetjen_merged.h5ad', 'Freytag_merged.h5ad', 'Sun_merged.h5ad', '10X_merged.h5ad']

        adata = []
        for k, name in enumerate(filenames):
            file = file_paths + name
            adata_temp = sc.read(file, cache=True)
            adata.append(adata_temp) 

        adata = adata[0].concatenate(adata[1:], batch_key='sample_ID', index_unique=None)

        adata.obs.index.rename('barcode', inplace=True)
        # Assign adata.X to be the preprocessed unnormalized data
        adata.X = adata.layers['pp_counts']

        # Normalize
        adata = log1p_normalize(adata)

        # Download
        adata.write("../../../../data/processed/immune_cells/merged/Immune_cells_merged_all.h5ad")

def auto_preprocessing_and_labeling(resolution: str = "0.8", delete: bool = False):
    """
    This function performs preprocessing of single-cell RNA-seq data, including labeling, for all data.

    Parameters:
        - resolution (str): Resolution used for creating clusters using the Louvain algorithm.
        - delete (bool): Whether to delete all intermediate AnnData objects or not. The final merged one will always be kept
    """
    
    # Human Bone Marrow
    BM_1_env = BM_1(res = resolution)
    BM_1_env.QC_filter_()
    BM_1_env.cell_labeling_()

    BM_2_env = BM_2(res = resolution)
    BM_2_env.QC_filter_()
    BM_2_env.cell_labeling_()

    BM_3_env = BM_3(res = resolution)
    BM_3_env.QC_filter_()
    BM_3_env.cell_labeling_()

    BM_4_env = BM_4(res = resolution)
    BM_4_env.QC_filter_()
    BM_4_env.cell_labeling_()

    BM_5_env = BM_5(res = resolution)
    BM_5_env.QC_filter_()
    BM_5_env.cell_labeling_()

    # Human PBMC
    PBMCs_1_env = PBMCs_1(res = resolution)
    PBMCs_1_env.QC_filter_()
    PBMCs_1_env.cell_labeling_()

    PBMCs_2_env = PBMCs_2(res = resolution)
    PBMCs_2_env.QC_filter_()
    PBMCs_2_env.cell_labeling_()

    PBMCs_3_env = PBMCs_3(res = resolution)
    PBMCs_3_env.QC_filter_()
    PBMCs_3_env.cell_labeling_()

    PBMCs_4_env = PBMCs_4(res = resolution)
    PBMCs_4_env.QC_filter_()
    PBMCs_4_env.cell_labeling_()

    PBMCs_5_env = PBMCs_5(res = resolution)
    PBMCs_5_env.QC_filter_()
    PBMCs_5_env.cell_labeling_()

    PBMCs_6_env = PBMCs_6(res = resolution)
    PBMCs_6_env.QC_filter_()
    PBMCs_6_env.cell_labeling_()

    # Merge Immune cells
    merge_env = Merge()
    merge_env.Oetjen_merge()
    merge_env.Freytag_merge()
    merge_env.Sun_merge()
    merge_env.merge_10X()
    merge_env.Immune_cells_merge_all()

    # Delete all temporary files and keep the final merged AnnData object
    if delete:
        os.remove("../../../../data/processed/immune_cells/bone_marrow_human/BM_1_adata.h5ad")
        os.remove("../../../../data/processed/immune_cells/bone_marrow_human/BM_2_adata.h5ad")
        os.remove("../../../../data/processed/immune_cells/bone_marrow_human/BM_3_adata.h5ad")
        os.remove("../../../../data/processed/immune_cells/bone_marrow_human/BM_4_adata.h5ad")
        os.remove("../../../../data/processed/immune_cells/bone_marrow_human/BM_5_adata.h5ad")

        os.remove("../../../../data/processed/immune_cells/pbmcs_human/PBMCs_1_adata.h5ad")
        os.remove("../../../../data/processed/immune_cells/pbmcs_human/PBMCs_2_adata.h5ad")
        os.remove("../../../../data/processed/immune_cells/pbmcs_human/PBMCs_3_adata.h5ad")
        os.remove("../../../../data/processed/immune_cells/pbmcs_human/PBMCs_4_adata.h5ad")
        os.remove("../../../../data/processed/immune_cells/pbmcs_human/PBMCs_5_adata.h5ad")
        os.remove("../../../../data/processed/immune_cells/pbmcs_human/PBMCs_6_adata.h5ad")

        os.remove("../../../../data/processed/immune_cells/merged/Oetjen_merged.h5ad")
        os.remove("../../../../data/processed/immune_cells/merged/Freytag_merged.h5ad")
        os.remove("../../../../data/processed/immune_cells/merged/Sun_merged.h5ad")
        os.remove("../../../../data/processed/immune_cells/merged/10X_merged.h5ad")











        
