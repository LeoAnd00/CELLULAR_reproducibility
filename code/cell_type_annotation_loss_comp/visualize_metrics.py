import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from IPython.display import display
import seaborn as sns


class VisualizeEnv():

    def __init__(self):
        self.color_dict = None

    def read_csv(self, file: list):
        """
        Reads a CSV file and updates the performance metrics dataframe.

        Parameters
        ----------
        file : list, optional
            file path of the CSV files to read (don't add .csv at the end, it automatically does this).

        Returns
        -------
        None

        Notes
        -----
        This method reads a CSV file containing performance metrics and updates the metrics dataframe.
        """

        self.metrics = pd.read_csv(f'{file}.csv', index_col=0)
        self.metrics.columns = ["Method", 
                                "Accuracy", 
                                "Balanced Accuracy",
                                "F1 Score",
                                "Dataset",
                                "Fold"]
        
    def LossCompBoxPlotVisualization(self, image_path: str=None):
        """
        Generate a box plot visualization for accuracy, balanced accuracy and F1 score for each dataset and each loss function.

        Parameters
        --------
        image_path : str, optional 
            If provided, the plot will be saved as an SVG file with the specified file path/name (.svg is added by the function at the end). Defaults to None (meaning no image will be downloaded).
            
        Returns
        -------
        None
        """

        metrics = self.metrics.copy()

        metrics['Method'][metrics['Method'] == "CELLULAR CL + Centroid Loss"] = "CELLULAR | Centroid + CL Loss"
        metrics['Method'][metrics['Method'] == "CELLULAR CL Loss"] = "CELLULAR | CL Loss"
        metrics['Method'][metrics['Method'] == "CELLULAR Centroid Loss"] = "CELLULAR | Centroid Loss"

        #metrics = metrics.loc[metrics["Dataset"] == dataset_name,:]

        # Set up the figure and axis with 4 columns per row
        ncols = 1
        nrows = 3
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.08 * ncols, (7.08/2) * nrows), sharey=False)

        columns_metrics = self.metrics.columns[1:4].to_list()

        for i, metric in enumerate(columns_metrics):
            # Calculate the row and column indices
            col_idx = i % nrows

            # Group by model type, calculate mean and std, and sort by mean value of the current metric
            visual_metrics = metrics[['Dataset','Method',metric]]

            axs[col_idx].set_ylabel(metric, fontsize=7)
            variable = visual_metrics[metric].to_list()
            group = visual_metrics['Dataset'].to_list()
            group2 = visual_metrics['Method'].to_list()
            hue_order = ["CELLULAR | Centroid + CL Loss", 
                         "CELLULAR | Centroid Loss", 
                         "CELLULAR | CL Loss"]

            if col_idx == 0:
                sns.boxplot(y = variable,
                        x = group,
                        hue = group2, 
                        width = 0.6,
                        linewidth=0.4,
                        hue_order = hue_order,
                        ax=axs[col_idx], 
                        showfliers = False)
                
                sns.move_legend(
                    axs[col_idx], "lower center",
                    bbox_to_anchor=(.5, 1), ncol=len(hue_order), title=None, frameon=False, fontsize=7
                )
            else:
                sns.boxplot(y = variable,
                        x = group,
                        hue = group2, 
                        width = 0.6,
                        linewidth=0.4,
                        hue_order = hue_order,
                        ax=axs[col_idx], 
                        showfliers = False)
                axs[col_idx].legend().remove()

            # Add grid
            # Calculate the x positions of the grid lines to be between the ticks
            x_ticks = axs[col_idx].get_xticks()
            x_grid_positions = (x_ticks[:-1] + x_ticks[1:]) / 2

            # Set the grid positions to be between the x ticks
            axs[col_idx].set_xticks(x_grid_positions, minor=True)

            # Add grid lines between the x positions
            axs[col_idx].grid(axis='x', linestyle='--', alpha=1.0, zorder=1, which='minor')

            axs[col_idx].tick_params(axis='both', which='major', labelsize=7)  # Adjust font size for tick labels

        #sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 0.8), title=None, frameon=False)

        # Adjust layout to prevent clipping of ylabel
        plt.tight_layout()

        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}.svg', format='svg', dpi=300)

        plt.show()

    def BoxPlotVisualization(self, files, dataset_names, image_path: str=None, minmax_norm: bool=True):
        """
        Generate a box plot visualization of the overall metric from the scib package to assess the embedding space for each dataset of each loss function.

        Parameters
        --------
        files
            List contraining file paths to be used. One for each dataset.
        dataset_names
            List of dataset names for each file.
        image_path : str, optional 
            If provided, the plot will be saved as an SVG file with the specified file path/name (.svg is added by the function at the end). Defaults to None (meaning no image will be downloaded).
        minmax_norm: bool, optional
            Whether to min-max noramlize all metrics from scib or not. Default is True.
            
        Returns
        -------
        None
        """

        all_dataframes = []
        for file, dataset_name in zip(files,dataset_names):
            metrics = pd.read_csv(f'{file}.csv', index_col=0)

            if minmax_norm:
                columns = ["ASW_label/batch", 
                            "PCR_batch", 
                            "graph_conn",
                            "ASW_label", 
                            "isolated_label_silhouette", 
                            "NMI_cluster/label", 
                            "ARI_cluster/label",
                            "isolated_label_F1",
                            "cell_cycle_conservation"]
                # Min-max normalize each metric
                for metric in columns:
                    for fold in np.unique(metrics["fold"]):
                        mask = metrics["fold"] == fold
                        metrics.loc[mask, metric] = (metrics.loc[mask, metric] - metrics.loc[mask, metric].min()) / (metrics.loc[mask, metric].max() - metrics.loc[mask, metric].min())

                # calc overall scores for each fold and method
                for fold in np.unique(metrics["fold"]):
                    for method in np.unique(metrics.index):
                        mask = metrics["fold"] == fold
                        mask2 = metrics.index == method
                        metrics.loc[mask & mask2,"Overall Batch"] = metrics.loc[mask & mask2,["ASW_label/batch", "PCR_batch", "graph_conn"]].mean(axis=1)
                        metrics.loc[mask & mask2,"Overall Bio"] = metrics.loc[mask & mask2,["ASW_label", 
                                                        "isolated_label_silhouette", 
                                                        "NMI_cluster/label", 
                                                        "ARI_cluster/label",
                                                        "isolated_label_F1",
                                                        "cell_cycle_conservation"]].mean(axis=1)
                        metrics.loc[mask & mask2,"Overall"] = 0.4 * metrics.loc[mask & mask2,"Overall Batch"] + 0.6 * metrics.loc[mask & mask2,"Overall Bio"] 


            metrics = metrics.iloc[:,:-2]
            metrics["Dataset"] = [dataset_name]*metrics.shape[0]
            all_dataframes.append(metrics)

        metrics = pd.concat(all_dataframes, axis=0)
        metrics.columns = ["ASW | Bio", 
                            "ASW | Batch", 
                            "PCR | Batch",
                            "Isolated Label ASW | Bio",
                            "GC | Batch",
                            "NMI | Bio",
                            "ARI | Bio",
                            "CC | Bio",
                            "Isolated Label F1 | Bio",
                            "Overall Score | Batch",
                            "Overall Score | Bio",
                            "Overall Score",
                            "Dataset"]

        metrics['Method'] = metrics.index

         # Replace model names
        metrics['Method'][metrics['Method'] == "CELLULAR CL + Centroid Loss"] = "CELLULAR | Centroid + CL Loss"
        metrics['Method'][metrics['Method'] == "CELLULAR CL Loss"] = "CELLULAR | CL Loss"
        metrics['Method'][metrics['Method'] == "CELLULAR Centroid Loss"] = "CELLULAR | Centroid Loss"

        #metrics = metrics.loc[metrics["Dataset"] == dataset_name,:]

        # Set up the figure and axis with 4 columns per row
        ncols = 1
        nrows = 1
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.08 * ncols, (7.08/2) * nrows), sharey=False)
        axs = [axs]

        columns_metrics = ["Overall Score"]

        for i, metric in enumerate(columns_metrics):
            # Calculate the row and column indices
            col_idx = i % nrows

            # Group by model type, calculate mean and std, and sort by mean value of the current metric
            visual_metrics = metrics[['Dataset','Method',metric]]

            axs[col_idx].set_ylabel(metric, fontsize=7)
            variable = visual_metrics[metric].to_list()
            group = visual_metrics['Dataset'].to_list()
            group2 = visual_metrics['Method'].to_list()
            hue_order = ["CELLULAR | Centroid + CL Loss", 
                         "CELLULAR | Centroid Loss", 
                         "CELLULAR | CL Loss"]

            if col_idx == 0:
                sns.boxplot(y = variable,
                        x = group,
                        hue = group2, 
                        width = 0.6,
                        linewidth=0.4,
                        hue_order = hue_order,
                        ax=axs[col_idx], 
                        showfliers = False,
                        showmeans=False)
                
                sns.move_legend(
                    axs[col_idx], "lower center",
                    bbox_to_anchor=(.5, 1), ncol=len(hue_order), title=None, frameon=False, fontsize=7
                )
            else:
                sns.boxplot(y = variable,
                        x = group,
                        hue = group2, 
                        width = 0.6,
                        linewidth=0.4,
                        hue_order = hue_order,
                        ax=axs[col_idx], 
                        showfliers = False,
                        showmeans=False)
                axs[col_idx].legend().remove()

            # Add grid
            # Calculate the x positions of the grid lines to be between the ticks
            x_ticks = axs[col_idx].get_xticks()
            x_grid_positions = (x_ticks[:-1] + x_ticks[1:]) / 2

            # Set the grid positions to be between the x ticks
            axs[col_idx].set_xticks(x_grid_positions, minor=True)

            # Add grid lines between the x positions
            axs[col_idx].grid(axis='x', linestyle='--', alpha=1.0, zorder=1, which='minor')

            axs[col_idx].tick_params(axis='both', which='major', labelsize=7)  # Adjust font size for tick labels

        #sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 0.75), title=None, frameon=False)

        # Adjust layout to prevent clipping of ylabel
        plt.tight_layout()

        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}.svg', format='svg', dpi=300)

        plt.show()