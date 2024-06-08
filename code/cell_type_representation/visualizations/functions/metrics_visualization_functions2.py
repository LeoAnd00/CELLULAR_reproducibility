import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import display
import seaborn as sns


class VisualizeEnv():

    def __init__(self):
        self.color_dict = None

    def read_csv(self, files: list, minmax_norm: bool=True):
        """
        Reads a CSV file and updates the performance metrics dataframe.

        Parameters
        ----------
        files : list, optional
            List of file paths of the CSV files to read (don't add .csv at the end, it automatically does this).
        minmax_norm : bool, optional
            Whether to min-max normalize metrics or not. Default is True.

        Returns
        -------
        None

        Notes
        -----
        This method reads a CSV file containing performance metrics and updates the metrics dataframe.
        """

        all_dataframes = []
        for file in files:
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

            metrics = metrics.iloc[:,:-3]

            all_dataframes.append(metrics)

        self.metrics = pd.concat(all_dataframes, axis=0)
        self.metrics.columns = ["ASW | Bio", 
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
                                "Overall Score"]

    def BoxPlotVisualization_Appendix(self, image_path: str=None):
        """
        Generate a box plot visualization for each metric.

        Parameters
        --------
        image_path : str, optional 
            If provided, the plot will be saved as an SVG file with the specified file path/name (.svg is added by the function at the end). Defaults to None (meaning no image will be downloaded).
        
        Returns
        -------
        None
        """

        metrics = self.metrics.copy()
        metrics['Method'] = self.metrics.index

        metrics['Method'][metrics['Method'] == "Unintegrated"] = "Unint"

        # Set up the figure and axis with 4 columns per row
        ncols = 3
        nrows = -(-len(self.metrics.columns) // ncols)  # Ceiling division
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, (7/4) * nrows), sharey=False)

        # Get unique model types in order of performance on Overall metric
        metrics_temp = metrics.groupby(['Method'])["Overall Score"].agg(['mean', 'std']).reset_index()
        metrics_temp = metrics_temp.sort_values(by='mean')
        unique_model_types = metrics_temp['Method'].unique()

        # Define a colormap based on unique model types
        cmap = cm.get_cmap('tab20', len(unique_model_types))

        # Map each model type to a color using the colormap
        if self.color_dict is None:
            self.color_dict = {model_type: cmap(1 - j / (len(unique_model_types) - 1)) for j, model_type in enumerate(unique_model_types)}

        order = ["CELLULAR", 
                "scGen", 
                "scANVI", 
                "scVI", 
                "TOSICA", 
                "PCA", 
                "Unint"]
        
        for i, metric in enumerate(reversed(self.metrics.columns)):
            # Calculate the row and column indices
            row_idx = i // ncols
            col_idx = i % ncols

            # Group by model type, calculate mean and std, and sort by mean value of the current metric
            visual_metrics = pd.DataFrame({"Method": metrics['Method'], f"{metric}": metrics[metric]})

            # Plot boxplots for each model type in the specified subplot
            sns.boxplot(y=metric, 
                        x='Method', 
                        data=visual_metrics, 
                        ax=axs[row_idx, col_idx], 
                        order=order, 
                        hue='Method',
                        hue_order = order,
                        width = 0.6,
                        linewidth=0.4,
                        #palette=[self.color_dict[model_type] for model_type in visual_metrics['Method'].unique()], 
                        dodge=False, 
                        showfliers = False,
                        showmeans=False, 
                        zorder=2)


            # Set labels and title for each subplot
            axs[row_idx, col_idx].set_ylabel('Score', fontsize=5)
            axs[row_idx, col_idx].set_xlabel('')
            axs[row_idx, col_idx].set_title(metric, fontsize=5)

            # Ensure x-axis is visible for each subplot
            axs[row_idx, col_idx].tick_params(bottom=True, rotation=90)

            # Add grid
            #axs[row_idx, col_idx].grid(axis='y', linestyle='--', alpha=1.0, zorder=0)

            axs[row_idx, col_idx].tick_params(axis='both', which='major', labelsize=5)  # Adjust font size for tick labels

            axs[row_idx, col_idx].legend().remove()

        # Move the legend
        #handles, labels = axs[0, 1].get_legend_handles_labels()  # Get handles and labels from any subplot
        #fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=len(order), title=None, frameon=False, fontsize="medium")

        # Adjust layout to prevent clipping of xlabel
        plt.tight_layout()

        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}.svg', format='svg', dpi=300)

        plt.show()

    def BoxPlotVisualization(self, files, dataset_names, image_path: str=None, minmax_norm: bool=True):
        """
        Generate a box plot visualization for the overall metric.

        Parameters
        --------
        files
            List of file paths containing metric results for datasets.
        dataset_names
            Name of the datasets for each file in files.
        image_path : str, optional 
            If provided, the plot will be saved as an SVG file with the specified file path/name (.svg is added by the function at the end). Defaults to None (meaning no image will be downloaded).
        minmax_norm : bool, optional
            Whether to min-max normalize metrics or not. Default is True.
        
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


            metrics = metrics.iloc[:,:-3]
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

        metrics['Method'][metrics['Method'] == "Unintegrated"] = "Unint"

        #metrics = metrics.loc[metrics["Dataset"] == dataset_name,:]

        # Set up the figure and axis with 4 columns per row
        ncols = 1
        nrows = 1
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=((7/2), (7*1.5/6)), sharey=False)
        axs = [axs]

        columns_metrics = ["Overall Score"]

        for i, metric in enumerate(columns_metrics):
            # Calculate the row and column indices
            col_idx = i % nrows

            # Group by model type, calculate mean and std, and sort by mean value of the current metric
            visual_metrics = metrics[['Dataset','Method',metric]]

            axs[col_idx].set_ylabel(metric, fontsize=5)
            variable = visual_metrics[metric].to_list()
            group = visual_metrics['Dataset'].to_list()
            group2 = visual_metrics['Method'].to_list()
            hue_order = ["CELLULAR", 
                         "scGen", 
                         "scANVI", 
                         "scVI", 
                         "TOSICA", 
                         "PCA", 
                         "Unint"]
            
            #sns.move_legend(axs[col_idx], "upper left", bbox_to_anchor=(1, 0.75))
            """sns.boxplot(y = variable,
                        x = group,
                        hue = group2, 
                        width = 0.6,
                        linewidth=0.4,
                        hue_order = hue_order,
                        ax=axs[col_idx], 
                        showfliers = False)
            if col_idx != 1:
                axs[col_idx].legend().remove()"""

            if col_idx == 0:
                sns.boxplot(y = variable,
                        x = group,
                        hue = group2, 
                        width = 0.6,
                        linewidth=0.2,
                        hue_order = hue_order,
                        ax=axs[col_idx], 
                        showfliers = False,
                        showmeans=False)
                
                #sns.move_legend(
                #    axs[col_idx], "lower center",
                #    bbox_to_anchor=(.5, 1), ncol=len(hue_order), title=None, frameon=False, fontsize=7
                #)
                sns.move_legend(
                    axs[col_idx], "right",
                    bbox_to_anchor=(1.37, 0.5), ncol=1, title=None, frameon=False, fontsize=5
                )
            else:
                sns.boxplot(y = variable,
                        x = group,
                        hue = group2, 
                        width = 0.6,
                        linewidth=0.2,
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
            axs[col_idx].set_xticklabels(axs[col_idx].get_xticklabels(), rotation=45, ha='right', fontsize=5)

            # Add grid lines between the x positions
            axs[col_idx].grid(axis='x', linestyle='--', alpha=1.0, zorder=1, which='minor')

            axs[col_idx].tick_params(axis='both', which='major', labelsize=5)  # Adjust font size for tick labels


        #sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 0.75), title=None, frameon=False)

        # Adjust layout to prevent clipping of ylabel
        plt.tight_layout()

        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}.svg', format='svg', dpi=300)

        plt.show()

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
        """
        metrics = self.metrics.copy()
        metrics['Model Type'] = self.metrics.index
        # Replace model names
        
        metrics = metrics.groupby(['Model Type']).agg(['mean', 'std']).reset_index()
        metrics = metrics.sort_values(('Overall Score', 'mean'), ascending=False)
       
        styled_metrics = metrics.style.background_gradient(cmap=bg_color)
        display(styled_metrics)

