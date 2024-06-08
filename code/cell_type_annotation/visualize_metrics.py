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

    def BoxPlotVisualization(self, image_path: str=None):
        """
        Generate a box plot visualization of average performance across datasets for each metric.

        Parameters
        --------
        image_path : str, optional 
            If provided, the plot will be saved as an SVG file with the specified file path/name (.svg is added by the function at the end). Defaults to None (meaning no image will be downloaded).
            
        Returns
        -------
        None
        """

        metrics = self.metrics.copy()

        metrics['Method'][metrics['Method'] == "CELLULAR"] = "CELLULAR | HVGs"
        metrics['Method'][metrics['Method'] == "TOSICA"] = "TOSICA | HVGs"
        metrics['Method'][metrics['Method'] == "scNym_HVGs"] = "scNym | HVGs"
        metrics['Method'][metrics['Method'] == "Seurat_HVGs"] = "Seurat | HVGs"
        metrics['Method'][metrics['Method'] == "SciBet_HVGs"] = "SciBet | HVGs"
        metrics['Method'][metrics['Method'] == "CellID_cell_HVGs"] = "CellID_cell | HVGs"
        metrics['Method'][metrics['Method'] == "CellID_group_HVGs"] = "CellID_group | HVGs"

        #metrics = metrics.loc[metrics["Dataset"] == dataset_name,:]

        # Set up the figure and axis with 4 columns per row
        ncols = 1
        nrows = 3
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=((7/3), (7*1.5/3)), sharey=False)

        columns_metrics = self.metrics.columns[1:4].to_list()

        for i, metric in enumerate(columns_metrics):
            # Calculate the row and column indices
            col_idx = i % nrows

            # Group by model type, calculate mean and std, and sort by mean value of the current metric
            visual_metrics = metrics[['Dataset','Method',metric]]

            axs[col_idx].set_ylabel(metric, fontsize=5)
            variable = visual_metrics[metric].to_list()
            group = visual_metrics['Dataset'].to_list()
            group2 = visual_metrics['Method'].to_list()
            hue_order = ["CELLULAR | HVGs", 
                         "scNym", 
                         "scNym | HVGs", 
                         "Seurat", 
                         "Seurat | HVGs", 
                         "TOSICA | HVGs", 
                         "SciBet", 
                         "SciBet | HVGs", 
                         "CellID_cell", 
                         "CellID_group", 
                         "CellID_cell | HVGs", 
                         "CellID_group | HVGs"]
            
            #sns.move_legend(axs[col_idx], "upper left", bbox_to_anchor=(1, 0.75))
            sns.boxplot(y = variable,
                        x = group,
                        hue = group2, 
                        width = 0.6,
                        linewidth=0.2,
                        hue_order = hue_order,
                        ax=axs[col_idx], 
                        showfliers = False)
            if col_idx != 1:
                axs[col_idx].legend().remove()

            # Add grid
            # Calculate the x positions of the grid lines to be between the ticks
            x_ticks = axs[col_idx].get_xticks()
            x_grid_positions = (x_ticks[:-1] + x_ticks[1:]) / 2

            # Set the grid positions to be between the x ticks
            axs[col_idx].set_xticks(x_grid_positions, minor=True)

            # Add grid lines between the x positions
            axs[col_idx].grid(axis='x', linestyle='--', alpha=1.0, zorder=1, which='minor')

            axs[col_idx].tick_params(axis='both', which='major', labelsize=5, width=0.5)  # Adjust font size for tick labels

            axs[col_idx].legend().remove()
            """if col_idx == 0:
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
                    bbox_to_anchor=(.5, 1), ncol=len(hue_order), title=None, frameon=False,
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
                axs[col_idx].legend().remove()"""

        #sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 0.8), title=None, frameon=False, fontsize=5)

        border_thickness = 0.5  # Set your desired border thickness here
        for ax in axs.ravel():
            for spine in ax.spines.values():
                spine.set_linewidth(border_thickness)

        # Adjust layout to prevent clipping of ylabel
        plt.tight_layout()

        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}.svg', format='svg', dpi=300)

        plt.show()

    def BarPlotAverageMetricsVisualization(self, image_path: str=None):
        """
        Generate a bar plot visualization for the average performance across metrics.

        Parameters
        --------
        image_path : str, optional 
            If provided, the plot will be saved as an SVG file with the specified file path/name (.svg is added by the function at the end). Defaults to None (meaning no image will be downloaded).
            
        Returns
        -------
        None
        """

        metrics = self.metrics.copy()

        metrics['Method'][metrics['Method'] == "CELLULAR"] = "CELLULAR | HVGs"
        metrics['Method'][metrics['Method'] == "TOSICA"] = "TOSICA | HVGs"
        metrics['Method'][metrics['Method'] == "scNym_HVGs"] = "scNym | HVGs"
        metrics['Method'][metrics['Method'] == "Seurat_HVGs"] = "Seurat | HVGs"
        metrics['Method'][metrics['Method'] == "SciBet_HVGs"] = "SciBet | HVGs"
        metrics['Method'][metrics['Method'] == "CellID_cell_HVGs"] = "CellID_cell | HVGs"
        metrics['Method'][metrics['Method'] == "CellID_group_HVGs"] = "CellID_group | HVGs"

        # Calculate averge score of each metric in terms of each method and dataset
        averages = metrics.groupby(["Method", "Dataset"])[["Accuracy", "Balanced Accuracy", "F1 Score"]].mean()

        # Group by "Dataset"
        grouped_averages = averages.groupby("Dataset")

        # Define min-max normalization function
        def min_max_normalize(x):
            return (x - x.min()) / (x.max() - x.min())

        # Apply min-max normalization to each group
        normalized_averages = grouped_averages.transform(min_max_normalize)

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=((7/4.5), (7*1.5/6)), sharey=False)#, gridspec_kw={'width_ratios': [3, 1]}

        for col_idx, ax in enumerate(axs):

            if col_idx == 0:
                metrics = pd.DataFrame({"Accuracy": normalized_averages.reset_index()["Accuracy"],
                                        "Balanced\nAccuracy": normalized_averages.reset_index()["Balanced Accuracy"],
                                        "F1 Score": normalized_averages.reset_index()["F1 Score"],
                                        "Method": normalized_averages.index.get_level_values("Method")})
                metrics.reset_index(drop=True, inplace=True)

                # Melt the DataFrame to reshape it
                metrics = pd.melt(metrics, id_vars=['Method'], var_name='Metric', value_name='Value')

                variable = metrics["Value"].to_list()
                group = metrics['Metric'].to_list()
                group2 = metrics['Method'].to_list()
                hue_order = ["CELLULAR | HVGs", 
                            "scNym", 
                            "scNym | HVGs", 
                            "Seurat", 
                            "Seurat | HVGs", 
                            "TOSICA | HVGs", 
                            "SciBet", 
                            "SciBet | HVGs", 
                            "CellID_cell", 
                            "CellID_group", 
                            "CellID_cell | HVGs", 
                            "CellID_group | HVGs"]
                
                #sns.move_legend(axs[col_idx], "upper left", bbox_to_anchor=(1, 0.75))
                sns.boxplot(y = variable,
                            x = group,
                            hue = group2, 
                            width = 0.6,
                            linewidth=0.2,
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

                axs[col_idx].tick_params(axis='both', which='major', labelsize=5, width=0.5, rotation=30)  # Adjust font size for tick labels

                #axs[col_idx].set_ylabel('Normalized Average Score Across Datasets', fontsize=5)
                axs[col_idx].set_ylabel('Score', fontsize=5)

                plt.yticks([0, 0.5, 1.0], fontsize=5)

            elif col_idx == 1:
                # Group by "Method"
                method_averages = normalized_averages.groupby("Method")[["Accuracy", "Balanced Accuracy", "F1 Score"]].mean()

                # Calculate the mean across metrics for each method
                method_averages['Overall'] = method_averages.mean(axis=1)

                metrics = pd.DataFrame({"Accuracy": method_averages["Accuracy"],
                                        "Balanced Accuracy": method_averages["Balanced Accuracy"],
                                        "F1 Score": method_averages["F1 Score"],
                                        "Overall": method_averages["Overall"],
                                        "Method": method_averages.index})
                metrics.reset_index(drop=True, inplace=True)

                # Melt the DataFrame to convert it to long format
                melted_metrics = pd.melt(metrics, id_vars='Method', var_name='Metric', value_name='Value')

                # Sort the melted DataFrame by the 'Overall' metric
                overall_sorted = melted_metrics[melted_metrics['Metric'] == 'Overall'].sort_values(by='Value', ascending=False)

                hue_order = ["CELLULAR | HVGs", 
                                "scNym", 
                                "scNym | HVGs", 
                                "Seurat", 
                                "Seurat | HVGs", 
                                "TOSICA | HVGs", 
                                "SciBet", 
                                "SciBet | HVGs", 
                                "CellID_cell", 
                                "CellID_group", 
                                "CellID_cell | HVGs", 
                                "CellID_group | HVGs"]

                # Plot the grouped bar plot with opaque bars and borders
                sns.barplot(x='Metric', y='Value', hue='Method', linewidth=0.4, ax=axs[col_idx], data=overall_sorted, hue_order=hue_order, ci=None, dodge=True, alpha=1.0, edgecolor='black')

                #axs[col_idx].set_ylabel('Average Score Across Metrics', fontsize=5)
                axs[col_idx].set_ylabel('Average Score', fontsize=5)
                axs[col_idx].set_xlabel('', fontsize=1)

                axs[col_idx].tick_params(axis='both', which='major', labelsize=5, width=0.5)  # Adjust font size for tick labels

                axs[col_idx].set_xticklabels([])
                axs[col_idx].tick_params(axis='x', length=0)

                axs[col_idx].legend().remove()

                plt.yticks([0, 0.5, 1.0], fontsize=5)

        #sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 0.8), title=None, frameon=False, fontsize=7)
        # Get handles and labels from both axes
        handles, labels = axs[1].get_legend_handles_labels()

        # Create a legend for the entire subplot
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 0.89), title=None, frameon=False, fontsize=5)

        # Annotate subplots with letters
        #for ax, letter in zip(axs.ravel(), ['a', 'b']):
        #    ax.text(-0.1, 1.15, letter, transform=ax.transAxes, fontsize=5, fontweight='bold', va='top')

        border_thickness = 0.5  # Set your desired border thickness here
        for ax in axs.ravel():
            for spine in ax.spines.values():
                spine.set_linewidth(border_thickness)

        # Adjust layout to prevent clipping of ylabel
        plt.tight_layout()

        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}.svg', format='svg', dpi=300, bbox_inches='tight')

        plt.show()
