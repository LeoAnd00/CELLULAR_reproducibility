# Load packages
import numpy as np
import pandas as pd
import scanpy as sc
import scib
import warnings
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import json
import re
import torch
import random
from novel_cell_type_classifier import classifier_train as benchmark

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class novel_cell_type_detection():

    def __init__(self):
        pass

    def main(self, 
            data_path: str, 
            model_path: str, 
            image_path: str, 
            dataset_names: str):
        """
        Execute the annotation generalizability benchmark pipeline for novel cell type detection.
        Performs 5-fold cross testing.
        It calculates the minimum confidence of novel cell types and non-novel cell types of each fold and saves them.

        Parameters:
        - data_path (str): File path to the AnnData object containing expression data and metadata.
        - model_path (str): Directory path to save the trained model and predictions.
        - image_path (str): Path where images will be saved.
        - dataset_names (str): Name of dataset.

        Returns:
        None 
        """

        fig, axes = plt.subplots(nrows=1, ncols=len(dataset_names), figsize=(8*len(dataset_names), 6))
        if len(dataset_names) == 1:
            axes = [axes]

        model_path_core = model_path
        data_path_core = data_path
        
        colors = sns.color_palette('deep', 2)
        self.confidence_dict = {}
        for idx, dataset_name in enumerate(dataset_names):

            self.confidence_dict[dataset_name] = {}

            print(f"Start working with dataset: {dataset_name}")

            model_path = f"{model_path_core}{dataset_name}/"
            data_path = f"{data_path_core}/{dataset_name}.h5ad"

            novel_cells, novel_cell_names = self.get_cell_type_names(dataset_name)

            min_non_novel_confidence = []
            min_novel_confidence = []

            for novel_cell, novel_cell_name in zip(novel_cells, novel_cell_names):

                print(f"Start working with removing cell type: {novel_cell_name}")

                for fold in range(5):
                    fold += 1

                    seed = 42

                    benchmark_env = benchmark(data_path=data_path,
                                            exclude_cell_types = novel_cell,
                                            dataset_name=dataset_name,
                                            image_path=f"{image_path}/",
                                            HVGs=2000,
                                            HVG=False,
                                            fold=fold,
                                            seed=seed)
                    
                    # Calculate for model
                    min_non_novel_confidence_temp, min_novel_confidence_temp = benchmark_env.threshold_investigation(save_path=f'{model_path}{novel_cell_name}/CELLULAR/')
                    
                    min_non_novel_confidence.append(min_non_novel_confidence_temp)
                    min_novel_confidence.append(min_novel_confidence_temp)

                    del benchmark_env


            self.confidence_dict[dataset_name]["min_non_novel_confidence"] = min_non_novel_confidence
            self.confidence_dict[dataset_name]["min_novel_confidence"] = min_novel_confidence

            # Concatenate max_confidence and min_confidence lists
            all_confidence = min_non_novel_confidence + min_novel_confidence

            # Create a list of labels (0 for max_confidence, 1 for min_confidence)
            labels = ['Min Likelihood of Non-Novel Cell Type'] * len(min_non_novel_confidence) + ['Min Likelihood of Novel Cell Type'] * len(min_novel_confidence)
            
            # Jitter plot
            sns.stripplot(x=labels, y=all_confidence, jitter=True, palette=colors, alpha=0.7, ax=axes[idx])
            axes[idx].set_title(dataset_name, fontsize=14)
            axes[idx].set_xlabel('')
            axes[idx].set_ylabel('Likelihood', fontsize=14)
            axes[idx].grid(axis='y', linewidth=1.5)

            axes[idx].tick_params(axis='x', which='major', labelsize=10) 
            axes[idx].tick_params(axis='y', which='major', labelsize=10) 

        #plt.tight_layout()
        # Save the plot as an SVG file
        if image_path:
            plt.savefig(f'{image_path}.svg', format='svg')
        plt.show()

        with open("results/likelihood.json", 'w') as f:
            json.dump(self.confidence_dict, f, indent=4)

    def get_cell_type_names(self, datset_name: str):
        """
        Retrieves a list of lists containing all unique cell type names for each dataset.

        Parameters:
        - dataset_names (str): Name of dataset.

        Returns:
        exclude_cell_types_list: list containing all cell types of specified dataset. 
        """

        if datset_name == "MacParland":
            exclude_cell_types_list = [['Mature_B_Cells'], 
                                ['Plasma_Cells'], 
                                ['alpha-beta_T_Cells'], 
                                ['gamma-delta_T_Cells_1'],
                                ['gamma-delta_T_Cells_2'],
                                ['Central_venous_LSECs'], 
                                ['Cholangiocytes'], 
                                ['Non-inflammatory_Macrophage'], 
                                ['Inflammatory_Macrophage'],
                                ['Erythroid_Cells'], 
                                ['Hepatic_Stellate_Cells'],
                                ['NK-like_Cells'],
                                ['Periportal_LSECs'],
                                ['Portal_endothelial_Cells'],
                                ['Hepatocyte_1'],
                                ['Hepatocyte_2'],
                                ['Hepatocyte_3'],
                                ['Hepatocyte_4'],
                                ['Hepatocyte_5'],
                                ['Hepatocyte_6']]
        elif datset_name == "Baron":
            exclude_cell_types_list = [['acinar'], 
                                    ['beta'], 
                                    ['delta'], 
                                    ['activated_stellate'], 
                                    ['ductal'], 
                                    ['alpha'], 
                                    ['epsilon'], 
                                    ['gamma'], 
                                    ['endothelial'], 
                                    ['quiescent_stellate'], 
                                    ['macrophage'], 
                                    ['schwann'], 
                                    ['mast'], 
                                    ['t_cell']]
        elif datset_name == "Zheng68k":
            exclude_cell_types_list = [["CD8+ Cytotoxic T"], 
                                    ["CD8+/CD45RA+ Naive Cytotoxic"], 
                                    ["CD4+/CD45RO+ Memory"], 
                                    ["CD19+ B"], 
                                    ["CD4+/CD25 T Reg"], 
                                    ["CD56+ NK"], 
                                    ["CD4+ T Helper2"], 
                                    ["CD4+/CD45RA+/CD25- Naive T"], 
                                    ["CD34+"], 
                                    ["Dendritic"], 
                                    ["CD14+ Monocyte"]]
        else:
            raise ValueError("Not a valid dataset name! Must be MacParland, Segerstolpe or Baron")

        exclude_cell_types_list_names = [item[0] for item in exclude_cell_types_list]

        return exclude_cell_types_list, exclude_cell_types_list_names
    
    def calc_precision_and_coverage(self, threshold: float):
        """
        Calculates the precision and coverage on all datasets when using the specified likelihood threshold.

        Parameters:
        - threshold (float): Likelihood threshold. If a sample is below this value, it's considered to contain a novel cell type.

        Returns:
        None
        """

        with open("results/likelihood.json", 'r') as f:
            self.confidence_dict = json.load(f)

        num_cell_types = [1/20, 1/14, 1/11] # Inverse of number fo cell types for each dataset. Used for min-max noramlization.

        all_true_positives = 0
        all_false_positives = 0
        all_all_postives = 0
        print(f"Calculations are done with a confidence threshold of {threshold}")
        print("If a data point exists with a confidence below this point, we assume there exists a novel cell type in the data")
        print("")
        for idx, dataset_name in enumerate(self.confidence_dict):
            data = self.confidence_dict[dataset_name]

            min_non_novel_confidence = [(x - num_cell_types[idx]) / (1 - num_cell_types[idx]) for x in data["min_non_novel_confidence"]]
            min_novel_confidence = [(x - num_cell_types[idx]) / (1 - num_cell_types[idx]) for x in data["min_novel_confidence"]]

            min_non_novel_confidence_temp = np.array(min_non_novel_confidence)
            min_novel_confidence_temp = np.array(min_novel_confidence)

            false_positives = min_non_novel_confidence_temp[min_non_novel_confidence_temp <= threshold]
            true_positives = min_novel_confidence_temp[min_novel_confidence_temp <= threshold]
            all_postives = min_novel_confidence_temp

            precision = len(true_positives) / (len(true_positives) + len(false_positives))

            coverage = len(true_positives) / len(all_postives)

            print("Dataset: ", dataset_name)
            print("_______________________")
            print("Precision: ", precision)
            print("Coverage: ", coverage)
            print("_______________________")
            print("")

            all_true_positives += len(true_positives)
            all_false_positives += len(false_positives)
            all_all_postives += len(all_postives)

        all_precision = 0
        try:
            all_precision = all_true_positives / (all_true_positives + all_false_positives)
        except:
            all_precision = 0

        all_coverage = all_true_positives / all_all_postives

        print("Average across datasets")
        print("_______________________")
        print("Precision: ", all_precision)
        print("Coverage: ", all_coverage)
        print("_______________________")
        print("")

    def jitter_plot(self, dataset_names: str, image_path: str=None, threshold: float=0.26):
        """
        Makes a jitter plot over the minimum likelihood fo eahc folder, and for each novel and non-novel cell type.

        Parameters:
        - image_path (str): Path (including name of file) where image will be saved.
        - dataset_names (list): List containing the name of all datasets
        - threshold (float): Likelihood threshold. Will draw a red line at this point.

        Returns:
        None
        """

        with open("results/likelihood.json", 'r') as f:
            self.confidence_dict = json.load(f)

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=((7)*0.75, (7*1.5/6)*0.75))
        
        colors = sns.color_palette('deep', 2)

        titles = ["MacParland", "Baron", "Zheng68k", "All Datasets"]
        num_cell_types = [1/20, 1/14, 1/11]

        min_non_novel_confidence_all = []
        min_novel_confidence_all = []

        index_1 = -1
        for idx, dataset_name in enumerate(dataset_names):
            index_1 += 1 
            
            data = self.confidence_dict[dataset_name]
            min_non_novel_confidence = [(x - num_cell_types[idx]) / (1 - num_cell_types[idx]) for x in data["min_non_novel_confidence"]]
            min_novel_confidence = [(x - num_cell_types[idx]) / (1 - num_cell_types[idx]) for x in data["min_novel_confidence"]]
            min_non_novel_confidence_all.extend(min_non_novel_confidence)
            min_novel_confidence_all.extend(min_novel_confidence)
            
            # Concatenate max_confidence and min_confidence lists
            all_confidence = min_non_novel_confidence + min_novel_confidence

            # Create a list of labels (0 for max_confidence, 1 for min_confidence)
            labels = ['Min Likelihood of Non-Novel Cell Types'] * len(min_non_novel_confidence) + ['Min Likelihood of Novel Cell Types'] * len(min_novel_confidence)
            
            # Jitter plot
            sns.stripplot(x=labels, y=all_confidence, jitter=True, palette=colors, size=1.5, alpha=0.7, ax=axes[index_1])
            axes[index_1].set_title(titles[idx], fontsize=5)
            axes[index_1].set_xlabel('')
            if index_1 == 0:
                axes[index_1].set_ylabel('Likelihood', fontsize=5)
            axes[index_1].grid(axis='y', linewidth=0.5)

            axes[index_1].tick_params(axis='x', which='major', length=5, labelsize=5, width=0.0) 
            axes[index_1].tick_params(axis='y', which='major', length=5, labelsize=5, width=0.3) 

            axes[index_1].set_xticklabels([])

            if index_1 == 0:
                axes[index_1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                axes[index_1].set_ylim((0,1))
            else:
                axes[index_1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                axes[index_1].set_ylim((0,1))
                axes[index_1].set_yticklabels([])
                axes[index_1].tick_params(axis='y', which='major', labelsize=5, width=0.0) 

        # Concatenate max_confidence and min_confidence lists
        all_confidence = min_non_novel_confidence_all + min_novel_confidence_all

        # Create a list of labels (0 for max_confidence, 1 for min_confidence)
        labels = ['Min Likelihood of Non-Novel Cell Types'] * len(min_non_novel_confidence_all) + ['Min Likelihood of Novel Cell Types'] * len(min_novel_confidence_all)
        
        # Jitter plot
        sns.stripplot(x=labels, y=all_confidence, jitter=True, palette=colors, size=1.5, alpha=0.7, ax=axes[3])
        axes[3].set_title(titles[3], fontsize=5)
        axes[3].set_xlabel('')
        axes[3].grid(axis='y', linewidth=0.5)
        axes[3].axhline(y=threshold, color='red', linestyle='--', linewidth=0.6, alpha=0.5)

        axes[3].tick_params(axis='x', which='major', length=5, labelsize=5, width=0.0) 

        axes[3].set_xticklabels([])
        axes[3].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        axes[3].set_ylim((0,1))
        axes[3].set_yticklabels([])
        axes[3].tick_params(axis='y', which='major', length=5, labelsize=5, width=0.0) 

        border_thickness = 0.5  # Set your desired border thickness here
        for ax in axes.ravel():
            for spine in ax.spines.values():
                spine.set_linewidth(border_thickness)

        # Create a single legend for the entire figure
        fig.legend(['Min Likelihood of Non-Novel Cell Types', 'Min Likelihood of Novel Cell Types'], loc='lower center', bbox_to_anchor=(1.15, 0.5), fontsize=5, frameon=False, ncol=1, title='')

        plt.tight_layout(pad=0.0)
        #plt.tight_layout()
        # Save the plot as an SVG file
        if image_path:
            fig.savefig(f'{image_path}.svg', format='svg', dpi=300, bbox_inches='tight')
        plt.show()

    def scatter_line_plot(self, image_path: str=None, threshold_: float=0.26):
        """
        Makes a scatter plot over the precision and coverage at different thresholds, and for each novel and non-novel cell type.

        Parameters:
        - image_path (str): Path (including name of file) where image will be saved.
        - threshold (float): Likelihood threshold. Will draw a red line at this point.

        Returns:
        None
        """

        with open("results/likelihood.json", 'r') as f:
            self.confidence_dict = json.load(f)

        num_cell_types = [1/20, 1/14, 1/11]

        thresholds = np.linspace(0.17, 0.8, 100)  # Example linspace, adjust as needed

        results_precision = np.zeros((100, 4))
        results_coverage = np.zeros((100, 4))

        for i, threshold in enumerate(thresholds):
            all_true_positives = 0
            all_false_positives = 0
            all_all_postives = 0
            for j, dataset_name in enumerate(self.confidence_dict):
                data = self.confidence_dict[dataset_name]

                min_non_novel_confidence = [(x - num_cell_types[j]) / (1 - num_cell_types[j]) for x in data["min_non_novel_confidence"]]
                min_novel_confidence = [(x - num_cell_types[j]) / (1 - num_cell_types[j]) for x in data["min_novel_confidence"]]

                min_non_novel_confidence_temp = np.array(min_non_novel_confidence)
                min_novel_confidence_temp = np.array(min_novel_confidence)

                false_positives = min_non_novel_confidence_temp[min_non_novel_confidence_temp <= threshold]
                true_positives = min_novel_confidence_temp[min_novel_confidence_temp <= threshold]
                all_postives = min_novel_confidence_temp

                precision = 0
                try:
                    precision = len(true_positives) / (len(true_positives) + len(false_positives))
                except:
                    precision = 0

                coverage = len(true_positives) / len(all_postives)

                results_precision[i,j] = precision
                results_coverage[i,j] = coverage

                all_true_positives += len(true_positives)
                all_false_positives += len(false_positives)
                all_all_postives += len(all_postives)

            all_precision = 0
            try:
                all_precision = all_true_positives / (all_true_positives + all_false_positives)
            except:
                all_precision = 0

            all_coverage = all_true_positives / all_all_postives

            results_precision[i,3] = all_precision
            results_coverage[i,3] = all_coverage

        # Create subplots
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=((7)*0.89, (7*1.5/6)*0.75))

        # Plot on each subplot
        colors = sns.color_palette('deep', 5)
        handles = []
        labels = []
        titles = ["MacParland", "Baron", "Zheng68k", "All Datasets"]
        counter = -1
        for j in range(4):
            counter += 1
            # Scatter plot with line connecting points
            precision_line, = axs[j].plot(thresholds, results_precision[:, counter], 'o-', color=colors[-1], markersize=0.25, linewidth=0.15, label='Precision')
            coverage_line, = axs[j].plot(thresholds, results_coverage[:, counter], 'o-', color=colors[-3], markersize=0.25, linewidth=0.15, label='Coverage')
            if (i == 0) and (j == 0):
                handles.append(precision_line)
                handles.append(coverage_line)
                labels.append('Precision')
                labels.append('Coverage')
            axs[j].set_title(titles[counter], fontsize=5)
            if j == 0:
                axs[j].set_ylabel('Percentage', fontsize=5)
            if j == 3:
                axs[3].axvline(x=threshold_, color='red', linestyle='--', linewidth=0.6, alpha=0.5)

            axs[j].tick_params(axis='x', length=5, which='major', labelsize=5, width=0.3) 
            axs[j].tick_params(axis='y', length=5, which='major', labelsize=5, width=0.3) 

            if j == 0:
                axs[j].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                axs[j].set_ylim((0,1.05))
            else:
                axs[j].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                axs[j].set_ylim((0,1.05))
                axs[j].set_yticklabels([])
                axs[j].tick_params(axis='y', length=5, which='major', labelsize=5, width=0.0) 

        fig.text(0.525, -0.1, 'Likelihood Threshold', ha='center', fontsize=5)

        border_thickness = 0.5  # Set your desired border thickness here
        for ax in axs.ravel():
            for spine in ax.spines.values():
                spine.set_linewidth(border_thickness)

        # Create a single legend for the entire figure
        fig.legend(['Precision','Coverage'], loc='lower center', bbox_to_anchor=(1.05, 0.5), fontsize=5, frameon=False, ncol=1, title='')

        # Adjust layout
        fig.tight_layout(pad=0.0)
        # Save the plot as an SVG file
        if image_path:
            fig.savefig(f'{image_path}.svg', format='svg', dpi=300, bbox_inches='tight')
        fig.show()
        