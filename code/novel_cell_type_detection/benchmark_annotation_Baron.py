# Load packages
import numpy as np
import pandas as pd
import scanpy as sc
import scib
import warnings
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import re
import torch
import random
from benchmark_classifiers import classifier_train as benchmark

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

### Commands to run on Alvis cluster
# Start by runing: cd CELLULAR_reproducibility/code/novel_cell_type_detection/
# Then:
# sbatch jobscript_annotation_baron.sh


def main(data_path: str, model_path: str, result_csv_path: str, image_path: str, dataset_name: str):
    """
    Execute the code for training model1 for novel cell type detection. 
    This code is for the Baron dataset.
    Performs 5-fold cross testing.

    Parameters:
    - data_path (str): File path to the AnnData object containing expression data and metadata.
    - model_path (str): Directory path to save the trained model and predictions.
    - result_csv_path (str): File path to save the benchmark results as a CSV file.
    - image_path (str): Path where images will be saved.
    - dataset_name (str): Name of dataset.

    Returns:
    None 
    """
    
    # Calculate for model at different number of patient for training and different random seeds
    folds = [1,2,3,4,5] 
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

    exclude_cell_types_list_names = exclude_cell_types_list

    seed = 42

    novel_cell_counter = -1
    for novel_cell in exclude_cell_types_list:
        novel_cell_counter += 1

        for fold in folds:

            print("fold: ", fold)

            benchmark_env = benchmark(data_path=data_path,
                                    exclude_cell_types = novel_cell,
                                    dataset_name=dataset_name,
                                    image_path=image_path,
                                    fold=fold,
                                    seed=seed)

            # Calculate for model
            print(f"Start training model, fold {fold} and seed {seed}")
            print()
            benchmark_env.CELLULAR_classifier(save_path=f'{model_path}{exclude_cell_types_list_names[novel_cell_counter][0]}/CELLULAR/', train=True)

            del benchmark_env

            # Empty the cache
            torch.cuda.empty_cache()

    print("Finished generalizability benchmark!")
        
if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Run the benchmark with specified data, model, and result paths.')
    parser.add_argument('data_path', type=str, help='Path to the data file.')
    parser.add_argument('model_path', type=str, help='Path to save or load the trained models.')
    parser.add_argument('result_csv_path', type=str, help='Path to save the benchmark results as a CSV file.')
    parser.add_argument('image_path', type=str, help='Path where images will be saved.')
    parser.add_argument('dataset_name', type=str, help='Name of dataset.')
    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(args.data_path, args.model_path, args.result_csv_path, args.image_path, args.dataset_name)