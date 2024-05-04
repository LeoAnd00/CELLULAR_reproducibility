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
# Start by runing: cd CELLULAR_reproducibility/code/cell_type_annotation_loss_comp/train_code/
# Then:
# sbatch jobscript_annotation_macparland.sh
# sbatch jobscript_annotation_segerstolpe.sh
# sbatch jobscript_annotation_baron.sh


def main(data_path: str, model_path: str, result_csv_path: str, image_path: str, dataset_name: str):
    """
    Execute the annotation generalizability benchmark pipeline for comparing loss functions. 
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
    
    folds = [1,2,3,4,5]
    num_folds = 5
    seed = 42
    counter = 0  
    for fold in folds:
        counter += 1

        print("fold: ", fold)

        benchmark_env = benchmark(data_path=data_path,
                                    dataset_name=dataset_name,
                                    image_path=image_path,
                                    HVGs=2000,
                                    fold=fold,
                                    seed=seed)

        # Calculate for model
        print(f"Start training model, fold {fold} and seed {seed}")
        print()
        benchmark_env.CELLULAR_full_loss_classifier(save_path=f'{model_path}CELLULAR_full_loss/', train=True, umap_plot=False, save_figure=False)
        benchmark_env.CELLULAR_CL_loss_classifier(save_path=f'{model_path}CELLULAR_CL_loss/', train=True, umap_plot=False, save_figure=False)
        benchmark_env.CELLULAR_centroid_loss_classifier(save_path=f'{model_path}CELLULAR_centroid_loss/', train=True, umap_plot=False, save_figure=False)
        
        benchmark_env.make_benchamrk_results_dataframe()

        #if counter > 1:
        #    benchmark_env.read_csv(name=result_csv_path)
        benchmark_env.read_csv(name=result_csv_path)

        benchmark_env.save_results_as_csv(name=result_csv_path)

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