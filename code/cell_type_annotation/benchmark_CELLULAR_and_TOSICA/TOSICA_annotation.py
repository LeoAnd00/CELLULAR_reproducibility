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
# Start by runing: cd CELLULAR_reproducibility/code/cell_type_annotation/benchmark_CELLULAR_and_TOSICA/
# Then:
# sbatch jobscript_TOSICA_macparland.sh
# sbatch jobscript_TOSICA_segerstolpe.sh
# sbatch jobscript_TOSICA_baron.sh
# sbatch jobscript_TOSICA_zheng68k.sh


def main(data_path: str, result_csv_path: str, dataset_name: str):
    """
    Execute the annotation generalizability benchmark pipeline. Performs 5-fold cross testing.

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
    seed = 42
    counter = 0  
    for fold in folds:
        counter += 1
        print("fold: ", fold)

        benchmark_env = benchmark(data_path=data_path,
                                    dataset_name=dataset_name,
                                    HVGs=2000,
                                    fold=fold,
                                    seed=seed)
        
        
        print("**Start benchmarking TOSICA method**")
        benchmark_env.tosica()

        if counter > 1:
            benchmark_env.read_csv(name=result_csv_path)
        #benchmark_env.read_csv(name=result_csv_path)

        benchmark_env.save_results_as_csv(name=result_csv_path)

        del benchmark_env

        # Empty the cache
        torch.cuda.empty_cache()

    print("Finished generalizability benchmark!")
        
if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Run the benchmark with specified data, model, and result paths.')
    parser.add_argument('data_path', type=str, help='Path to the data file.')
    parser.add_argument('result_csv_path', type=str, help='Path to save the benchmark results as a CSV file.')
    parser.add_argument('dataset_name', type=str, help='Name of dataset.')
    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(args.data_path, args.result_csv_path, args.dataset_name)