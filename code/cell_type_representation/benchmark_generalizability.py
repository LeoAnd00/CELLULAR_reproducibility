# Load packages
import warnings
import argparse
import re
import torch
import random
from benchmarks.benchmark_generalizability_fun import benchmark as benchmark

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

### Commands to run on Alvis cluster
# Start by runing: cd CELLULAR_reproducibility/code/cell_type_representation/
# Then:
# sbatch jobscript_generalizability_bone_marrow.sh
# sbatch jobscript_generalizability_pbmc.sh
# sbatch jobscript_generalizability_pancreas.sh
# sbatch jobscript_generalizability_kidney.sh
# sbatch jobscript_generalizability_all_merged.sh


def main(data_path: str, model_path: str, result_csv_path: str, image_path: str):
    """
    Execute the generalizability benchmark pipeline for making an embedding space. Performs 5-fold cross testing to
    evaluate generalizability.

    Parameters:
    - data_path (str): File path to the AnnData object containing expression data and metadata.
    - model_path (str): Directory path to save the trained model and predictions.
    - result_csv_path (str): File path to save the benchmark results as a CSV file.
    - image_path (str): Path where images will be saved.

    Returns:
    None 
    """
    
    train_split = 0.8
    folds = [1,2,3,4,5]
    num_folds = 5
    seed = 42
    counter = 0

    for fold in folds:
        counter += 1

        print("fold: ", fold)

        benchmark_env = benchmark(data_path=data_path, 
                                image_path=f'{image_path}train_pct_{train_split}_fold_{fold}_seed_{seed}_',
                                batch_key="patientID", 
                                HVG=True, 
                                HVGs=2000, 
                                num_folds=num_folds,
                                fold=fold,
                                pct_for_training=train_split,
                                seed=seed)
        
        """print("Start evaluating unintegrated data")
        print()
        benchmark_env.unintegrated(save_figure=False, umap_plot=False)

        print("Start evaluating PCA transformed data")
        print()
        benchmark_env.pca(save_figure=False, umap_plot=False)

        print("**Start benchmarking scVI method**")
        vae = benchmark_env.scvi(umap_plot=False,save_figure=False)

        print("**Start benchmarking scANVI method**")
        benchmark_env.scanvi(vae=vae,umap_plot=False,save_figure=False)

        print("**Start benchmarking scGen method**")
        benchmark_env.scgen(umap_plot=False,save_figure=False)"""

        print("**Start benchmarking TOSICA method**")
        benchmark_env.tosica(umap_plot=False,save_figure=False)

        # Calculate for model
        #print(f"Start training model with {train_split} percent of data for training, fold {fold} and seed {seed}")
        #print()
        #benchmark_env.CELLULAR_benchmark(save_path=f'{model_path}CELLULAR/', train=True, umap_plot=False, save_figure=False)
        
        benchmark_env.make_benchamrk_results_dataframe(min_max_normalize=False)

        benchmark_env.metrics["train_pct"] = [train_split]*benchmark_env.metrics.shape[0]
        benchmark_env.metrics["seed"] = [seed]*benchmark_env.metrics.shape[0]
        benchmark_env.metrics["fold"] = [fold]*benchmark_env.metrics.shape[0]

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
    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(args.data_path, args.model_path, args.result_csv_path, args.image_path)