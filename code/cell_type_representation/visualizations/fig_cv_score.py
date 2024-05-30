# Load packages
import warnings
import argparse
import re
import torch
import random
from functions.CV_score_visualization_functions import VisualizeEnv as VisualizeEnv

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

### Commands to run on Alvis cluster
# Start by runing: cd CELLULAR_reproducibility/code/cell_type_representation/visualizations/
# Then:
# sbatch jobscript_fig_cv_score.sh


def main(data_path):
    """
    Produces a violin plot, showing the CV score distribution accross cell type distance calculations for each dataset.

    Parameters:
    - data_path (str): Path to the processed data folder.

    Returns:
    None 
    """
    
    # Initialize Env
    VisEnv = VisualizeEnv()

    # Make violin plot of CV
    VisEnv.run_all(base_path=data_path)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run code for visualizing the CV scores between cell type clusters and for all datasets.')
    parser.add_argument('data_path', type=str, help='Path to the processed data folder.')
    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(args.data_path)