# Load packages
from novel_ct_detction_fun import novel_cell_type_detection as novel_cell_type_detection
import warnings
import argparse

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

### Commands to run on Alvis cluster
# Start by runing: cd CELLULAR_reproducibility/code/novel_cell_type_detection/visualization/
# Then:
# sbatch jobscript_annotation_confidence.sh


def main(data_path: str, 
        model_path: str, 
        image_path: str):
    """
    Execute the annotation generalizability benchmark pipeline for novel cell type detection.
    Performs 5-fold cross testing.

    Parameters:
    - data_path (str): File path to the AnnData object containing expression data and metadata.
    - model_path (str): Directory path to save the trained model and predictions.
    - image_path (str): Path where images will be saved.

    Returns:
    None 
    """
    env = novel_cell_type_detection()

    env.main(data_path=data_path, 
            model_path=model_path, 
            image_path=image_path, 
            dataset_names=["MacParland", "Baron", "Zheng68k"])

        
if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Run the benchmark with specified data, model, and result paths.')
    parser.add_argument('data_path', type=str, help='Path to the data file.')
    parser.add_argument('model_path', type=str, help='Path to save or load the trained models.')
    parser.add_argument('image_path', type=str, help='Path where images will be saved.')
    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(args.data_path, args.model_path, args.image_path)