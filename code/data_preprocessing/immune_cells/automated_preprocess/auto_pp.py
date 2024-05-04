import argparse
from functions import data_preprocessing as dp

# Run in cmd to not delete all intermediate AnnData objects: cd code\data_preprocessing\immune_cells\automated_preprocess 
#                                                            python auto_pp.py --resolution 0.8
# Run in cmd for deleting all intermediate AnnData objects: cd code\data_preprocessing\immune_cells\automated_preprocess
#                                                           python auto_pp.py --resolution 0.8 --flag

def main(resolution, flag):
    dp.auto_preprocessing_and_labeling(resolution, flag)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This function performs preprocessing of single-cell RNA-seq data, including labeling, for all data.")
    parser.add_argument("--resolution", type=str, required=True, help="Specify the resolution")
    parser.add_argument("--flag", action="store_true", help="Specify whether to delete all intermediate AnnData objects or not (True/False)")

    args = parser.parse_args()
    main(args.resolution, args.flag)