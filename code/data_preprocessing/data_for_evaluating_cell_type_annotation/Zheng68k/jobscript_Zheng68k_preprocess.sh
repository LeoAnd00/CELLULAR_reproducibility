#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-429 -p alvis
#SBATCH -t 02:00:00
#SBATCH --gpus-per-node=A100:1

apptainer exec /cephyr/users/leoan/Alvis/CELLULAR_reproducibility/MScEnv.sif python Zheng68k_preprocess.py '/cephyr/users/leoan/Alvis/CELLULAR_reproducibility/data/raw/data_for_evaluating_cell_type_annotation/Zheng68k/filtered_matrices_mex/hg19/' '/cephyr/users/leoan/Alvis/CELLULAR_reproducibility/data/processed/data_for_evaluating_cell_type_annotation/Zheng68k'