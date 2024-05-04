#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-429 -p alvis
#SBATCH -t 02:00:00
#SBATCH --gpus-per-node=A100:1

apptainer exec /cephyr/users/leoan/Alvis/CELLULAR_reproducibility/CELLULAR_env.sif python novel_cell_type_confidence.py '/mimer/NOBACKUP/groups/naiss2023-6-290/Leo_Andrekson/Masters_Thesis/data/processed/data_for_evaluating_cell_type_annotation' '../trained_models/Assess_generalisability/' 'figures/confidence_figure'