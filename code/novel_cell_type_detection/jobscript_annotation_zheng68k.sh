#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-429 -p alvis
#SBATCH -t 50:00:00
#SBATCH --gpus-per-node=A100:1

apptainer exec /cephyr/users/leoan/Alvis/CELLULAR_reproducibility/CELLULAR_env.sif python benchmark_annotation_Zheng68k.py '/mimer/NOBACKUP/groups/naiss2023-6-290/Leo_Andrekson/Masters_Thesis/data/processed/data_for_evaluating_cell_type_annotation/Zheng68k.h5ad' 'trained_models/Assess_generalisability/Zheng68k/' 'benchmark_results/Zheng68k/Benchmark_results' 'figures/' 'Zheng68k'