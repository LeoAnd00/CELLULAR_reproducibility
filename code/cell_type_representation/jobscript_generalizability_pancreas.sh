#!/usr/bin/env bash
#SBATCH -A NAISS2023-5-429 -p alvis
#SBATCH -t 15:00:00
#SBATCH --gpus-per-node=A100:1

apptainer exec /cephyr/users/leoan/Alvis/CELLULAR_reproducibility/CELLULAR_env.sif python benchmark_generalizability.py '/mimer/NOBACKUP/groups/naiss2023-6-290/Leo_Andrekson/Masters_Thesis/data/processed/pancreas_cells/pancreas_1_adata.h5ad' 'trained_models/Assess_generalisability/Pancreas/' 'benchmarks/results/Generalizability/Pancreas/Benchmark_results' '_Generalizability/Pancreas/'