#!/bin/bash
#SBATCH --time=00:10:00 # 10 minutes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --account=rrg-mechefsk
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

PROJECT_DIR=$1

SCRATCH_DATA_DIR=~/scratch/arxiv-code-search/data

module load python/3.8
source ~/arxiv/bin/activate

python $PROJECT_DIR/src/models_classical/compile.py -p $PROJECT_DIR --n_cores 12 --interim_dir_name classical_results_interim --final_dir_name final_results_classical

