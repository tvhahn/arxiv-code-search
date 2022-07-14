#!/bin/bash
#SBATCH --time=00:10:00 # 30 min
#SBATCH --array=1-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --account=rrg-mechefsk
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

PROJECT_DIR=$1

NOW_TIME=$2

SCRATCH_DATA_DIR=~/scratch/arxiv-code-search/data

module load python/3.8
source ~/arxiv/bin/activate

python $PROJECT_DIR/src/models_classical/train.py \
    --save_dir_name classical_results_interim_2022-07-14 \
    --path_data_dir $SCRATCH_DATA_DIR \
    --path_emb_dir $SCRATCH_DATA_DIR/processed/embeddings \
    --emb_file_name df_embeddings_2022-07-14.pkl \
    --rand_search_iter 8000

