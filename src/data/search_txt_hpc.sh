#!/bin/bash
#SBATCH --time=00:20:00 # 30 minutes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --account=rrg-mechefsk
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

module load python/3.8

source ~/arxiv/bin/activate

python src/data/search_txt.py --txt_root_dir /home/tvhahn/scratch/arxiv-code-search/data/raw/txts \
    --overwrite \
    --keep_old_files \
    --index_file_no 5 \
    --max_token_len 350