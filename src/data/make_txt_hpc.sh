#!/bin/bash
#SBATCH --time=00:20:00 # 30 minutes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --account=rrg-mechefsk
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

module load python/3.8

source ~/doiscrape/bin/activate

python src/data/make_txt.py --n_cores 32 --raw_data_dir /home/tvhahn/scratch/doi-scrape/raw