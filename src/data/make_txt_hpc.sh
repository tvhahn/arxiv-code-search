#!/bin/bash
#SBATCH --time=00:20:00 # 30 minutes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --account=rrg-mechefsk
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

module load python/3.8

source ~/arxiv/bin/activate

python src/data/make_txt.py --n_cores 8 --pdf_root_dir /home/tvhahn/scratch/arxiv-code-search/raw/pdfs