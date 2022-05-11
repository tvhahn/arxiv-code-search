#!/bin/bash
#SBATCH --time=00:02:00 # 30 minutes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --account=rrg-mechefsk
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

module load python/3.8

PROJECT_DIR=$1
SCRATCH_DIR=~/scratch/arxiv-code-search

source ~/arxiv/bin/activate

# copy processed data from scratch to the temporary directory used for batch job
# this will be faster as the train_model.py can access the training data faster
mkdir -p $SLURM_TMPDIR/data
cp -r $SCRATCH_DIR/processed $SLURM_TMPDIR/data

# load tensorboard
# cd
# tensorboard --logdir=scratch/arxiv-code-search/models/interim/logs --samples_per_plugin images=250 --host 0.0.0.0 &

# begin training
python $PROJECT_DIR/src/models/train_model.py \
    --proj_dir $PROJECT_DIR \
    --path_data_dir $SLURM_TMPDIR/data \
    --path_model_dir $SCRATCH_DIR/models \
    --batch_size 1 \
    --n_epochs 1000 \
    --model_time_suffix var4 \
    --checkpoint  2021_11_18_103508_var4 \
