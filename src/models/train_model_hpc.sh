#!/bin/bash
#SBATCH --account=rrg-mechefsk
#SBATCH --gres=gpu:1       # request GPU "generic resource"
#SBATCH --cpus-per-task=1   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=14000M      # memory per node
#SBATCH --time=0-00:02      # time (DD-HH:MM)
#SBATCH --job-name=var4
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

module load python/3.8

PROJECT_DIR=$1

source ~/arxiv/bin/activate

# copy processed data from scratch to the temporary directory used for batch job
# this will be much faster as the train_model.py rapidly access the training data
mkdir $SLURM_TMPDIR/data
cp -r ~/scratch/earth-mantle-surrogate/processed $SLURM_TMPDIR/data

# load tensorboard
# cd
# tensorboard --logdir=scratch/earth-mantle-surrogate/models/interim/logs --samples_per_plugin images=250 --host 0.0.0.0 &

# begin training
python $PROJECT_DIR/src/models/train_model.py \
    --proj_dir $PROJECT_DIR \
    --path_data $SLURM_TMPDIR/data/processed \
    --batch_size 1 \
    --var_to_include 4 \
    --learning_rate 1e-4 \
    --critic_iterations 5 \
    --num_epochs 1000 \
    --lambda_gp 10 \
    --gen_pretrain_epochs 1 \
    --model_time_suffix var4 \
    --checkpoint  2021_11_18_103508_var4 \
    # --cat_noise
