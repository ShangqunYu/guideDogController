#!/bin/bash

#SBATCH -p gpu-preempt             # Submit job to gpu-preempt partition
#SBATCH -t 20:00:00                # Set max job time for 20 hours
#SBATCH --ntasks=1                 # Set the number of tasks to 1
#SBATCH --gpus-per-task=1          # Request 1 GPU per task
#SBATCH --constraint=l40s          # Request access to a rtx8000 GPU
#SBATCH --mem=50G                  # Request 50GB of memory
#SBATCH --output=logs/isaaclab_go1_vision/slurm-%A_%a.out  # Specify the output log file
#SBATCH --error=logs/isaaclab_go1_vision/slurm-%A_%a.err   # Specify the error log file
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=tmdang.daros@gmail.com # Send email to my daros email.

# Activate the conda environment
module load conda/latest
conda activate isaaclab

# Execute the Python script with the specific prompt
python -u source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Rough-Unitree-Go1-v0 --num_envs=2750 --headless
