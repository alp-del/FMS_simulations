#!/bin/bash -l

# Slurm Directives
#SBATCH --job-name=train_job           # Job name
#SBATCH --output=train_output.log      # Standard output log file
#SBATCH --error=train_error.log        # Standard error log file
#SBATCH --ntasks=1                     # One task only
#SBATCH --cpus-per-task=4              # Four CPU cores per task
#SBATCH --mem=16G                      # Request 16 GB of memory
#SBATCH -p a100-4 
#SBATCH --gres=gpu:a100:1              # Request one GPU
#SBATCH --time=26:00:00                # Max wall time
#SBATCH --mail-type=BEGIN,END,FAIL     # Email notifications on job status
#SBATCH --mail-user=li004221@umn.edu   # Your email address

# Working Directory
cd ~/centering  

# Load Modules
module purge
module load python/3.6.3
module load cuda/10.1
module load cudnn/7.6.5

# Activate Virtual Environment
source myenv/bin/activate

# Execute the Job Script
bash run1.sh
