#!/bin/bash
#SBATCH --job-name=train_miniGPT
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/017157582/miniGPT/logs/%j_out.txt
#SBATCH --error=/home/017157582/miniGPT/logs/%j_err.txt
#SBATCH --mail-user=my_email@sjsu.edu
#SBATCH --mail-type=END

module purge

module load python3/3.12.12
module load ml/torch/2.6

source ~/miniGPT/.venv/bin/activate

cd ~/projects/miniGPT
python3 train.py


