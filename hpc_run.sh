#!/bin/bash
#SBATCH --job-name=train_miniGPT
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j_out.txt
#SBATCH --error=logs/%j_err.txt
#SBATCH --mail-user=my_email@sjsu.edu
#SBATCH --mail-type=END

module load python3/3.11.7
module load cuda

conda activate miniGPT

cd ~/projects/miniGPT
python train.py

