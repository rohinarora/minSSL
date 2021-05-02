#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --time=4:00:00
#SBATCH --job-name=fpdef
#SBATCH --partition=multigpu
#SBATCH --mem=50Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --exclusive

source activate simclr1
python pre_train.py --comment "_default_benchmark"
