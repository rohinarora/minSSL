#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 2
#SBATCH --time=8:00:00
#SBATCH --job-name=temp1
#SBATCH --partition=multigpu
#SBATCH --mem=30Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:v100-sxm2:1

source activate simclr1

python pre_train.py --temperature 0.14 --comment "_tmp14"
