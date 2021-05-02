#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --time=4:00:00
#SBATCH --job-name=fp16
#SBATCH --partition=multigpu
#SBATCH --mem=50Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --exclusive

source activate simclr1
python pre_train.py --fp16-precision --comment "_default_fp_benchmark"
