#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 1
#SBATCH --time=5:00:00
#SBATCH --job-name=batch3
#SBATCH --partition=multigpu
#SBATCH --mem=20Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:v100-sxm2:1

source activate simclr1
python linear_eval.py -e 400 --config_file runs/Apr29_17-04-56_d1020_default_config/config.yml --comment "_default_config_400_scratch" 