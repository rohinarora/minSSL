#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --time=8:00:00
#SBATCH --job-name=dtle3
#SBATCH --partition=ce-mri
#SBATCH --mem=50Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:a100:1

source activate simclr1
python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "runs/Apr30_03-51-19_d3099_default" -e 400 --comment "_default_cfg_linear_ssl_e400"
