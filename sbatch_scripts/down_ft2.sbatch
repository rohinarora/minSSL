#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --time=8:00:00
#SBATCH --job-name=down_ft2
#SBATCH --partition=ce-mri
#SBATCH --mem=50Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:a100:1

source activate simclr1
python downstream_eval.py --downstream_task fine_tune -tm SSL -rd "runs/Apr30_03-51-19_d3099_default" -e 400 --comment "_default_cfg_ft_ssl_e400" &
