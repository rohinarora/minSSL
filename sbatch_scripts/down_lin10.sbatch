#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --time=24:00:00
#SBATCH --job-name=batch2
#SBATCH --partition=ce-mri
#SBATCH --mem=100Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:a100:1

source activate simclr1
python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "runs/Apr30_00-17-03_d1024_b1024" --comment "_b1024_cfg_linear_ssl"  -b 1024


