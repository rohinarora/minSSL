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
python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "runs/May01_00-03-13_d1002_b2048" --comment "_b2048_cfg_linear_ssl"  -b 2048

