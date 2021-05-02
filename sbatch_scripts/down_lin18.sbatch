#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --time=8:00:00
#SBATCH --job-name=epoch7
##SBATCH --partition=gpu
#SBATCH --partition=ce-mri
#SBATCH --mem=50Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:a100:1

source activate simclr1


python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "runs/May01_10-33-51_d3102_tmp175" --comment "_tmp175_cfg_linear_ssl" -e 400
