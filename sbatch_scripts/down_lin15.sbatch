#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --time=8:00:00
#SBATCH --job-name=epoch7
##SBATCH --partition=gpu
#SBATCH --partition=ce-mri
#SBATCH --mem=50Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:v100:1

source activate simclr1
python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "runs/Apr30_23-07-46_d3102_e350" --comment "_e350_cfg_linear_ssl" &
sleep 60
python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "runs/Apr30_00-30-32_d1012_e500" --comment "_e500_cfg_linear_ssl"


