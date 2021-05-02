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
python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "runs/Apr30_12-13-29_d3100_e10" --comment "_e10_cfg_linear_ssl" &
sleep 60
python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "runs/Apr30_12-14-00_d3100_e30" --comment "_e30_cfg_linear_ssl"


