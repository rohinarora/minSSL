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
python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "runs/Apr30_01-28-19_d3098_e250" --comment "_e250_cfg_linear_ssl" &
sleep 60
python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "runs/Apr30_23-02-57_d1016_e300" --comment "_e300_cfg_linear_ssl"
