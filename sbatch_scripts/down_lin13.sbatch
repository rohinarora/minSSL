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
python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "runs/Apr30_12-16-03_d3100_e150" --comment "_e150_cfg_linear_ssl" &
sleep 60
python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "runs/Apr30_12-15-02_d3100_e90" --comment "_e90_cfg_linear_ssl"


