#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 2
#SBATCH --time=24:00:00
#SBATCH --job-name=dl4
#SBATCH --partition=multigpu
#SBATCH --mem=60Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:v100-sxm2:1

source activate simclr1
python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "Apr30_11-54-51_d3099_stl10_resnet50" --comment "_resnet50_stl10_cfg_linear_ssl"
