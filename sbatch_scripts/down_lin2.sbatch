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
python downstream_eval.py --downstream_task linear_eval -tm ImageNet --config_file runs/tmp_default/config.yml -e 400 --comment "_default_cfg_linear_imagenet_e400"