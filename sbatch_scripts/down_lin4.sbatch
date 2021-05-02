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
python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "runs/Apr29_23-17-12_d1008_cifar10_resnet18" --comment "_resnet18_cifar10_cfg_linear_ssl" 