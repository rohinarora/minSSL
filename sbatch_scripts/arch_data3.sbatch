#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 1
#SBATCH --time=24:00:00
#SBATCH --job-name=arch_data3
#SBATCH --partition=multigpu
#SBATCH --mem=50Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:v100-sxm2:1

source activate simclr1
python pre_train.py --dataset-name cifar10  --arch resnet18 --comment "_cifar10_resnet18"
