#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --job-name=arch_data
#SBATCH --partition=multigpu
#SBATCH --mem=100Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:v100-sxm2:1

source activate simclr1
python pre_train.py --dataset-name cifar10  --arch resnet50  &
sleep 60
python pre_train.py --dataset-name stl10  --arch resnet50  &
sleep 60
python pre_train.py --dataset-name cifar10  --arch resnet18
