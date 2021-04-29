#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --job-name=batch3
#SBATCH --partition=multigpu
#SBATCH --mem=100Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:v100-sxm2:1

source activate simclr1
python pre_train.py -b 128 &
sleep 60
python pre_train.py -b 512 &
sleep 60
python pre_train.py -b 1024
