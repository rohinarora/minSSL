#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --job-name=temp1
#SBATCH --partition=multigpu
#SBATCH --mem=100Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:v100-sxm2:1

source activate simclr1
python pre_train.py --temperature 0.035 --comment "_tmp35" &
sleep 60
python pre_train.py --temperature 0.105 --comment "_tmp105" &
sleep 60
python pre_train.py --temperature 0.14 --comment "_tmp14"
