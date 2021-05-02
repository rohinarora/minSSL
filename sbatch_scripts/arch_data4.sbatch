#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --job-name=arch_data4
#SBATCH --partition=ce-mri
#SBATCH --mem=50Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:a100:1

source activate simclr1
python pre_train.py --comment "_default"
