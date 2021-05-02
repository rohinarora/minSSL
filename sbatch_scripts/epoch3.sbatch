#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 1
#SBATCH --time=24:00:00
#SBATCH --job-name=epoch3
#SBATCH --partition=ce-mri
#SBATCH --mem=30Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:a100:1

source activate simclr1
python pre_train.py -e 400 --comment "_e400"
