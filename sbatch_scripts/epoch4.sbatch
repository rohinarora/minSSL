#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --time=24:00:00
#SBATCH --job-name=epoch4
#SBATCH --partition=ce-mri
#SBATCH --mem=50Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:a100:1

source activate simclr1
python pre_train.py -e 450 --comment "_e450"
