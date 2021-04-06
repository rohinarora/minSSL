#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --job-name=2048
#SBATCH --partition=multigpu
#SBATCH --mem=10Gb
#SBATCH --output=2048.%j.out
#SBATCH --exclusive
#SBATCH --gres=gpu:v100-sxm2:1

scenv simclr1
python run.py -b 2048
