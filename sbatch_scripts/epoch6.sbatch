#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --time=24:00:00
#SBATCH --job-name=epoch6
#SBATCH --partition=ce-mri
#SBATCH --mem=100Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:a100:1

source activate simclr1
python pre_train.py -e 10 --comment "_e10" &
sleep 60
python pre_train.py -e 30 --comment "_e30" &
sleep 60
python pre_train.py -e 90 --comment "_e90" &
sleep 60
python pre_train.py -e 150 --comment "_e150" 
