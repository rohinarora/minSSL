#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --time=24:00:00
#SBATCH --job-name=batch2
#SBATCH --partition=ce-mri
#SBATCH --mem=100Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:a100:1


source activate simclr1
python pre_train.py -b 128 --comment "_b128"&
sleep 60
python pre_train.py -b 512 --comment "_b512"&
sleep 60
python pre_train.py -b 1024 --comment "_b1024"
