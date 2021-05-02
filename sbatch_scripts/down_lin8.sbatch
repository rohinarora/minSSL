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
python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "runs/May01_00-13-17_d1016_b128" --comment "_b128_cfg_linear_ssl"  -b 128 &
sleep 60
python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "runs/Apr30_12-01-38_d3099_b64" --comment "_b64_cfg_linear_ssl" -b 64




