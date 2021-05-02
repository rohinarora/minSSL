#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --time=24:00:00
#SBATCH --job-name=dl4
#SBATCH --partition=ce-mri
#SBATCH --mem=60Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:a100:1

source activate simclr1
python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "runs/May01_06-13-44_d1010_default_benchmark" --comment "_default_benchmark_cfg_linear_ssl" -e 1000 &
sleep 60
python downstream_eval.py --downstream_task linear_eval -tm SSL -rd "runs/May01_06-18-44_d1022_default_fp_benchmark" --comment "_fp_benchmark_cfg_linear_ssl" -e 1000