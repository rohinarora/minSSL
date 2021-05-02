#!/bin/bash

#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --time=24:00:00
#SBATCH --job-name=down_label
#SBATCH --partition=ce-mri
#SBATCH --mem=100Gb
#SBATCH --output=xepoch.%j.out
#SBATCH --gres=gpu:a100:1

source activate simclr1

python downstream_eval.py --downstream_task fine_tune -tm SSL -frac 0.1 -rd "runs/Apr30_03-51-19_d3099_default" --comment "_default_cfg_label_0.1_SSL" &
sleep 60
python downstream_eval.py --downstream_task fine_tune -tm SSL -frac 0.25 -rd "runs/Apr30_03-51-19_d3099_default" --comment "_default_cfg_label_0.25_SSL" &
sleep 60
python downstream_eval.py --downstream_task fine_tune -tm SSL -frac 0.5 -rd "runs/Apr30_03-51-19_d3099_default" --comment "_default_cfg_label_0.5_SSL" &
sleep 60
python downstream_eval.py --downstream_task fine_tune -tm SSL -frac 0.75 -rd "runs/Apr30_03-51-19_d3099_default" --comment "_default_cfg_label_0.75_SSL"
