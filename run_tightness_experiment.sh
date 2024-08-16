#!/bin/bash

# Usage: sbatch run_tightness_experiment.sh <n_episodes>
# Example: sbatch run_tightness_experiment.sh 20

#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=tightness
#SBATCH --partition=unkillable

# Check if n_episodes is provided as a command line argument
if [ $# -eq 0 ]; then
    echo "Please provide n_episodes as a command line argument"
    exit 1
fi

n_episodes=$1


module --quiet load anaconda/3
conda activate "$HOME/miniconda3/envs/cb"
python run_tightness_experiment.py --n_episodes=$n_episodes --device="cuda"
python make_tightness_plots.py