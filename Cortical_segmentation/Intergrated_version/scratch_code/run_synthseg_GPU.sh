#!/bin/bash

#SBATCH --job-name=synthseg_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00  # Adjusted job time, if needed
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --mem=20G  # Adjusted memory requirement


# Load the Anaconda module
#module load anaconda

ENV_NAME="synthseg_test2"
# Activate the environment
source activate ${ENV_NAME}

# Run your Python script
echo "Environment setup complete. Ready to run scripts."
echo "Running script..."

python /users/zxu/synthseg/SynthSeg/scripts/commands/SynthSeg_predict_ukb.py --i /users/zxu/synthseg/input --o /users/zxu/synthseg/output_GPU --parc --vol /users/zxu/synthseg/output_GPU/volumes.csv --resolutionconversion --keep_intermediate_files --relabel --label_correction --save_brain --save_analyseformat --qc /users/zxu/synthseg/output_GPU/qc.csv 