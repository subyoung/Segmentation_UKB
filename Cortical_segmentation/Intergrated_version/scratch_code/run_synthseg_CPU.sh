#!/bin/bash

#SBATCH --job-name=synthseg_cpu
#SBATCH --time=02:00:00  # Adjusted job time, if needed
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --cpus-per-task=8  # Number of CPU cores per task
#SBATCH --mem=32G  # Adjusted memory requirement


# Load the Anaconda module
module load anaconda

ENV_NAME="synthseg_38"
# Activate the environment
source activate ${ENV_NAME}

# Run your Python script
echo "Environment setup complete. Ready to run scripts."
echo "Running script..."

python /users/zxu/synthseg/SynthSeg/scripts/commands/SynthSeg_predict_ukb.py --i /users/zxu/synthseg/input --o /users/zxu/synthseg/output --parc --cpu --threads 30 --vol /users/zxu/synthseg/output/volumes.csv --resolutionconversion --keep_intermediate_files --relabel --label_correction --save_brain --save_analyseformat --qc /users/zxu/synthseg/output/qc.csv 