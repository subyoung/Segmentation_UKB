#!/bin/bash

#SBATCH --job-name=synthseg_gpu_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00  # Adjust job time as needed
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --mem=20G  # Adjust memory requirement as needed

# Load the Anaconda module
# module load anaconda

# Activate the Conda environment
# change the env name if it is different
source activate synthseg_38   

# Verify environment and TensorFlow installation
echo "Environment setup complete. Ready to run scripts."
conda list tensorflow

# Run your Python script
echo "Running script..."

#please adjust to your own path
python /users/zxu/synthseg/SynthSeg/scripts/commands/SynthSeg_predict_ukb.py --i /users/zxu/synthseg/test/input --o /users/zxu/synthseg/test/output --parc --vol /users/zxu/synthseg/test/output/volumes.csv --resolutionconversion --keep_intermediate_files --relabel --label_correction --save_brain --save_analyseformat --qc /users/zxu/synthseg/test/output/qc.csv
