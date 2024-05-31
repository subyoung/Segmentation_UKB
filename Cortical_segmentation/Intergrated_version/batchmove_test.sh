#!/bin/bash

#SBATCH --job-name=synthseg_cpu
#SBATCH --time=02:00:00  # Adjusted job time, if needed
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --cpus-per-task=8  # Number of CPU cores per task
#SBATCH --mem=32G  # Adjusted memory requirement

# Define paths
path="/dcs05/qiao/data/ukb/mri_brain/visit2_batch0"
testpath="/users/zxu/synthseg/batch_move_test/input/"
maskpath="/users/zxu/synthseg/batch_move_test/input/mask/"

# Create arrays to hold file paths
declare -a filelist
declare -a masklist

# Counter to limit the number of files processed
n=0
# Counter for number of files moved
t1_moved=0
mask_moved=0

# Record start time
start_time=$(date +%s)

# Traverse the directory tree
while IFS= read -r -d '' dir; do
  for file in "$dir"/*; do
    if [[ $file == *T1.nii.gz && $file != *._* ]]; then
      if [[ $(basename "$(dirname "$file")") == "T1" ]]; then
        filelabel=$(basename "$(dirname "$(dirname "$file")")")
        filelist+=("$filelabel|$file")
        maskfile="$dir/T1_brain_mask.nii.gz"
        if [[ -f $maskfile ]]; then
          masklist+=("$filelabel|$maskfile")
        else
          echo "No mask file found for $filelabel"
        fi
        n=$((n+1))
        #if [[ $n -ge 10 ]]; then
        #  break 2
        #fi
      fi
    fi
  done
done < <(find "$path" -type d -print0)

# Sort filelist
IFS=$'\n' sorted=($(sort <<<"${filelist[*]}"))
unset IFS

# Copy T1 files to the testpath directory
for item in "${sorted[@]}"; do
  filelabel="${item%%|*}"
  filepath="${item##*|}"
  cp "$filepath" "${testpath}${filelabel}_T1.nii.gz"
  t1_moved=$((t1_moved + 1))
  echo "${testpath}${filelabel}_T1.nii.gz"
done

# Copy mask files to the maskpath directory
for item in "${masklist[@]}"; do
  filelabel="${item%%|*}"
  maskfilepath="${item##*|}"
  cp "$maskfilepath" "${maskpath}${filelabel}_T1_brain_mask.nii.gz"
  mask_moved=$((mask_moved + 1))
  echo "${maskpath}${filelabel}_T1_brain_mask.nii.gz"
done

# Record end time
end_time=$(date +%s)

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"

# Print the number of T1 files moved
echo "Number of T1 files moved: $t1_moved"
