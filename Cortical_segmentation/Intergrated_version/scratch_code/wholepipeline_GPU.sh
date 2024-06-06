#!/bin/bash

#SBATCH --job-name=synthseg_wholepipeline_GPU
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00  # Adjusted job time, if needed
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --mem=80G  # Adjusted memory requirement

# Record start time
start_time=$(date +%s)

ENV_NAME="synthseg_38"
# Activate the environment
source activate ${ENV_NAME}

# Define paths
path="/dcs05/qiao/data/ukb/mri_brain/visit2_batch0"
testpath="/users/zxu/synthseg/batch_move_test/input2/"
maskpath="/users/zxu/synthseg/batch_move_test/input2/mask/"

# Flag to determine if filtering by filelabel is needed
filter_flag=0  # Set to 1 to filter by certain names, 0 to process all
# List of filelabels to process
declare -a allowed_filelabels=("1006733_2" "1009281_2" "1011775_2" "1017157_2" "1000512_2") # Replace with your desired labels
# set a batch size limit(how many files included, set 1000 should be able to include all)
batchsize=1000 #set 1000 to filter by number, other number like 10 to limit the size under 10

# Create arrays to hold file paths
declare -a filelist
declare -a masklist

# Counter to limit the number of files processed
n=0
# Counter for number of files moved
t1_moved=0
mask_moved=0

# Function to check if a value is in an array
function contains() {
  local e
  for e in "${@:2}"; do
    [[ "$e" == "$1" ]] && return 0
  done
  return 1
}


# Traverse the directory tree
echo "Start searching for files..."
while IFS= read -r -d '' dir; do
  for file in "$dir"/*; do
    if [[ $file == *T1.nii.gz && $file != *._* ]]; then
      if [[ $(basename "$(dirname "$file")") == "T1" ]]; then
        filelabel=$(basename "$(dirname "$(dirname "$file")")")
        if [[ $filter_flag -eq 1 ]]; then
          if contains "$filelabel" "${allowed_filelabels[@]}"; then
            filelist+=("$filelabel|$file")
            maskfile="$dir/T1_brain_mask.nii.gz"
            if [[ -f $maskfile ]]; then
              masklist+=("$filelabel|$maskfile")
            else
              echo "No mask file found for $filelabel"
            fi
            n=$((n+1))
            if [[ $n -ge $batchsize ]]; then
              break 2
            fi
          fi
        else
          filelist+=("$filelabel|$file")
          maskfile="$dir/T1_brain_mask.nii.gz"
          if [[ -f $maskfile ]]; then
            masklist+=("$filelabel|$maskfile")
          else
            echo "No mask file found for $filelabel"
          fi
          n=$((n+1))
          if [[ $n -ge $batchsize ]]; then
            break 2
          fi
        fi
      fi
    fi
  done
done < <(find "$path" -type d -print0)

# Sort filelist
IFS=$'\n' sorted=($(sort <<<"${filelist[*]}"))
unset IFS

# Check if the intermediate folder exists, if not, create it
if [ ! -d "$testpath" ]; then
  mkdir -p "$testpath"
  echo "Intermediate folder created at ${testpath}"
fi


# Copy T1 files to the testpath directory
echo "Start copying T1 files to intermediate folder..."
for item in "${sorted[@]}"; do
  filelabel="${item%%|*}"
  filepath="${item##*|}"
  cp "$filepath" "${testpath}${filelabel}_T1.nii.gz"
  t1_moved=$((t1_moved + 1))
  echo "${testpath}${filelabel}_T1.nii.gz"
done

# Check if the mask folder exists, if not, create it
if [ ! -d "$maskpath" ]; then
  mkdir -p "$maskpath"
  echo "Mask folder created at ${maskpath}"
fi

# Copy mask files to the maskpath directory
echo "Start copying mask files to intermediate folder..."
for item in "${masklist[@]}"; do
  filelabel="${item%%|*}"
  maskfilepath="${item##*|}"
  cp "$maskfilepath" "${maskpath}${filelabel}_T1_brain_mask.nii.gz"
  mask_moved=$((mask_moved + 1))
  echo "${maskpath}${filelabel}_T1_brain_mask.nii.gz"
done

# Record end time
end_time1=$(date +%s)

# Calculate and print the execution time
execution_time1=$((end_time1 - start_time))
# Convert execution time to hours, minutes, and seconds
hours1=$((execution_time1 / 3600))
minutes1=$(( (execution_time1 % 3600) / 60 ))
seconds1=$((execution_time1 % 60))
# Print the number of files moved
echo "Number of T1 files moved: $t1_moved"
echo "Number of mask files moved: $mask_moved"
printf "Data moving to intermediate folder execution time: %02d h %02d min %02d s\n" $hours1 $minutes1 $seconds1

# Run your Python script
echo "Environment setup complete. Ready to run scripts."
echo "Running script..."

python /users/zxu/synthseg/SynthSeg/scripts/commands/SynthSeg_predict_ukb.py --i /users/zxu/synthseg/batch_move_test/input2 --o /users/zxu/synthseg/batch_move_test/output_GPU --parc --robust --resolutionconversion --relabel --label_correction --qc /users/zxu/synthseg/batch_move_test/output_GPU/qc.csv 

# Record end time
end_time=$(date +%s)
# Calculate and print the execution time
execution_time=$((end_time - start_time))

# Convert execution time to hours, minutes, and seconds
hours=$((execution_time / 3600))
minutes=$(( (execution_time % 3600) / 60 ))
seconds=$((execution_time % 60))

echo ""
printf "Data moving to intermediate folder execution time: %02d h %02d min %02d s\n" $hours1 $minutes1 $seconds1
printf "Total execution time: %02d h %02d min %02d s\n" $hours $minutes $seconds