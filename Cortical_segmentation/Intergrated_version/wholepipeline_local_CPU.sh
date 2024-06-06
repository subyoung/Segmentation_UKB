#!/bin/bash

#SBATCH --job-name=synthseg_wholepipeline_local_CPU
#SBATCH --time=10:00:00  # Adjusted job time, if needed
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --cpus-per-task=8  # Number of CPU cores per task
#SBATCH --mem=40G  # Adjusted memory requirement

#this script allow you to process and save the segmentation on original folder of T1 files 
#and save the qc metrics of all files in one csv. 

# Record start time
start_time=$(date +%s)

ENV_NAME="synthseg_38"
# Activate the environment
source activate ${ENV_NAME}

# Define paths
path="/users/zxu/ukb_data/visit2_batch1"
qc_file="$path/qc.csv"  #the path of qc file
# testpath="/users/zxu/synthseg/batch_move_test/input2/"
# maskpath="/users/zxu/synthseg/batch_move_test/input2/mask/"

# Flag to determine if filtering by filelabel is needed
filter_flag=0  # Set to 1 to filter by certain names, 0 to process all
# List of filelabels to process
declare -a allowed_filelabels=("1000835_2" "1002136_2" "1002513_2" "1003316_2" "1003323_2" "1003780_2" "1004174_2" "1004671_2" "1005569_2" "1006733_2" "1009281_2" "1011368_2" "1011775_2" "1012850_2" "1012865_2" "1014743_2" "1017157_2" "1018227_2" "1018833_2" "1019200_2" "1019305_2" "1019722_2" "1020919_2") # Replace with your desired labels
# set a batch size limit(how many files included, set 1000 should be able to include all)
batchsize=1000 #set 1000 to filter by number, other number like 10 to limit the size under 10

# Create arrays to hold file paths
declare -a filelist
declare -a masklist

# Counter to limit the number of files processed
n=0

# Output files for T1 file paths and corresponding output paths
input_file_path="input_file_paths.txt"
output_file_path="output_file_paths.txt"
qc_file_path="qc_file_paths.txt"
# Clear the output file if it exists
> "$input_file_path"
> "$output_file_path"
> "$qc_file_path"

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

# Print sorted file paths to the output file and create output paths
for item in "${sorted[@]}"; do
  filelabel="${item%%|*}"
  file="${item##*|}"
  echo "$file" >> "$input_file_path"
  output_file="${file%/*}/${filelabel}_T1_synthseg.nii.gz"
  echo "$output_file" >> "$output_file_path"
  # qc_file="${file%/*}/${filelabel}_T1_qc.csv" #if you want to put qc metric in separated csv, uncomment this line
  echo "$qc_file" >> "$qc_file_path"
done

echo ""
echo "$n files have been located"
echo "T1 file paths have been written to $input_file_path"
echo "Output file paths have been written to $output_file_path"
echo "QC metrics paths have been written to $qc_file"


# Run your Python script
echo ""
echo "Environment setup complete. Ready to run scripts."
echo "Running script..."

# remember to change the path as needed
python /users/zxu/synthseg/SynthSeg/scripts/commands/SynthSeg_predict_ukb.py --i /users/zxu/ukb_data/input_file_paths.txt --o /users/zxu/ukb_data/output_file_paths.txt --parc --robust --cpu --threads 30 --resolutionconversion --relabel --label_correction --qc /users/zxu/ukb_data/qc_file_paths.txt

# Record end time
end_time=$(date +%s)
# Calculate and print the execution time
execution_time=$((end_time - start_time))

# Convert execution time to hours, minutes, and seconds
hours=$((execution_time / 3600))
minutes=$(( (execution_time % 3600) / 60 ))
seconds=$((execution_time % 60))

echo ""
# printf "Data moving to intermediate folder execution time: %02d h %02d min %02d s\n" $hours1 $minutes1 $seconds1
printf "Total execution time: %02d h %02d min %02d s\n" $hours $minutes $seconds