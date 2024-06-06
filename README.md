<a name="readme-top"></a>
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#cortical-segmentation">Cortical Segmentation</a>
      <ul>
        <li><a href="#getting-started-in-jhpce-server">Getting Started in JHPCE server</a>
          <ul>
            <li><a href="#prerequisites">Prerequisites</a></li>
            <li><a href="#installation">Installation</a></li>
            <li><a href="#method-1">Method 1</a></li>
            <li><a href="#method-2">Method 2</a></li>
          </ul>
        </li>
        <li><a href="#getting-started-in-local-machine">Getting Started in Local Machine</a></li>
        <li><a href="#usage">Usage</a>
          <ul>
            <li><a href="#try-it-in-one-command">Try it in one command</a></li>
            <li><a href="#try-it-using-sbatch">Try it using sbatch</a></li>
            <li><a href="#for-ukb-dataset">For UKB dataset</a>
              <ul>
                <li><a href="#the-current-pipeline">The Current Pipeline</a></li>
                <li><a href="#setting-flags">Setting Flags</a></li>
                <li><a href="#recommended-usage">Recommended Usage</a></li>
                <li><a href="#execution-efficiency">Execution Efficiency</a></li>
              </ul>
            </li>
          </ul>
        </li>
        <li><a href="#other-useful-commands-of-jhpce">Other useful commands of JHPCE</a></li>
        <li><a href="#roadmap">Roadmap</a></li>
        <li><a href="#contact">Contact</a></li>
        <li><a href="#acknowledgments">Acknowledgments</a></li>
      </ul>
    </li>
    <li><a href="#subcortical-segmentation">Subcortical Segmentation</a></li>
  </ol>
</details>





<!-- ABOUT THE PROJECT -->
# About The Project

This is the segmentation tool for UKB dataset. Cortical segmentation is mainly based on and modified from **[SynthSeg](https://github.com/BBillot/SynthSeg)** repository while method for subcortical segmentation is still under consideration(05/26/2024). SynthSeg is the first deep learning tool for segmentation of brain scans of any contrast and resolution and details can found in **[SynthSeg](https://github.com/BBillot/SynthSeg)** repository and their publication below.

**Robust machine learning segmentation for large-scale analysisof heterogeneous clinical brain MRI datasets** \
B. Billot, M. Colin, Y. Cheng, S.E. Arnold, S. Das, J.E. Iglesias \
PNAS (2023) \
[ [article](https://www.pnas.org/doi/full/10.1073/pnas.2216399120#bibliography) | [arxiv](https://arxiv.org/abs/2203.01969) | [bibtex](bibtex.bib) ]



<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CORTICAL SEGMENTATION -->
# Cortical Segmentation

* For cortical segmentation, the Synthseg tool has been modified to apply on UKB dataset. Additional resolution conversion, relabeling and leakage correction method are added. Scripts for both integrated version and separated version are both included in this repository.
* Example result with CSF
  ![1716979862674](https://github.com/subyoung/Segmentation_UKB/assets/51379801/1b297caf-4971-4d5c-b626-97f75525bdd2)

  
<!-- GETTING STARTED -->
## Getting Started in JHPCE server

<!-- PREREQUISTIES -->
### Prerequisites

To use the tool on JHPCE server, make sure you have the JHPCE account and have the access to the data. 

<!-- INSTALLATION -->
### Installation

_Below is the instruction how you can set up the environment and install the tool in JHPCE server. The instruction is based on integrated version_

#### Method 1
1. Clone the repo in your local machine
   ```sh
   git clone https://github.com/subyoung/Segmentation_UKB.git
   ```
2. Go to this [onedrve link](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/zxu105_jh_edu/Eq_ZuJaWex5CqsEUg7sGdakBw7OpAGAmWjCUC-hEhpAyQg?e=Sg8ZCH), and download the missing models. Then simply copy all models in h5 format to [Cortical_segmentation/Intergrated_version/SynthSeg/models](https://github.com/subyoung/Segmentation_UKB/tree/87c02e6e02de794496ff813b0b51f20ad55dcd25/Cortical_segmentation/Intergrated_version/SynthSeg/models). 

3. Upload [the whloe cortical segmentation folder](https://github.com/subyoung/Segmentation_UKB/tree/87c02e6e02de794496ff813b0b51f20ad55dcd25/Cortical_segmentation/Intergrated_version) into JHPCE server, e.g., using node jhpce-transfer01.jhsph.edu. Recommand using transfer01 node to upload/download for faster speed. See [JHPCE File-transfer Instruction](https://jhpce.jhu.edu/access/file-transfer/) for the instruction.
   
4. Open a cpu session or gpu session for the setup can be done sucessfully.
   ```sh
   #open a gpu session
   srun --pty --x11 --partition gpu --gpus=1 --mem=20G bash
   #or a cpu session
   srun --time=08:00:00 --mem=10G --cpus-per-task=4 --pty bash
   ```
5. Check whether the *conda* module has been activate. Try:
   ```sh
   conda
   ```
   If it returns the instruction of *conda*, you can go to step 6. Otherwise, load *conda* module first:
    ```sh
   module load anaconda
   ```
6. Using [synthseg_38_environment.yml](https://github.com/subyoung/Segmentation_UKB/blob/87c02e6e02de794496ff813b0b51f20ad55dcd25/Cortical_segmentation/Intergrated_version/setup/synthseg_38_environment.yml) to create an environment and download all the packages. \
   Redirect to the folder with synthseg_38_environment.yml in the command line, for example:
   ```sh
   cd /users/zxu/Cortical_segmentation/Intergrated_version
   ```
   Create the environment. You can change the environment name by adjusting the first line `name: synthseg_38` of the yml file if you want.
   ```sh
   conda env create -f synthseg_38_environment.yml
   ```
7. Activate the new environment
   ```sh
   #remember to adjust the env name if you changed the name before
   conda activate synthseg_38
   ```
8. If you are using GPU, you can test if tensorflow has been installed correctly. All dynamic libraries should be loaded sucessfully, and the visible GPU devices should be added.
   ```sh
   python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   Otherwise, if you are using CPU, you can test with the tensorflow version. However, using CPU to run tensorflow sucessfully does not grant using GPU successfully. The best way to test is using a GPU session.
   ```sh
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```
9. (optional) Test with using *--help* command of our script
    ```sh
      python /path-to-your-SynthSeg-folder/SynthSeg/scripts/commands/SynthSeg_predict_ukb.py --help
      #e.g., python /users/zxu/synthseg/SynthSeg/scripts/commands/SynthSeg_predict_ukb.py --help
    ```
    It should return the instruction of our script.
10. Run the script on the testfile using command line or the shell script *run_synthseg_GPU_test.sh*. No matter which way you choose, please adjust the file path as your path of testfile and SynthSeg command\
    *If using command line:*
     ```sh
     #using GPU
     python /path-to-your-SynthSeg-folder/SynthSeg/scripts/commands/SynthSeg_predict_ukb.py --i /users/zxu/synthseg/test/input --o /users/zxu/synthseg/test/output --parc --robust --vol /users/zxu/synthseg/test/output/volumes.csv --resolutionconversion --keep_intermediate_files --relabel --label_correction --save_brain --save_analyseformat --qc /users/zxu/synthseg/test/output/qc.csv
     #using CPU
     python /path-to-your-SynthSeg-folder/SynthSeg/scripts/commands/SynthSeg_predict_ukb.py --i /users/zxu/synthseg/test/input --o /users/zxu/synthseg/test/output --parc --robust --cpu --threads 30 --vol /users/zxu/synthseg/test/output/volumes.csv --resolutionconversion --keep_intermediate_files --relabel --label_correction --save_brain --save_analyseformat --qc /users/zxu/synthseg/test/output/qc.csv
     ```
     *If using shell script:*
     ```sh
      #remember cd to the folder of the shell script first or add full path of the script, e.g., /users/zxu/synthseg/run_synthseg_GPU_test.sh
      #remember to adjust the env name if you used other name instead of synthseg_38
      #remember to adjust the file path as your path of testfile in the shell script
      sbatch run_synthseg_GPU_test.sh
     ```
11.  If step 10 runs without error, you're now ready to use SynthSeg in JHPCE ! :tada: \
     Remember to activate your environment every time you need to run the scripts:
     ```sh
     conda activate synthseg_38
     ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>
   
   

#### Method 2 
**(Currently the method 2 has some issues with tensorflow, pls use method 1 for now)** \
1-5. Same with Method 1.  \
6. Create an environment to install the basic packages.
   ```sh
   # Conda, Python 3.8:
   conda create -n synthseg_38
   ```
   Proceed with *y* when it sloved the environment and asked your permisson to proceed.
   ```sh
   Proceed ([y]/n)? y
   ```
7. Activate the created environment
   ```
   conda activate synthseg_38
   ```
   If the environment switched to syntheg_38 successfully, then go to step 7.
8. Further install other dependencies using pip
   ```sh
   pip install tensorflow-gpu==2.2.0 keras==2.3.1 protobuf==3.20.3 numpy==1.23.5 nibabel==5.0.1 matplotlib==3.6.2
   ```
9. Test the installation of tensorflow. If the following tests are all passed, you can go to step 11\
   *Test whether tensorflow is installed correctly.*
   ```sh
   conda list tensorflow
   ```
   It should return you the path of tensorflow:
   ```sh
   # packages in environment at /users/zxu/.conda/envs/synthseg_test:
    #
    # Name                    Version                   Build  Channel
    tensorflow-estimator      2.2.0                    pypi_0    pypi
    tensorflow-gpu            2.2.0                    pypi_0    pypi
   ```
   *Test whether tensorflow can be called*
   ```sh
    python -c "import tensorflow as tf; print(tf.__version__)"
   ```
   It should return you the version of tensorflow:
   ```sh
    2.2.0
   ```
10. Redirect to the SynthSeg folder with setup.py in the command line, e.g., if the folder is */users/zxu/Intergrated_version/SynthSeg*
     ```sh
     cd /users/zxu/Intergrated_version/SynthSeg
     ```
    Under the *SynthSeg* folder, with and activated environment, further install all the dependencies.
     ```sh
     python setup.py install
     ```
     After the installation, go back to step 9 to test.
11. Test with step 8-10 in Method 1.
12. If all test runs without error, you're now ready to use SynthSeg in JHPCE ! :tada: \
     Remember to activate your environment every time you need to run the scripts:
     ```sh
     conda activate synthseg_38
     ```



<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Getting Started in local machine
To be done
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage
**IMPORTANT:** \
For current version of this tool, you can either put all input files under one folder for batch processing like this:
```
INPUR_FOLDER/
  ├ A.nii.gz
  ├ B.nii.gz
  ├ *.nii.gz
  ├ /mask
    ├ A_mask.nii.gz
    ├ B_mask.nii.gz
    ├ *_mask.nii.gz  
```
or use a ```.txt``` file where each line is the path of an image to segment, like this:
```
path-to-file-1
path-to-file-2
path-to-file-3
path-to-file-4
...
```
Noted if you use ```.txt``` for input, the output path should also be put into ```.txt``` file to load.

### Try it in one command

Once all the python packages are installed (see above), you can simply test SynthSeg on your own data with:
```
python ./scripts/commands/SynthSeg_predict_ukb.py --i <input> --o <output> [--parc --robust --ct --vol <vol> --qc <qc> --post <post> --resample <resample>] [--resolutionconversion] [--keep_intermediate_files] [--relabel] [--label_correction] [--save_analyseformat] [--save_brain]
```

where:
- `<input>` path to a scan to segment, or to a folder. This can also be the path to a text file, where each line is the
path of an image to segment.
- `<output>` path where the output segmentations will be saved. This must be the same type as `<input>` (i.e., the path 
to a file, a folder, or a text file where each line is the path to an output segmentation).
- `--parc` (optional) to perform cortical parcellation in addition to whole-brain segmentation.
- `--robust` (optional) to use the variant for increased robustness (e.g., when analysing clinical data with large space
spacing). This can be slower than the other model.
- `--ct` (optional) use on CT scans in Hounsfield scale. It clips intensities to [0, 80].
- `<vol>` (optional) path to a CSV file where the volumes (in mm<sup>3</sup>) of all segmented regions will be saved for all scans 
(e.g. /path/to/volumes.csv). If `<input>` is a text file, so must be `<vol>`, for which each line is the path to a 
different CSV file corresponding to one subject only.
- `<qc>` (optional) path to a CSV file where QC scores will be saved. The same formatting requirements as `<vol>` apply.
- `<post>` (optional) path where the posteriors, given as soft probability maps, will be saved (same formatting 
requirements as for `<output>`).
- `<resample>` (optional) SynthSeg segmentations are always given at 1mm isotropic resolution. Hence, 
images are always resampled internally to this resolution (except if they are already at 1mm resolution). 
Use this flag to save the resampled images (same formatting requirements as for `<output>`).

Integrated correction flags for UKB dataset:
- `--resolutionconversion` (optional) Convert the resolution of the input image to 1mm isotropic resolution before segmentation.(Updated 24/04/24 by Ziyang)
- `--keep_intermediate_files` (optional) Keep the intermediate files during the resolution conversion. Default is False to save space. (Updated 24/04/24 by Ziyang)
- `--relabel` (optional) Relabel the segmentation. (Updated 24/04/24 by Ziyang)
- `--label_correction` (optional) Correct the labels in the segmentation using the original mask. (Updated 22/05/24 by Ziyang)
- `--save_analyseformat` (optional) Save the segmentation in Analyse format. (Updated 24/04/24 by Ziyang)
- `--save_brain` (optional) Save the brain mask and brain generated by SynthSeg. If `--label_correction` is used, the mask and brain generated by SynthSeg may not be necessary, but they can still be saved. (Updated 22/05/24 by Ziyang)

Additional optional flags are also available:
- `--cpu`: (optional) to enforce the code to run on the CPU, even if a GPU is available.
- `--threads`: (optional) number of threads to be used by Tensorflow (default uses one core). Increase it to decrease 
the runtime when using the CPU version.
- `--crop`: (optional) to crop the input images to a given shape before segmentation. This must be divisible by 32.
Images are cropped around their centre, and their segmentations are given at the original size. It can be given as a 
single (i.e., `--crop 160`), or several integers (i.e, `--crop 160 128 192`, ordered in RAS coordinates). By default the
whole image is processed. Use this flag for faster analysis or to fit in your GPU.
- `--fast`: (optional) to disable some operations for faster prediction (twice as fast, but slightly less accurate). 
This doesn't apply when the --robust flag is used.
- `--v1`: (optional) to run the first version of SynthSeg (SynthSeg 1.0, updated 29/06/2022).


**IMPORTANT 1:** SynthSeg always give results at 1mm isotropic resolution, regardless of the input. However, this can 
cause some viewers to not correctly overlay segmentations on their corresponding images. In this case, you can use the
`--resample` flag to obtain a resampled image that lives in the same space as the segmentation, such that they can be 
visualised together with any viewer. \
**IMPORTANT 2:** For UKB dataset, the intergrated correction flags can be used to get results in original resolution.

The complete list of segmented structures is available in [ROI label.xlsx](https://github.com/subyoung/Segmentation_UKB/blob/87c02e6e02de794496ff813b0b51f20ad55dcd25/Cortical_segmentation/ROI%20label.xlsx) along with their
corresponding values before and after the relabelling. This table also details the order in which the posteriors maps are sorted.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Try it using sbatch

The best way to run this tool on JHPCE is using sbatch and shell script `sbatch run_synthseg_GPU.sh`. The example shell script using [CPU](https://github.com/subyoung/Segmentation_UKB/blob/87c02e6e02de794496ff813b0b51f20ad55dcd25/Cortical_segmentation/Intergrated_version/testfile/run_synthseg_CPU.sh) and [GPU](https://github.com/subyoung/Segmentation_UKB/blob/87c02e6e02de794496ff813b0b51f20ad55dcd25/Cortical_segmentation/Intergrated_version/testfile/run_synthseg_GPU.sh) has been provided. 
```sh
#!/bin/bash

#SBATCH --job-name=synthseg_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00  # Adjusted job time, if needed
#SBATCH --ntasks=1  # Number of tasks
#SBATCH --mem=20G  # Adjusted memory requirement


# Load the Anaconda module
#module load anaconda

ENV_NAME="synthseg_38"
# Activate the environment
source activate ${ENV_NAME}

# Run your Python script
echo "Environment setup complete. Ready to run scripts."
echo "Running script..."

#adjust the path as you need
python /users/zxu/synthseg/SynthSeg/scripts/commands/SynthSeg_predict_ukb.py --i /users/zxu/synthseg/input --o /users/zxu/synthseg/output_GPU --parc --robust --vol /users/zxu/synthseg/output_GPU/volumes.csv --resolutionconversion --keep_intermediate_files --relabel --label_correction --save_brain --save_analyseformat --qc /users/zxu/synthseg/output_GPU/qc.csv 
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### For UKB dataset 
#### The current pipeline
![1716975131352](https://github.com/subyoung/Segmentation_UKB/assets/51379801/e1f6076b-fe22-4802-87d3-c9f513ceaf00)
<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Setting flags
Therefore, we need to set at least `---parc --robust --resolutionconversion --relabel --label_correction` as True. \
`--cpu --threads 30` can be set true if you want to use CPU instead of GPU, and you can adjust the threads. \
`--keep_intermediate_files --save_analyseformat --save_brain` are optional, depending on whether you want the intermediate files and final results in analyse format.
![1716975199877](https://github.com/subyoung/Segmentation_UKB/assets/51379801/66423256-480b-468b-97ab-34d199f1c390)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Recommended usage
We recommand to process and save the output in original batch folder, and save the qc metrics of all files in one csv file. The current version of this tool using ```.txt``` to give inputs is based on the folder struction of ukb_data, which is ```batch_name/case_id/T1/file.nii.gz```, while other folder struction may not apply and cause naming issue. \
Below is the example shell script to search for the T1 files and mask files in order, then generated three ```.txt``` files for the modified SynthSeg to process: 1) a list of paths of T1 inputs 2) a list of paths of segmentation results 3) a list of paths of qc metrics(same path for one batch for ukb project). The command of modified SynthSeg will take these into the process and save all results in original folder of each T1 files. See [wholepipeline_local_CPU](https://github.com/subyoung/Segmentation_UKB/blob/17e4429f7544239fbff046e423622421bf4867e3/Cortical_segmentation/Intergrated_version/wholepipeline_local_CPU.sh) and [wholepipeline_local_GPU](https://github.com/subyoung/Segmentation_UKB/blob/17e4429f7544239fbff046e423622421bf4867e3/Cortical_segmentation/Intergrated_version/wholepipeline_local_GPU.sh) for details. Remember to adjust the env name and all paths in these scripts when you use it.
```sh
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
path="/users/zxu/ukb_data/visit2_batch0" #path of the batch
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
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

#### Execution efficiency
For batch 1, there are 189 cases in total.
![image](https://github.com/subyoung/Segmentation_UKB/assets/51379801/ab73df9e-63b5-4e7b-b62c-98c0964706ca)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Other useful commands of JHPCE
For complete instruction of JHPCE, please refer to [Joint HPC Exchange](https://jhpce.jhu.edu/).
1. Open a session
   ```sh
   #open a gpu session
   srun --pty --x11 --partition gpu --gpus=1 --mem=20G bash
   #or a cpu session
   srun --time=08:00:00 --mem=10G --cpus-per-task=4 --pty bash
   ```
2. Check current jobs
   ```sh
   squeue --me
   ```
3. About environments
   ```sh
   #List all environments
   conda env list
   #List all packages under the current environment
   conda list
   #Delete some environment, e.g.,synthseg_test
   conda env remove --name synthseg_test
4. Check cuda version(under GPU session)
   ```sh
   nvidia-smi
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

- [x] Add installation instruction in server
- [x] Add usage
- [ ] Add installation instruction in local machine
- [ ] Add examples

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- CONTACT -->
## Contact

Ziyang Xu - zxu105@jh.edu

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

[SynthSeg](https://github.com/BBillot/SynthSeg)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- SUBCORTICAL SEGMENTATION -->
# Subcortical Segmentation
* For subcortical segmentation...

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started in JHPCE server
### Prerequisites
### Installation
## Usage

<!-- ROADMAP -->
## Roadmap
<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- CONTACT -->
## Contact

Ziyang Xu - zxu105@jh.edu
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
# Acknowledgments

<p align="right">(<a href="#readme-top">back to top</a>)</p>


