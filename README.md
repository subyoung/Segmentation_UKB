<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
# About The Project

This is the segmentation tool for UKB dataset. Cortical segmentation is mainly based on and modified from **[SynthSeg](https://github.com/BBillot/SynthSeg)** repository while method for subcortical segmentation is still under consideration(05/26/2024). SynthSeg the first deep learning tool for segmentation of brain scans of any contrast and resolution and details can found in **[SynthSeg](https://github.com/BBillot/SynthSeg)** repository and their publication below.

**Robust machine learning segmentation for large-scale analysisof heterogeneous clinical brain MRI datasets** \
B. Billot, M. Colin, Y. Cheng, S.E. Arnold, S. Das, J.E. Iglesias \
PNAS (2023) \
[ [article](https://www.pnas.org/doi/full/10.1073/pnas.2216399120#bibliography) | [arxiv](https://arxiv.org/abs/2203.01969) | [bibtex](bibtex.bib) ]





<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CORTICAL SEGMENTATION -->
# Cortical Segmentation

* For cortical segmentation, the Synthseg tool has been modified to apply on UKB dataset. Additional resolution conversion, relabeling and leakage correction method are added. Scripts for both integrated version and separated version are both included in this repository.
  
<!-- GETTING STARTED -->
## Getting Started in Jhpce server

### Prerequisites

To use the tool on Jhpce server, make sure you have the Jhpce account and have the access to the data. 

### Installation

_Below is the instruction how you can set up the environment and install the tool in Jhpce server. The instruction is based on integrated version_

1. Clone the repo in your local machine
   ```sh
   git clone https://github.com/subyoung/Segmentation_UKB.git
   ```
2. Go to this [onedrve link](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/zxu105_jh_edu/Eq_ZuJaWex5CqsEUg7sGdakBw7OpAGAmWjCUC-hEhpAyQg?e=Sg8ZCH), and download the missing models. Then simply copy all models in h5 format to [./Cortical_segmentation/Intergrated_version/SynthSeg/models](https://github.com/subyoung/Segmentation_UKB/tree/a6ccbe48bf78b762a3b8d476ce30028e2900ae64/Cortical_segmentation/Intergrated_version/SynthSeg/models). 

3. Upload [the whloe cortical segmentation folder](https://github.com/subyoung/Segmentation_UKB/tree/a6ccbe48bf78b762a3b8d476ce30028e2900ae64/Cortical_segmentation/Intergrated_version) into jhpce server, e.g., using node jhpce-transfer01.jhsph.edu. Recommand using transfer01 node to upload/download for faster speed. See [Jhpce File-transfer Instruction](https://jhpce.jhu.edu/access/file-transfer/) for the instruction.
   
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
   If it returns the instruction of *conda*, you can go to step 5. Otherwise, load *conda* module first:
    ```sh
   module load anaconda
   ```
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
11. (optional) Test with using *--help* command of our script
  ```sh
    python /path-to-your-SynthSeg-folder/SynthSeg/scripts/commands/SynthSeg_predict_ukb.py --help
    #e.g., python /users/zxu/synthseg/SynthSeg/scripts/commands/SynthSeg_predict_ukb.py --help
  ```
  It should return the instruction of our script.
12. Run the script on the testfile using command line or the shell script *run_synthseg_GPU_test.sh*. No matter which way you choose, please adjust the file path as your path of testfile and SynthSeg command. \ 
  *If using command line:*
  ```sh
    python /users/zxu/synthseg/SynthSeg/scripts/commands/SynthSeg_predict_ukb.py --i /users/zxu/synthseg/test/input --o /users/zxu/synthseg/test/output --parc --vol /users/zxu/synthseg/test/output/volumes.csv --resolutionconversion --keep_intermediate_files --relabel --label_correction --save_brain --save_analyseformat --qc /users/zxu/synthseg/test/output/qc.csv
  ```
   *If using shell script:*
   ```sh
    #remember cd to the folder of the shell script first or add full path of the script, e.g., /users/zxu/synthseg/run_synthseg_GPU_test.sh
    sbatch run_synthseg_GPU_test.sh
   ```
13. If step 12 runs without error, you're now ready to use SynthSeg in Jhpce ! :tada: \
    Remember to activate your environment every time you need to run the scripts:
    ```sh
    conda activate synthseg_38
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started in local machine
To be done

<!-- USAGE EXAMPLES -->
## Usage

### Try it in one command !

Once all the python packages are installed (see below), you can simply test SynthSeg on your own data with:
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


**IMPORTANT:** SynthSeg always give results at 1mm isotropic resolution, regardless of the input. However, this can 
cause some viewers to not correctly overlay segmentations on their corresponding images. In this case, you can use the
`--resample` flag to obtain a resampled image that lives in the same space as the segmentation, such that they can be 
visualised together with any viewer. For UKB dataset, the intergrated correction flags can be used to get results in original resolution.

The complete list of segmented structures is available in [ROI label.xlsx](https://github.com/subyoung/Segmentation_UKB/blob/a6ccbe48bf78b762a3b8d476ce30028e2900ae64/Cortical_segmentation/ROI%20label.xlsx) along with their
corresponding values before and after the relabelling. This table also details the order in which the posteriors maps are sorted.

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

## Getting Started in Jhpce server
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


