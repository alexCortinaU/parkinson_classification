# Pre-processing scripts

This is a compilation of different scripts that were used for various tasks:

- **qMRI dataset creation.** MATLAB script to run SPM's hMRI toolbox. Comprises the creation of folders and unzipping files, perform tissue segmentation and bias field correction, hMRI's AutoReorient, and the final qMRI maps creation (after previously tuning the parameters). For this you would need to have SPM and the hMRI toolbox installed, and to set the paths correctly.

- **Freesurfer tools.** Bash scripts to run Free Surfer (previously I had downloaded the latest version that has the required DL tools) synthseg and synthstrip. Note: synthstrip worked so much better than SPM's tissue segmentation.

- **Python notebooks with miscellaneous tasks.** Including reading data, windowing (intensity clipping) the maps, skull stripping (with SPM), and some basic statistics.

- **Spatial normalization (registration) for MNI PD25 atlas propagation.** The bash script to be able to divide and conquer in SLURM (or just run a single subject), calls the python script atlas_propagation.py that uses ANTS to perform configurable registration (default is SyNRA).