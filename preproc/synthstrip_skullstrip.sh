#!/bin/bash


# To make the mri_synthstrip work from freesurfer you need to follow this instructions:
# https://surfer.nmr.mgh.harvard.edu/fswiki//FS7_linux
# (to set up environment and variable)

cd $HOME
cd freesurfer
export FREESURFER_HOME=$HOME/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
# test freesurfer is correctly set up with
which freeview


BIDS_DIRECTORY=/mnt/projects/7TPD/bids/derivatives/hMRI_acu/mni_PD25_20170213_nifti

# Iterate through existing subjects in dataset
# for ((i=58;i<75;i+=1)); do
# SUBJ=$( printf '%03d' $i)
# SUB_ID="sub-${SUBJ}"
# echo $SUB_ID
# Select bias field corrected T1w image (m prefix)
INPUT_IMG="${BIDS_DIRECTORY}/ss_PD25-T1MPRAGE-template-1mm.nii"
test -f ${INPUT_IMG} && echo "File exists"

# mri_info ${INPUT_IMG}


MASK_IMG="${BIDS_DIRECTORY}/brain_mask_PD25-R2starmap-atlas-1mm.nii"
OUTPUT_IMG="${BIDS_DIRECTORY}/synthseg_ss_PD25-T1MPRAGE-template-1mm.nii"
# Generate brain mask and skull stripped image
mri_synthstrip -i ${INPUT_IMG} -o ${OUTPUT_IMG} -m ${MASK_IMG}

# Generate segmentations
# mri_synthseg --i ${INPUT_IMG} --o ${OUTPUT_IMG} --cpu

echo "------"
echo "Job finished"
