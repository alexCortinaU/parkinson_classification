#!/bin/bash
#SBATCH --partition=HPC
#SBATCH --mem-per-cpu 20G
#SBATCH--cpus-per-task 6

# To make the mri_synthseg work from freesurfer you need to follow this instructions:
# https://surfer.nmr.mgh.harvard.edu/fswiki//FS7_linux
# (to set up environment and variable)

# test freesurfer is correctly set up with
which freeview

ROOT_DIR=/mrhome/alejandrocu/Documents/batch_jobs

# Iterate through existing subjects in dataset
# for ((i=3;i<4;i+=1)); do
# SUBJ=$( printf '%03d' $i)
# SUB_ID="sub-${SUBJ}"
# echo $SUB_ID
# Select bias field corrected T1w image (m prefix)

INPUT_DIR="${ROOT_DIR}/input_files.txt"

# INPUT_IMG="${INPUT_DIR}/${SUB_ID}_ses-01prisma3t_echo-01_part-magnitude-acq-MToff_MPM_MTsat_w.nii"
# test -f ${INPUT_IMG} && echo "File exists"
# MASK_IMG="${BIDS_DIRECTORY}/outputs/${SUB_ID}_brain_mask_mt1w.nii"

OUTPUT_DIR="${ROOT_DIR}/output_files.txt"

QC_PATH="${ROOT_DIR}/qc_files.txt"

# Generate segmentations
mri_synthseg --i ${INPUT_DIR} --o ${OUTPUT_DIR} --qc ${QC_PATH} --threads 6 --cpu
# done
echo "------"
echo "Job finished"
