#!/bin/bash
#SBATCH --partition=HPC
#SBATCH --mem 64G
#SBATCH--cpus-per-task 64

# This script is to run the atlas propagation in the HPC cluster, you can either run it for a single subject or for a list of subjects
# To run it for a single subject you need to specify the subject ID in the line 40
# To run it for a list of subjects you need to specify the list of subjects as an array when running sbatch, for example:
# sbatch --array=1,3,5,7 pd25atlas_normalization.sh
# sbatch --array=1-24 pd25atlas_normalization.sh

source /mrhome/alejandrocu/anaconda3/etc/profile.d/conda.sh

conda activate 7tpd

# or move to the directory where the script is
cd /mrhome/alejandrocu/Documents/parkinson_classification

echo "Starting.."

sleep 5s

#SUBJ=$( printf '%03d' $SLURM_ARRAY_TASK_ID)
#SUB_ID="sub-${SUBJ}"

SUB_ID="sub-003"
echo $SUB_ID

python -u atlas_propagation.py ${SUB_ID}
# done

