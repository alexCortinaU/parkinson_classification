import pandas as pd
from pathlib import Path
import argparse

from datetime import datetime
import os
import json
from time import time
import numpy as np
import ants

this_path = Path(os.getcwd())

def reg(subj: str, 
        qmri_map: str, 
        pd25_temp: str, 
        transform_type: str = 'SyNRA', 
        syn_metric: str = 'CC', 
        aff_metric: str = 'GC', 
        reg_iterations: tuple = (100, 70, 50, 20), 
        shrink_factors: tuple = (8, 4 , 2, 1), 
        smooth_sigmas: tuple = (3, 2, 1, 0), 
        grad_step: float = 0.1
        ):
    
    # full pipe loop
    md_df = pd.read_csv('/mrhome/alejandrocu/Documents/parkinson_classification/bids_3t.csv')

    subj_ids = md_df['id'].values
    subj_ids = np.delete(subj_ids, np.where(subj_ids == 'sub-058'))

        
    # qmri_map = 'R2s'
    # pd25_temp = 'R2s'
    # transform_type = 'antsRegistrationSyNQuick[s]' # antsRegistrationSyNQuick[s], SyN, SyNRA
    # syn_metric = 'CC' # CC, mattes
    # aff_metric = 'GC'
    # reg_iterations = (100, 70, 50, 20)
    # shrink_factors = (8, 4 , 2, 1)
    # smooth_sigmas = (3, 2, 1, 0)
    # grad_step = 0.1
    exp_name = f'{pd25_temp}_{qmri_map}_{transform_type}_{syn_metric}' # {transform_type}
    print(f'exp_name: {exp_name}')

    if pd25_temp == 'T1':
        pd25template_path = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/mni_PD25_20170213_nifti/ss_PD25-T1MPRAGE-template-1mm.nii')
    elif pd25_temp == 'R2s':
        pd25template_path = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/mni_PD25_20170213_nifti/ss_PD25-R2starmap-atlas-1mm.nii') # ss_PD25-T1MPRAGE-template-1mm.nii

    pdlabels_path = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/mni_PD25_20170213_nifti/PD25-subcortical-1mm.nii')
    pdlabels_img = ants.image_read(str(pdlabels_path))

    start = datetime.now()
    print(f'start {subj}')
    base_path = Path(f'/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI/{subj}/Results/brain_masked')
    if qmri_map == 'R1':
        qmri_map_path = base_path / f'{subj}_ses-01prisma3t_echo-01_part-magnitude-acq-MToff_MPM_R1_w.nii'
    elif qmri_map == 'R2s':
        qmri_map_path = base_path / f'{subj}_ses-01prisma3t_echo-01_part-magnitude-acq-MToff_MPM_R2s_WLS1_w.nii'
    
    fi = ants.image_read(str(pd25template_path))
    mi = ants.image_read(str(qmri_map_path))
    out_path = base_path.parent / 'antsreg_greedy'
    out_path.mkdir(parents=True, exist_ok=True)
    mytx = ants.registration(fixed=fi, moving=mi,
                            type_of_transform = transform_type, #'antsRegistrationSyNQuick[s]', SyNRA
                            outprefix= str(out_path / exp_name),
                            syn_metric = syn_metric,
                            aff_metric = aff_metric,
                            reg_iterations = reg_iterations,
                            aff_shrink_factors = shrink_factors,
                            aff_smoothing_sigmas = smooth_sigmas,
                            grad_step = grad_step
                            )
    reg_time = datetime.now() - start
    print(f'finished registration subj {subj} in time: {reg_time}')
    save_dict = {'fwdtransforms': mytx['fwdtransforms'],
                'invtransforms': mytx['invtransforms'],
                'transform_type': transform_type,
                'syn_metric': syn_metric,
                'aff_metric': aff_metric,
                'reg_iterations': reg_iterations,
                'aff_shrink_factors': shrink_factors,
                'aff_smoothing_sigmas': smooth_sigmas,
                'grad_step': grad_step,
                'reg_time': str(reg_time)}
    with open(str(out_path / f'{exp_name}_outs.json'), 'w') as f:
        json.dump(save_dict, f, sort_keys=True, indent=4)

    # save warped images
    start = datetime.now()
    ants.image_write(mytx['warpedmovout'], str(out_path / f'{exp_name}_{qmri_map}topd25{pd25_temp}_warped.nii'))
    ants.image_write(mytx['warpedfixout'], str(out_path / f'{exp_name}_pd25{pd25_temp}to{qmri_map}_warped.nii'))
    # warp atlas
    warped_atlas = ants.apply_transforms(fixed=mi, moving=pdlabels_img, transformlist=mytx['invtransforms'], interpolator='genericLabel')
    ants.image_write(warped_atlas, str(out_path / f'{exp_name}_pd25{pd25_temp}to{qmri_map}_warped_atlas.nii'))

    return reg_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run registration for subject')
    parser.add_argument("subj", help="subject id")
    parser.add_argument("atlas", help="Type of atlas template to use from PD25, ['T1', 'R2s']",  nargs='?', default='T1')
    parser.add_argument("moving", help="Type of moving image to use, ['R1', 'R2s']", nargs='?', default='R1')
    args = parser.parse_args()

    print(f'running registration for {args.subj}')
    ex_time = reg(args.subj, args.moving, args.atlas)
    print(f'finished registration for {args.subj} in time: {ex_time}')