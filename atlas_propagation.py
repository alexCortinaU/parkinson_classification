# from PIL import Image
import pandas as pd
# import matplotlib.pyplot as plt
from pathlib import Path
# import torchio as tio
# import nibabel as nib
# import SimpleITK as sitk
# from scipy.stats import iqr
# from scipy.stats import f_oneway

from datetime import datetime
import os
import json
from time import time
import numpy as np
# import nipype.interfaces.ants as npants
import ants

this_path = Path(os.getcwd())

def main():
    # full pipe loop
    md_df = pd.read_csv('/mrhome/alejandrocu/Documents/parkinson_classification/bids_3t.csv')
    # md_df_hc = md_df[md_df['group'] == 0]
    # md_df_pd = md_df[md_df['group'] == 1]

    subj_ids = md_df['id'].values
    subj_ids = np.delete(subj_ids, np.where(subj_ids == 'sub-058'))

    # pd25template_path = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/mni_PD25_20170213_nifti/PD25-T1MPRAGE-template-1mm.nii')
    pd25template_path = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/mni_PD25_20170213_nifti/ss_PD25-T1MPRAGE-template-1mm.nii')

    pdlabels_path = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/mni_PD25_20170213_nifti/PD25-subcortical-1mm.nii')
    pdlabels_img = ants.image_read(str(pdlabels_path))

    exp_name = 'ssPD25_synquick_CC'
    print(f'exp_name: {exp_name}')
    transform_type = 'antsRegistrationSyNQuick[s]' # antsRegistrationSyNQuick[s], SyN, SyNRA
    syn_metric = 'CC' # CC, mattes
    
    for idx, subj in enumerate(subj_ids):
        # subj = 'sub-004'
        start = datetime.now()
        print(f'start {subj}')
        base_path = Path(f'/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI/{subj}/Results/brain_masked')
        r1_map_path = base_path / f'{subj}_ses-01prisma3t_echo-01_part-magnitude-acq-MToff_MPM_R1_w.nii'
        fi = ants.image_read(str(pd25template_path))
        mi = ants.image_read(str(r1_map_path))
        out_path = base_path.parent / 'antsreg'
        out_path.mkdir(parents=True, exist_ok=True)
        mytx = ants.registration(fixed=fi, moving=mi,
                                type_of_transform = transform_type, #'antsRegistrationSyNQuick[s]', SyNRA
                                outprefix= str(out_path / exp_name),
                                syn_metric = syn_metric
                                )
        print(f'finished registration subj {subj} in time: {datetime.now() - start}')
        save_dict = {'fwdtransforms': mytx['fwdtransforms'],
                    'invtransforms': mytx['invtransforms'],
                    'transform_type': transform_type,
                    'syn_metric': syn_metric,}
        with open(str(out_path / f'{exp_name}_outs.json'), 'w') as f:
            json.dump(save_dict, f)
        # save warped images
        start = datetime.now()
        # warped_image = ants.apply_transforms(fixed=fi, moving=mi, transformlist=mytx['fwdtransforms'])
        # ants.image_write(warped_image, str(out_path / f'{exp_name}_r1topd25_warped.nii'))
        # warped_image = ants.apply_transforms(fixed=mi, moving=fi, transformlist=mytx['invtransforms'])
        # ants.image_write(warped_image, str(out_path / f'{exp_name}_pd25tor1_warped.nii'))

        ants.image_write(mytx['warpedmovout'], str(out_path / f'{exp_name}_r1topd25_warped.nii'))
        ants.image_write(mytx['warpedfixout'], str(out_path / f'{exp_name}_pd25tor1_warped.nii'))
        # warp atlas
        warped_atlas = ants.apply_transforms(fixed=mi, moving=pdlabels_img, transformlist=mytx['invtransforms'], interpolator='genericLabel')
        ants.image_write(warped_atlas, str(out_path / f'{exp_name}_pd25tor1_warped_atlas.nii'))
        print(f'finished warping subj {subj} in time: {datetime.now() - start}')
        # if idx > 1:
        #     break

if __name__ == '__main__':
    main()