import torch
import torch.nn.functional as F

from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torchio as tio
import nibabel as nib
import SimpleITK as sitk
from scipy.stats import iqr
from scipy.stats import f_oneway

import os
import json
import numpy as np
import copy
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import torchvision
import pytorch_lightning as pl
from torchvision import models
from torchvision import transforms

import yaml
# from dataset.hmri_dataset import HMRIDataModule, HMRIDataModuleDownstream
# from models.pl_model import Model, ContrastiveLearning, ModelDownstream

from dataset.hmri_dataset import HMRIDataModule, HMRIControlsDataModule, HMRIPDDataModule
from models.pl_model import Model, Model_AE, GenerateReconstructions, ComputeRE

from utils.utils import reconstruct
from utils.utils import save_nifti_from_array, crop_img
from utils.general_utils import save_sitk_from_nda
this_path = Path().resolve()

def mask_atlas(subject:str, group: str, save=True):
    anat_path = Path(f'/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI/{subject}/Results/Masks')
    # read brain mask
    brain_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(anat_path/f'{subject}_brain_mask_mtsat_w.nii')))

    x_min, x_max, y_min, y_max, z_min, z_max = crop_img(brain_mask, return_dims=True)

    # read atlas
    atlas = sitk.ReadImage(str(anat_path/f'inv_reoriented_{subject}_mT1w_{group}_pd25_PD25-subcortical-1mm_uint8.nii'))
    atlas_nda = sitk.GetArrayFromImage(atlas)
    # crop atlas
    atlas_nda_c = atlas_nda[x_min:x_max, y_min:y_max, z_min:z_max] 
    if save:
        save_sitk_from_nda(atlas_nda_c,
                           anat_path/f'{subject}_PD25-subcortical-1mm_cropped.nii',
                           atlas)
    else:
        return atlas_nda_c

def get_atlas_nda(subject, group):

    # crop PD25 atlas as brain mask and save
    mask_atlas(subject, group)
    og_path = Path(f"/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI/{subject}/Results/")
    
    # set atlas path
    pd25_path = og_path / f'Masks/{subject}_PD25-subcortical-1mm_cropped.nii'
    
    # create tio subject and preprocess
    preprocess = tio.Compose(
            [   tio.ToCanonical(),
                tio.CropOrPad(180, padding_mode='minimum')
            ]
        )
    atlas_tiosubj_o = tio.Subject(image=tio.ScalarImage(pd25_path))
    atlas_tiosubj_p = preprocess(atlas_tiosubj_o)
    atlas_tiosubj_nda = atlas_tiosubj_p['image'][tio.DATA].cpu().numpy()[0]

    return atlas_tiosubj_nda

def get_statistics_from_map(xai_map: np.ndarray, subject: str, group: str):
    # get atlas nda
    atlas_nda = get_atlas_nda(subject, group)
    # get unique values in atlas
    atlas_vals = np.unique(atlas_nda)
    stats = []
    for label in atlas_vals:
        map_stats = {}
        map_stats['label'] = label
        map_stats['group'] = group
        map_stats['subject'] = subject     
        masked_map = xai_map[atlas_nda == label]
        map_stats['mean'] = np.mean(masked_map)
        map_stats['std'] = np.std(masked_map)
        map_stats['median'] = np.median(masked_map)
        map_stats['max'] = np.max(masked_map)
        map_stats['min'] = np.min(masked_map)
        map_stats['iqr'] = iqr(masked_map)
        stats.append(map_stats)

    return pd.DataFrame(stats)

def get_re_map(subj_idx, model, data, ovlap=6, vae=False):
    hc_patches, hc_locations, hc_sampler, hc_subject = data.get_grid(subj=subj_idx, overlap=ovlap, mode='val')
    hc_data = [hc_patches, hc_locations, hc_sampler, hc_subject, 'sub_xx'] # last subj_id not relevant
    rec_error = reconstruct(hc_data, model, 
                            overlap_mode='hann', 
                            save_img=False,  
                            vae=vae)
    
    return rec_error, hc_subject

def main():
    # read model from checkpoint and set up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chkpt_path = Path('/mrhome/alejandrocu/Documents/parkinson_classification/p2_hmri_outs/hp_tuning/pd_hmri_lr0.001_ps128/version_0/checkpoints/last.ckpt')

    # load config file
    exp_dir = chkpt_path.parent.parent.parent
    with open(exp_dir /'config_dump.yml', 'r') as f:
        exp_cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

    # set random seed for reproducibility
    pl.seed_everything(exp_cfg['dataset']['random_state'],  workers=True)

    model = Model_AE.load_from_checkpoint(chkpt_path, **exp_cfg['model'])
    model = model.to(device)
    model.eval()

    if 'vae' in exp_cfg['model']['net']:
        is_vae = True
    else:
        is_vae = False
    print(f'Is model vae: {is_vae}')
    # create datasets

    root_dir = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI')
    md_df = pd.read_csv(this_path/'bids_3t.csv')
    md_df_hc = md_df[md_df['group'] == 0]
    md_df_pd = md_df[md_df['group'] == 1]

    augmentations = tio.Compose([])
    data_hc = HMRIControlsDataModule(md_df=md_df_hc,
                            root_dir=root_dir,
                            augment=augmentations,
                            **exp_cfg['dataset'])
    data_hc.prepare_data()
    data_hc.setup()

    data_pd = HMRIPDDataModule(md_df=md_df_pd,
                                root_dir=root_dir,
                                augment=augmentations,
                                **exp_cfg['dataset'])
    data_pd.prepare_data()
    data_pd.setup()

    dfs = pd.DataFrame()
    for i in range(len(data_hc.md_df_val)):
        subject = data_hc.md_df_train.iloc[i]['id']
        re_map, subj_img = get_re_map(i, model, data_hc)
        print(re_map.shape)
        df = get_statistics_from_map(re_map, subject, 'HC')
        dfs = pd.concat([dfs, df], axis=0)
        break
    dfs.reset_index(drop=True, inplace=True)
    print(dfs)

    # save_nifti_from_array(subj_id=subject,
    #                           arr=re_map,
    #                           path=Path('/mrhome/alejandrocu/Documents/parkinson_classification/p2_hmri_outs/hp_tuning/pd_hmri_lr0.001_ps128') / f'test_{subject}_re.nii.gz')
    subj_img = subj_img['image'][tio.DATA][0].cpu().numpy()
    save_nifti_from_array(subj_id=subject,
                            arr=subj_img,
                            path=Path('/mrhome/alejandrocu/Documents/parkinson_classification/p2_hmri_outs/hp_tuning/pd_hmri_lr0.001_ps128') / f'test_{subject}_og_img.nii.gz')

if __name__ == '__main__':
    main()
    