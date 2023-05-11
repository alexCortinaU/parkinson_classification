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
from datetime import datetime
import yaml
# from dataset.hmri_dataset import HMRIDataModule, HMRIDataModuleDownstream
# from models.pl_model import Model, ContrastiveLearning, ModelDownstream

from dataset.hmri_dataset import HMRIDataModule, HMRIControlsDataModule, HMRIPDDataModule
from models.pl_model import Model, Model_AE, GenerateReconstructions, ComputeRE
from GenerativeModels.generative.networks.nets import VQVAE

from utils.utils import reconstruct
from utils.utils import save_nifti_from_array, crop_img
from utils.general_utils import save_sitk_from_nda
from tqdm import tqdm
this_path = Path().resolve()

CHKPT_PATHS = {
    'vqvae': {
        'MTsat': '/mrhome/alejandrocu/Documents/parkinson_classification/vqvae_models/normative_vqvae_run3_MTsat/normative_vqvae_run3_MTsat_vqvae_model.pt',
        'R1': '/mrhome/alejandrocu/Documents/parkinson_classification/vqvae_models/normative_vqvae_run3_R1/normative_vqvae_run3_R1_vqvae_model.pt',
        'R2s_WLS1': '/mrhome/alejandrocu/Documents/parkinson_classification/vqvae_models/normative_vqvae_run3_R2s_WLS1/normative_vqvae_run3_R2s_WLS1_vqvae_model.pt',
        'PD_R2scorr': '/mrhome/alejandrocu/Documents/parkinson_classification/vqvae_models/normative_vqvae_run3_PD_R2scorr/normative_vqvae_run3_PD_R2scorr_vqvae_model.pt'
            },
    'svae': {
        'MTsat': '/mrhome/alejandrocu/Documents/parkinson_classification/p2_hmri_outs/normative_svae_MTsat_gamma_0.95/version_0/checkpoints/last.ckpt',
        'R1': '/mrhome/alejandrocu/Documents/parkinson_classification/p2_hmri_outs/normative_svae_R1_gamma_0.95/version_0/checkpoints/last.ckpt',
        'R2s_WLS1': '/mrhome/alejandrocu/Documents/parkinson_classification/p2_hmri_outs/normative_svae_R2s_WLS1_gamma_0.95/version_0/checkpoints/last.ckpt',
        'PD_R2scorr': '/mrhome/alejandrocu/Documents/parkinson_classification/p2_hmri_outs/normative_svae_PD_R2scorr_gamma_0.95/version_0/checkpoints/last.ckpt'
            },
    'autoencoder': {
        'MTsat': '/mrhome/alejandrocu/Documents/parkinson_classification/p2_hmri_outs/normative_autoencoder_MTsat/version_0/checkpoints/last.ckpt',
        'R1': '/mrhome/alejandrocu/Documents/parkinson_classification/p2_hmri_outs/normative_autoencoder_R1/version_0/checkpoints/last.ckpt',
        'R2s_WLS1': '/mrhome/alejandrocu/Documents/parkinson_classification/p2_hmri_outs/normative_autoencoder_R2s_WLS1/version_0/checkpoints/last.ckpt',
        'PD_R2scorr': '/mrhome/alejandrocu/Documents/parkinson_classification/p2_hmri_outs/normative_autoencoder_PD_R2scorr/version_0/checkpoints/last.ckpt'
            }
    }   

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

def get_re_map(subj_idx, model, data, ovlap=6, ae_type='vae', error_type='mse', device='cuda'):
    hc_patches, hc_locations, hc_sampler, hc_subject = data.get_grid(subj=subj_idx, overlap=ovlap, mode='val')
    hc_data = [hc_patches.to(device), hc_locations.to(device), hc_sampler, hc_subject, 'sub_xx'] # last subj_id not relevant
    rec_error, rec_img = reconstruct(hc_data, model, 
                            overlap_mode='hann', 
                            save_img=False,  
                            ae_type=ae_type,
                            error_type=error_type)

    return rec_error, rec_img, hc_subject

def load_vqvae_model(chkpt_path, device, channels, latent_size):
    # load model
    model = VQVAE(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                num_res_layers=2,
                downsample_parameters=((2, 3, 1, 1), (2, 3, 1, 1)),
                upsample_parameters=((2, 3, 1, 1, 1), (2, 3, 1, 1, 1)),
                num_channels=channels, #(96, 96),
                num_res_channels=channels, #,
                num_embeddings=latent_size, # 256,
                embedding_dim=32,
                act='LEAKYRELU'
                )
    model.load_state_dict(torch.load(chkpt_path))
    return model

def generate_recons_from_chkpt(chkpt_path, error_types):
    # read model from checkpoint and set up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config file
    if 'vqvae' in chkpt_path.name:
        exp_dir = chkpt_path.parent
        with open(exp_dir /'config_dump.yml', 'r') as f:
            exp_cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]
    else:
        exp_dir = chkpt_path.parent.parent.parent
        with open(exp_dir /'config_dump.yml', 'r') as f:
            exp_cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]
    
    # set random seed for reproducibility
    pl.seed_everything(exp_cfg['dataset']['random_state'],  workers=True)

    # load model
    ae_type = exp_cfg['model']['net']
    if ae_type == 'autoencoder' or ae_type == 'svae':
        model = Model_AE.load_from_checkpoint(chkpt_path, **exp_cfg['model'])
    elif ae_type == 'vqvae':
        model = load_vqvae_model(chkpt_path, 
                                 device,
                                 channels=exp_cfg['model']['channels'],
                                 latent_size=exp_cfg['model']['latent_size'])
    else:
        raise ValueError('Wrong model type!')
    
    model = model.to(device)
    model.eval()

    # create datasets
    root_dir = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI')
    md_df = pd.read_csv(this_path/'bids_3t.csv')
    md_df_hc = md_df[md_df['group'] == 0]
    md_df_pd = md_df[md_df['group'] == 1]

    map_type = exp_cfg['dataset']['map_type']
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

    # obtain reconstruction and RE error maps
    progress_bar = tqdm(range(len(data_pd.md_df)), desc='Reconstructing', total=len(data_pd.md_df), ncols=110)
    for i in progress_bar:
    # for i in range(len(data_pd.md_df)):
        subject_idx = data_pd.md_df.iloc[i]['id']
        for p, error_type in enumerate(error_types):
            re_map, rec_img, subj_img = get_re_map(i, 
                                                model, 
                                                data_pd,
                                                ae_type=ae_type,
                                                error_type=error_type,
                                                device=device)
            save_path = Path('/mrhome/alejandrocu/Documents/parkinson_classification/reconstructions')
            save_path = save_path / f'{subject_idx}/{map_type[0]}/{ae_type}'
            save_path.mkdir(parents=True, exist_ok=True)
            
            # save images
            save_nifti_from_array(arr=re_map.cpu().numpy(),
                                subj_id=subject_idx,
                                path=save_path/f'{subject_idx}_{error_type}_re_map.nii',
                                affine_matrix=subj_img['image']['affine'],
                                header=None)
            if p == 0:
                save_nifti_from_array(arr=rec_img[0].cpu().numpy(),
                                    subj_id=subject_idx,
                                    path=save_path/f'{subject_idx}_rec_img.nii',
                                    affine_matrix=subj_img['image']['affine'],
                                    header=None)
        save_dict = {'chkpt_path': str(chkpt_path),
                        'error_types': str(error_types)}
        with open(str(save_path / f'reconstruction_outs.json'), 'w') as f:
            json.dump(save_dict, f)
        # break

def main():
    init_time = datetime.now()
    # chkpt_path = Path('/mrhome/alejandrocu/Documents/parkinson_classification/vqvae_models/normative_vqvae_run3_MTsat/normative_vqvae_run3_MTsat_vqvae_model.pt')
    for ae_type, chkpt_dict in CHKPT_PATHS.items():
        print(f'Running {ae_type}...')
        for map_type, chkpt_path in chkpt_dict.items():
            print(f'Running {map_type}...')
            generate_recons_from_chkpt(Path(chkpt_path), error_types=['ssim', 'l1', 'mse', 'l2']) # ['ssim', 'l1', 'mse', 'l2']
    print(f'----- \n Finished in {datetime.now() - init_time}')

if __name__ == '__main__':
    main()
