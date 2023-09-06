import os
import sys; sys.path.insert(0, os.path.abspath("../"))
from pathlib import Path
this_path = Path().resolve()
import numpy as np
import torch
import SimpleITK as sitk
import pandas as pd
import nibabel as nib
import pytorch_lightning as pl
import torchio as tio
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

from dataset.hmri_dataset import HMRIControlsDataModule, HMRIPDDataModule
from models.pl_model import Model_AE
from utils.utils import save_nifti_from_array, reconstruct

def main():
    # params
    ovlap = 6 # even number in pixel units
    roi_mask = None
    ckpt_path = Path('/home/alejandrocu/Documents/parkinson_classification/p2_hmri_outs/hp_tuning/p2_hmri_ch5_ps64/version_0/checkpoints/epoch=218-val_loss=0.0619-val_mse=0.0141.ckpt')
    print(f'Using checkpoint: {ckpt_path.parent.parent.parent.name}')
    print('---------------------------------')
    # read config file
    exp_dir = ckpt_path.parent.parent.parent
    with open(exp_dir /'config_dump.yml', 'r') as f:
        cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

    # create dataset
    root_dir = Path('/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI')
    md_df = pd.read_csv(this_path.parent/'bids_3t.csv')
    md_df_hc = md_df[md_df['group'] == 0]
    md_df_pd = md_df[md_df['group'] == 1]
    data = HMRIControlsDataModule(md_df=md_df_hc,
                                    root_dir=root_dir, 
                                    **cfg['dataset'])
    data.prepare_data()
    data.setup()

    data_pd = HMRIPDDataModule(md_df=md_df_pd,
                            root_dir=root_dir,  
                            **cfg['dataset'])
    data_pd.prepare_data()
    data_pd.setup()

    # create model
    model = Model_AE.load_from_checkpoint(ckpt_path, net='autoencoder', **cfg['model'])
    model = model.to('cuda')
    model.eval()

    re_all = []
    for i in tqdm(range(len(data.md_df_val)), total=len(data.md_df_val)):    
        for overlap in ['crop', 'average', 'hann']:    
            re_subj = {}    
            hc_patches, hc_locations, hc_sampler, hc_subject = data.get_grid(subj=i, overlap=ovlap, mode='val')
            hc_subj_id = data.md_df_train.iloc[i]['id']
            hc_data = [hc_patches, hc_locations, hc_sampler, hc_subject, hc_subj_id]
            if overlap == 'hann':
                if hc_subj_id == 'sub-027':
                    hc_rerror = reconstruct(hc_data, model, ckpt_path, overlap_mode=overlap, save_img=True, type='hn_hc')
            hc_rerror = reconstruct(hc_data, model, ckpt_path, overlap_mode=overlap, save_img=False, type='hc')
            re_subj['overlap'] = overlap
            re_subj['mean_re'] = np.mean(hc_rerror)
            re_subj['id'] = hc_subj_id
            re_subj['group'] = 'hc'
            re_all.append(re_subj)
    # print('hc done')
    # print('----------------------------------')

    for i in tqdm(range(len(data_pd.md_df)), total=len(data_pd.md_df)):    
        for overlap in ['crop', 'average', 'hann']:    
            re_subj = {}    
            pd_patches, pd_locations, pd_sampler, pd_subject = data_pd.get_grid(subj=i, overlap=ovlap)
            pd_subj_id = data_pd.md_df.iloc[i]['id']
            pd_data = [pd_patches, pd_locations, pd_sampler, pd_subject, pd_subj_id]
            if overlap == 'hann':
                if pd_subj_id == 'sub-004':
                    pd_rerror = reconstruct(pd_data, model, ckpt_path, overlap_mode=overlap, save_img=True, type='hn_pd')
            pd_rerror = reconstruct(pd_data, model, ckpt_path, overlap_mode=overlap, save_img=False, type='pd')
            re_subj['overlap'] = overlap
            re_subj['mean_re'] = np.mean(pd_rerror)
            re_subj['id'] = pd_subj_id
            re_subj['group'] = 'pd'
            re_all.append(re_subj)

    # print('pd done')
    # print('----------------------------------')

    # save dataframe with results
    out_dir = Path('/home/alejandrocu/Documents/parkinson_classification/reconstructions') / Path(ckpt_path).parent.parent.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(re_all).to_csv(out_dir/'reconstruction_errors.csv', index=False)

if __name__ == "__main__":
    main()
