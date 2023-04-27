from datetime import datetime
import os
from glob import glob
from pathlib import Path
this_path = Path().resolve()
import torch
import monai
import torchmetrics

import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml
import nibabel as nib
import SimpleITK as sitk
import cc3d

from dataset.ppmi_dataset import PPMIDataModule
from dataset.hmri_dataset import HMRIDataModule
from models.pl_model import Model

def save_nifti_from_array(subj_id: str,
                          arr: np.ndarray, 
                          path: Path):
    """subj_id = df.iloc[subj_idx]['id']
    print(f'Predicting for subject {subj_id} (with target {target.cpu().numpy()}')
    Save a nifti file from a numpy array and data from original nifti
    """
    img_name = f'{subj_id}_ses-01prisma3t_echo-01_part-magnitude-acq-MToff_MPM_MTsat_w.nii'
    img = nib.load(Path(f'/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI/{subj_id}/Results') 
                   / img_name)
    nifti = nib.Nifti1Image(arr, img.affine, img.header)
    nib.save(nifti, path)

def get_data_and_model(ckpt_path: Path, 
                      dataset: str = 'hmri'):

    # read config file
    exp_dir = ckpt_path.parent.parent.parent
    with open(exp_dir /'config_dump.yml', 'r') as f:
        cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]
    
    # create PPMI dataset
    
    if dataset == 'ppmi':
        # read metadata file and get the first scan for each subject
        root_dir = Path("/mnt/scratch/7TPD/mpm_run_acu/PPMI")
        md_df = pd.read_csv(root_dir/'t1_3d_3t_1mm_pdhc_2_16_2023.csv')
        md_df['Acq Date'] = md_df['Acq Date'].apply(pd.to_datetime)
        md_df.sort_values(by='Acq Date', inplace=True)
        first_acq_idx = md_df.duplicated(subset=['Subject'])
        md_df_first = md_df.loc[~first_acq_idx, :]    
        data = PPMIDataModule(md_df=md_df_first, root_dir=root_dir, shuffle=False, **cfg['dataset'])

        model = Model.load_from_checkpoint(ckpt_path, **cfg['model'])
    
    elif dataset == 'hmri':
        root_dir = Path('/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI')
        md_df = pd.read_csv(this_path/'bids_3t.csv')
        data = HMRIDataModule(md_df=md_df, root_dir=root_dir, **cfg['dataset']) #shuffle=False

        pretrained_model = get_pretrained_model(chkpt_path=Path(cfg['model']['chkpt_path']),
                                 input_channels=cfg['model']['in_channels'])
        
        model = Model.load_from_checkpoint(ckpt_path, net=pretrained_model.net, **cfg['model'])

    # create dataset
    data.prepare_data()
    data.setup()
    # print("Training:  ", len(data.train_set))
    # print("Validation: ", len(data.val_set))
    # print("Test:      ", len(data.test_set))

    return data, model
    
    

def predict_from_ckpt(ckpt_path: Path, 
                      dataloader: str = 'test', 
                      return_preds: bool = True,
                      dataset: str = 'hmri'):

    data, model = get_data_and_model(ckpt_path=ckpt_path,
                                    dataset=dataset)

    # obtain the dataloader
    if dataloader == 'test':
        dl = data.test_dataloader()
    elif dataloader == 'val':
        dl = data.val_dataloader()
    elif dataloader == 'train':
        dl = data.train_dataloader()
    
    # create model from checkpoint
    
    model.eval()

    # predict
    trainer = pl.Trainer(accelerator='gpu')
    predictions = trainer.predict(model, dl)

    preds = [item for sublist in predictions for item in sublist]
    preds_sf = [torch.softmax(item, dim=0) for item in preds]
    y_hat = [torch.argmax(item).cpu().numpy() for item in preds_sf]

    if return_preds:
        return y_hat, preds, data
    else:
        return y_hat, data

def get_pretrained_model(chkpt_path: Path, input_channels: int = 4):
    """Loads a pretrained model from a checkpoint and 
    replaces the first layer with a new one with the specified number of input channels

    Args:
        chkpt_path (Path): Path to checkpoint
        input_channels (int, optional): Conv3d layer input channels. Defaults to 4,
        as expected by the hMRI dataset

    Returns:
        Model: Pytorch Lightning model instance
    """
    # load config file
    exp_dir = chkpt_path.parent.parent.parent
    with open(exp_dir /'config_dump.yml', 'r') as f:
        exp_cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

    model = Model.load_from_checkpoint(chkpt_path, **exp_cfg['model'])

    if exp_cfg['model']['net'] == '3dresnet':
        if input_channels > 1:
            model.net.conv1 = torch.nn.Conv3d(in_channels= input_channels, 
                                        out_channels=64, 
                                        kernel_size=(7, 7, 7), 
                                        stride=(2, 2, 2), 
                                        padding=(3, 3, 3), 
                                        bias=False)
            print('Model loaded and first layer replaced')
            return model
        else:
            return model
    else:
        print('Model not supported')
        return None

# Reconstruction

def reconstruct(data, model, ckpt_path=None, overlap_mode='crop', save_img=False, out_dir=None, type='pd', vae=False):
    patches, locations, sampler, subject, subj_id = data
    input_imgs = patches.to(model.device)
    aggregator = tio.data.GridAggregator(sampler, overlap_mode=overlap_mode)

    with torch.no_grad():
        if vae:
            x_hat, _, _, _ = model(input_imgs)
        else:
            x_hat = model(input_imgs)

    aggregator.add_batch(x_hat, locations)
    reconstructed = aggregator.get_output_tensor()

    # Compute reconstruction error
    subject = subject['image'][tio.DATA]
    diff = [torch.pow(subject[i] - reconstructed[i], 2) for i in range(subject.shape[0])]
    rerror = torch.sqrt(torch.sum(torch.stack(diff), dim=0))
    rerror = rerror.cpu().numpy()
    
    if ckpt_path is not None:
        if out_dir is None:
            out_dir = Path('/home/alejandrocu/Documents/parkinson_classification/reconstructions') / Path(ckpt_path).parent.parent.parent.name
            out_dir.mkdir(parents=True, exist_ok=True)
        
    if save_img:
        save_nifti_from_array(subj_id=subj_id,
                              arr=reconstructed[0].cpu().numpy(),
                              path=out_dir / f'{type}_{subj_id}_recon.nii.gz')
        save_nifti_from_array(subj_id=subj_id,
                              arr=rerror,
                              path=out_dir / f'{type}_{subj_id}_re_error.nii.gz')
        save_nifti_from_array(subj_id=subj_id,
                              arr=subject[0].cpu().numpy(),
                              path=out_dir / f'{type}_{subj_id}_original.nii.gz')
    
    return rerror

# Brain segmentation 

def get_bounding_box_of_segmentation(binary_mask: np.ndarray):
    """
    Get the bounding box of a binary mask
    """
    # get bounding box
    bounding_box = np.argwhere(binary_mask)
    x_min, y_min, z_min = bounding_box.min(axis=0)
    x_max, y_max, z_max = bounding_box.max(axis=0)
    return x_min, x_max, y_min, y_max, z_min, z_max

def crop_img(img: np.ndarray, return_dims: bool = False, margin: int = 10):
    """
    Crop an image to its bounding box
    """
    # get bounding box
    x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box_of_segmentation(img)
    # crop with extra margin
    x_min = max(0, x_min - margin)
    x_max = min(img.shape[0], x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(img.shape[1], y_max + margin)
    z_min = max(0, z_min - margin)
    z_max = min(img.shape[2], z_max + margin)

    if return_dims:
        return x_min, x_max, y_min, y_max, z_min, z_max
    
    return img[x_min:x_max, y_min:y_max, z_min:z_max]

def obtain_brain_mask(subject: str, masks_path: Path = None):
    
    if masks_path is None:
        masks_path = Path(f'/mnt/scratch/7TPD/mpm_run_acu/bids/{subject}/ses-01prisma3t/anat')

    # get segmentation labels from all atlas
    base_str = f'{subject}_ses-01prisma3t_echo-01_part-magnitude-acq-T1w_MPM.nii'
    volumes = []
    for cl in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']:
        img = sitk.ReadImage(str(masks_path/f'{cl}{base_str}'))
        volumes.append(sitk.GetArrayFromImage(img))

    volumes_nda = np.stack(volumes, axis=0)
    labels = np.argmax(volumes_nda, axis=0)

    # get brain mask from 0 (gm) and 1 (wm) labels
    brain_mask = np.int16(labels < 2)

    # filter small objects
    cc_mask = cc3d.largest_k(brain_mask, k=1)
    
    return cc_mask, masks_path