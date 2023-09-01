from datetime import datetime
import os
import tempfile
from glob import glob
from pathlib import Path
this_path = Path().resolve()
from sklearn.model_selection import train_test_split
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from torch.utils import tensorboard

from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import monai
import torchmetrics
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ResizeWithPadOrCrop,
    RandAdjustContrast,
    RandBiasField,
    RandAffine,
    ScaleIntensity,
)
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml

from dataset.ppmi_dataset import PPMIDataModule
from models.pl_model import Model
from utils.utils import get_data_and_model, predict_from_ckpt, save_nifti_from_array

def main():
    # define parameters
    ckpt_path = '/home/alejandrocu/Documents/parkinson_classification/p1_hmri_outs/brainhmri3dresnet-da02_bz3_focal_adam_lr0.001/version_0/checkpoints/epoch=49-val_auroc=0.7959.ckpt'
    subj_idx = 11
    depth_slice = 100
    mask_size = 10
    n_batch= 20
    overlap= 0.25
    occ_sens_b_box = None  # [-1, -1, -1, -1, depth_slice - 1, depth_slice]
    split = 'val'
    save_img = True
    
    # get dataset and model from checkpoint
    data, model = get_data_and_model(ckpt_path=Path(ckpt_path), dataset='hmri')

    # get input image and target
    if split == 'val':        
        input, target = data.val_set[subj_idx]['image'][tio.DATA], data.val_set[subj_idx]['label']
        df = data.md_df_val
    elif split == 'test':
        input, target = data.test_set[subj_idx]['image'][tio.DATA], data.test_set[subj_idx]['label']
        df = data.md_df_test
    else:
        input, target = data.train_set[subj_idx]['image'][tio.DATA], data.train_set[subj_idx]['label']
        df = data.md_df_train

    # set up model and data for inference
    subj_id = df.iloc[subj_idx]['id']
    print(f'Predicting for subject {subj_id} ({split}) with target {target.cpu().numpy()}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.net.to(device)

    img, label = input.unsqueeze(0).to(device), target.to(device)

    # create occlusion sensitivity object
    occ_sens = monai.visualize.OcclusionSensitivity(nn_module=model.net, 
                                                    mask_size=mask_size, 
                                                    n_batch=n_batch, 
                                                    overlap=overlap)
    # run occlusion sensitivity
    if occ_sens_b_box is not None:
        occ_result, _ = occ_sens(x=img, b_box=occ_sens_b_box)
    else:
        occ_result, _ = occ_sens(x=img)

    # save the results
    out_dir = Path('/home/alejandrocu/Documents/parkinson_classification/occ_sens') / Path(ckpt_path).parent.parent.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f'{subj_id}_occ_result.npy', 'wb') as f:
        np.save(f, occ_result)

    if save_img:
        # save the images for visualization
        save_nifti_from_array(subj_id=subj_id,
                              arr=occ_result[0, label.argmax().item()].cpu().numpy(), 
                              path=out_dir / f'{subj_id}_occ_result.nii.gz')
        save_nifti_from_array(subj_id=subj_id,
                                arr=img[0, 0].cpu().numpy(),
                                path=out_dir / f'{subj_id}_cropped_img.nii.gz')
        print(f'Images saved to {out_dir}')

if __name__ == "__main__":
    main()