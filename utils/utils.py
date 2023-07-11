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
from torchmetrics.functional import structural_similarity_index_measure

from dataset.ppmi_dataset import PPMIDataModule
from dataset.hmri_dataset import HMRIDataModule
from models.pl_model import Model

def unprocess_image(image_path: Path, original_image_path: Path, reshape_size: int = 180):

    """Reverse torchio preprocessing 

    Args:
        image_path (Path): Path to the image to be unprocessed (usually a XAI map)
        original_image_path (Path): Path to the original image (usually brain_masked volume)
        reshape_size (int, optional): Reshape size for the CropOrPad tio transform. Defaults to 180.

    Returns:
        img_to_original (nib): Nibabel image with the same orientation and dimensions as the original image
    """
    og_subject = tio.Subject(image=tio.ScalarImage(original_image_path))
    preprocess = tio.Compose(
            [   tio.ToCanonical(),
                tio.CropOrPad(reshape_size, 
                              padding_mode='minimum')
            ])
    og_subject_proc = preprocess(og_subject)    

    inv_crop = og_subject_proc.get_inverse_transform()
    image_nib = nib.load(image_path)
    if image_nib.shape != og_subject_proc.image.shape[1:]:
        print(f'Image shape {image_nib.shape} does not match preprocessed original image shape {og_subject_proc.image.shape[1:]}')
        return None

    img_ornt = nib.orientations.io_orientation(image_nib.affine)
    psr_ornt = nib.orientations.io_orientation(og_subject.image.affine)
    # Uncrop
    image_nib = inv_crop(image_nib)
    # FromCanonical
    from_canonical = nib.orientations.ornt_transform(img_ornt, psr_ornt)
    img_to_original = image_nib.as_reoriented(from_canonical)
    
    if (img_to_original.affine == og_subject.image.affine).all:
        return img_to_original
    else:
        print('Failed: Affine mismatch')
        return None
    
def get_indexes_from_cfg(chkpt_path):

    exp_dir = chkpt_path.parent.parent.parent
    with open(exp_dir /'config_dump.yml', 'r') as f:
        cv_cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]
    
    train_index = cv_cfg['train_idxs'].replace('[', '').replace(']', '').replace('\n', '').split(' ')
    train_index = [int(i) for i in train_index if i != '']

    test_index = cv_cfg['test_idxs'].replace('[', '').replace(']', '').replace('\n', '').split(' ')
    test_index = [int(i) for i in test_index if i != '']

    return train_index, test_index

def save_nifti_from_array(arr: np.ndarray,                           
                          subj_id: str,
                          path: Path,                          
                          affine_matrix = None,
                          header = None):
    """
    Save a nifti file from a numpy array and data from original nifti (if not provided)
    """
    if affine_matrix is None:
        img_name = f'{subj_id}_ses-01prisma3t_echo-01_part-magnitude-acq-MToff_MPM_MTsat_w.nii'
        img = nib.load(Path(f'/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI/{subj_id}/Results') 
                    / img_name)
        affine_matrix = img.affine
        header = img.header

    nifti = nib.Nifti1Image(arr, affine_matrix, header)
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
        model.net.conv1 = torch.nn.Conv3d(in_channels= input_channels, 
                                    out_channels=64, 
                                    kernel_size=(7, 7, 7), 
                                    stride=(2, 2, 2), 
                                    padding=(3, 3, 3), 
                                    bias=False)
        return model
    else:
        print('Model not supported')
        return None

# Reconstruction

def reconstruct(data, model, ckpt_path=None, overlap_mode='hann', save_img=False, out_dir=None, type='pd', ae_type='vae', error_type: str = 'L2'):
    input_imgs, locations, sampler, subject, subj_id = data
    # input_imgs = patches.to(model.device)
    aggregator = tio.data.GridAggregator(sampler, overlap_mode=overlap_mode)

    with torch.no_grad():
        if ae_type == 'svae':
            reconst_imgs, _, _, _ = model(input_imgs)
        elif ae_type == 'vqvae':
            reconst_imgs, _ = model(input_imgs)
        else:
            reconst_imgs = model(input_imgs)

    aggregator.add_batch(reconst_imgs, locations)
    reconstructed = aggregator.get_output_tensor()

    # Compute reconstruction error
    subject = subject['image'][tio.DATA]
    if error_type == 'l2':    
        diff = [torch.pow(subject[i] - reconstructed[i], 2) for i in range(subject.shape[0])]
        rerror = torch.sqrt(torch.sum(torch.stack(diff), dim=0))
    elif error_type == 'l1':
        diff = [torch.abs(subject[i] - reconstructed[i]) for i in range(subject.shape[0])]
        rerror = torch.sum(torch.stack(diff), dim=0)
    elif error_type == 'ssim':
        _, rerror = structural_similarity_index_measure(subject, reconstructed, reduction=None, return_full_image=True)
        rerror = rerror.squeeze()
    elif error_type == 'mse':
        diff = [torch.pow(subject[i] - reconstructed[i], 2) for i in range(subject.shape[0])]
        rerror = torch.mean(torch.stack(diff), dim=0)

    if ckpt_path is not None:
        if out_dir is None:
            out_dir = Path('/home/alejandrocu/Documents/parkinson_classification/reconstructions') / Path(ckpt_path).parent.parent.parent.name
            out_dir.mkdir(parents=True, exist_ok=True)
        
    if save_img:
        save_nifti_from_array(subj_id=subj_id,
                              arr=reconstructed[0].cpu().numpy(),
                              path=out_dir / f'{type}_{subj_id}_recon.nii.gz')
        save_nifti_from_array(subj_id=subj_id,
                              arr=rerror.cpu().numpy(),
                              path=out_dir / f'{type}_{subj_id}_re_error.nii.gz')
        save_nifti_from_array(subj_id=subj_id,
                              arr=subject[0].cpu().numpy(),
                              path=out_dir / f'{type}_{subj_id}_original.nii.gz')
    
    return rerror, reconstructed

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

def crop_img(img: np.ndarray, return_dims: bool = False):
    """
    Crop an image to its bounding box
    """
    # get bounding box
    x_min, x_max, y_min, y_max, z_min, z_max = get_bounding_box_of_segmentation(img)
    # crop with extra margin
    x_min = max(0, x_min - 10)
    x_max = min(img.shape[0], x_max + 10)
    y_min = max(0, y_min - 10)
    y_max = min(img.shape[1], y_max + 10)
    z_min = max(0, z_min - 10)
    z_max = min(img.shape[2], z_max + 10)

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