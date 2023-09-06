import torch
import torch.nn.functional as F

from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torchio as tio
from datetime import datetime

import os
import json
import numpy as np
import monai
import copy
from matplotlib.colors import LinearSegmentedColormap
from utils.utils import save_nifti_from_array


import torchvision
from torchvision import models
from torchvision import transforms
import pytorch_lightning as pl

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import LRP
from captum.attr import Saliency


import yaml
from dataset.hmri_dataset import HMRIDataModule, HMRIDataModuleDownstream
from models.pl_model import Model, ContrastiveLearning, ModelDownstream
from utils.utils import get_pretrained_model
this_path = Path().resolve()

class SoftmaxedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.model(x))
    
def get_subj_and_model(subj_id: str, ckpt_path: Path):

    # load config file    
    exp_dir = ckpt_path.parent.parent.parent
    with open(exp_dir /'config_dump.yml', 'r') as f:
        exp_cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]
    
    print(exp_dir.name)
    # set random seed for reproducibility
    pl.seed_everything(exp_cfg['dataset']['random_state'],  workers=True)

    # load model
    model = Model.load_from_checkpoint(ckpt_path, **exp_cfg['model'])

    # create dataset
    exp_cfg['dataset']['val_batch_size'] = 1
    exp_cfg['dataset']['train_batch_size'] = 1
    exp_cfg['dataset']['shuffle'] = False
    map_type = exp_cfg['dataset']['map_type'][0]

    root_dir = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI')
    md_df = pd.read_csv(this_path/'bids_3t.csv')

    augmentations = tio.Compose([                                        
                                # tio.RandomAffine(scales=(0.15, 0.15, 0.15), 
                                #                 degrees=(15, 0, 15),
                                #                 default_pad_value=0),
                                # tio.RandomGamma(p=0.5)
                                ])
    
    data = HMRIDataModule(md_df=md_df,
                        root_dir=root_dir,
                        augment=augmentations,
                        **exp_cfg['dataset'])
    data.prepare_data()
    data.setup()

    df = data.md_df_val
    subj_idx = df.index[df.id == subj_id].tolist()[0]

    input, target = data.val_set[subj_idx]['image'], data.val_set[subj_idx]['label']

    return input, target, model.net, map_type


def obtain_xai_maps(subj_id: str, ckpt_path: Path, 
                    save_img: bool = True, 
                    occlusion_att: str = 'ps5_stride3',
                    stride: int = 3,
                    patch_size: int = 5, 
                    grad_based_att: str = 'IntegratedGradients', 
                    n_steps: int = 200):

    """Performs occlusion and gradient based attribution methods for a given subject and model checkpoint.
    It saves the resulting XAI maps as nifti files.

    Args:

        subj_id (str): Subject ID to perform the attribution methods.
        ckpt_path (Path): Path to the checkpoint to load the model.
        save_img (bool, optional): Whether to save the images or not. Defaults to True.
        occlusion_att (str, optional): String to append to resulting XAI map. If None, it will not perform OS.
         Defaults to 'ps5_stride3'.
        stride (int, optional): Stride to use for the occlusion method. Defaults to 3.
        patch_size (int, optional): Patch size to use for the occlusion method. Defaults to 5.
        grad_based_att (str, optional): String to append to resulting XAI map. If None, it will not perform GB.
         Defaults to 'IntegratedGradients'.
        n_steps (int, optional): Number of steps to use for the gradient based methods. Defaults to 200.

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_image, target, network, map_type = get_subj_and_model(subj_id=subj_id, ckpt_path=ckpt_path)
    input, affine_m = input_image[tio.DATA].to(device).unsqueeze(0), input_image['affine']
    print(f'input shape: {input.shape}')
    print(f'Predicting for subject {subj_id} (with target {target.cpu().numpy()})')

    # prediction

    network = network.to(device)
    network.eval()

    with torch.no_grad():
        output = network(input)
        # print(f"model's logits output: {output}")
        print(f"Models output: {output}")
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    print(pred_label_idx, prediction_score)

    # occlusion
    if occlusion_att is not None:
        print('Performing Occlusion Sensitivity')
        start = datetime.now()
        # Captum
        occlusion = Occlusion(network)        
        attributions_occ = occlusion.attribute(input,
                                            strides = (1, stride, stride, stride),
                                            target=pred_label_idx,
                                            sliding_window_shapes=(1, patch_size, patch_size, patch_size),
                                            baselines=0)

        print('___________________\n')
        print(f'Occlusion attribution took {datetime.now() - start}')

        
        out_dir = Path(f'/mrhome/alejandrocu/Documents/parkinson_classification/xai_outs/{subj_id}/occ_sens') / ckpt_path.parent.parent.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # save the results for Monai map
        # save_nifti_from_array(subj_id=subj_id,
        #                         arr=occ_result[0, target.argmax().item()].cpu().detach().numpy(), 
        #                         path=out_dir / f'{subj_id}_{map_type}_{occlusion_att}_occ_result_logits.nii.gz')
        if save_img:
            # save the images for visualization
            save_nifti_from_array(subj_id=subj_id,
                                arr=attributions_occ[0, 0].detach().cpu().numpy(),
                                affine_matrix=affine_m, 
                                path=out_dir / f'final_{subj_id}_{map_type}_{occlusion_att}_occ_result.nii.gz')
            save_nifti_from_array(subj_id=subj_id,
                                    arr=input[0, 0].detach().cpu().numpy(),
                                    affine_matrix=affine_m, 
                                    path=out_dir / f'final_{subj_id}_{map_type}_{occlusion_att}_og_img.nii.gz')
        print(f'-------------- \n Images saved to {out_dir}')
        # del occlusion
        # del attributions_occ

    # integrated gradients
    if grad_based_att is not None:
        print('Performing Gradient Based Attribution')
        start = datetime.now()

        if 'Integrated' in grad_based_att:
            print('Performing Integrated Gradients')
        
            integrated_gradients = IntegratedGradients(network)
            attributions_ig = integrated_gradients.attribute(input, 
                                                            target=pred_label_idx, 
                                                            n_steps=n_steps, 
                                                            internal_batch_size=1)

            # noise_tunnel = NoiseTunnel(integrated_gradients)

            # attributions_ig_nt = noise_tunnel.attribute(input, 
            #                                             nt_samples=1, 
            #                                             nt_type='smoothgrad_sq', 
            #                                             target=pred_label_idx,
            #                                             nt_samples_batch_size=1)

            print('___________________\n')
            print(f'Integrated gradientes attribution took {datetime.now() - start}')


            # save the results
            out_dir = Path(f'/mrhome/alejandrocu/Documents/parkinson_classification/xai_outs/{subj_id}/grad_based') / Path(ckpt_path).parent.parent.parent.name
            out_dir.mkdir(parents=True, exist_ok=True)

            if save_img:
                # save the images for visualization
                save_nifti_from_array(subj_id=subj_id,
                                    arr=attributions_ig[0, 0].detach().cpu().numpy(), 
                                    affine_matrix=affine_m,
                                    path=out_dir / f'final_{subj_id}_{map_type}_{grad_based_att}_nsteps_{n_steps}_result.nii.gz') #n_step{n_steps}
                print(f'-------------- \n Images saved to {out_dir}')
            del integrated_gradients
            del attributions_ig
        
        elif 'shap' in grad_based_att:
            print('Performing Gradient SHAP Attribution')
            torch.manual_seed(0)
            np.random.seed(0)

            gradient_shap = GradientShap(network)

            # Defining baseline distribution of images
            rand_img_dist = torch.cat([input * 0, input * 1])

            attributions_gs = gradient_shap.attribute(input,
                                                    n_samples=n_steps,
                                                    stdevs=0.0001,
                                                    baselines=rand_img_dist,
                                                    target=pred_label_idx)
            
            # save the results
            out_dir = Path('/mrhome/alejandrocu/Documents/parkinson_classification/grad_based') / Path(ckpt_path).parent.parent.parent.name
            out_dir.mkdir(parents=True, exist_ok=True)

            if save_img:
                # save the images for visualization
                save_nifti_from_array(subj_id=subj_id,
                                    arr=attributions_gs[0, 0].cpu().detach().numpy(), 
                                    path=out_dir / f'{subj_id}_{grad_based_att}_gradshap_n_samples_{n_steps}_result.nii.gz') #n_step{n_steps}
                print(f'-------------- \n Images saved to {out_dir}')

def main():

    map_type = 'R2s_WLS1'
    ckpt_path = Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/6A_hMRI_R2s_WLS1_optim_adam_lr_0.01/version_0/checkpoints/epoch=42-val_auroc=0.8112.ckpt')
    subj_ids = ['sub-019', 'sub-032']

    # 'sub-021', 'sub-064', 'sub-042', 'sub-066', 'sub-041', 'sub-019',
    #             'sub-036', 'sub-039', 'sub-035', 'sub-071', 'sub-030', 'sub-015', 'sub-032', 'sub-050'

    for subj_id in subj_ids:
        print(f'Predicting for subject {subj_id}, map {map_type}')
        obtain_xai_maps(subj_id=subj_id, 
                        ckpt_path=ckpt_path,
                        occlusion_att='occ_s5_ps8',
                        stride=5,
                        patch_size=8,
                        grad_based_att='IntegratedGrads',
                        save_img=True)

if __name__ == "__main__":
    main()