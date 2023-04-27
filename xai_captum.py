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
import copy
from matplotlib.colors import LinearSegmentedColormap
from utils.utils import save_nifti_from_array


import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import LRP


import yaml
from dataset.hmri_dataset import HMRIDataModule, HMRIDataModuleDownstream
from models.pl_model import Model, ContrastiveLearning, ModelDownstream
from utils.utils import get_pretrained_model
this_path = Path().resolve()

def obtain_xai_maps(subj_idx: int, 
                    ckpt_path: Path,
                    save_img: bool = True, 
                    occlusion_att: str = 'ps10_stride5', 
                    grad_based_att: str = 'IntegratedGradients', 
                    n_steps: int = 200):

    # define parameters
    # save_img = True
    # ckpt_path = Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4A_hMRI_R2s_WLS1_optim_adam_lr_0.01/version_0/checkpoints/epoch=60-val_auroc=0.9388.ckpt')
    
    # occlusion_att = 'ps10_stride5' #'ps5_stride1'
    # grad_based_att = 'IntegratedGradients'
    # n_steps = 200

    # read config file and set up data and params
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    exp_dir = ckpt_path.parent.parent.parent
    with open(exp_dir /'config_dump.yml', 'r') as f:
        cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

    map_type = cfg['dataset']['map_type']
    print(f'Using {map_type} maps')

    if map_type[0] in ['R2_WLS1', 'PD_R2scorr']:
        root_dir = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI')
        md_df = pd.read_csv(this_path/'bids_3t.csv')
        data = HMRIDataModule(md_df=md_df, root_dir=root_dir, **cfg['dataset']) #shuffle=False

        # loading model from checkpoint
        model = Model.load_from_checkpoint(ckpt_path, **cfg['model'])
        network = copy.deepcopy(model.net)
        network.eval()
        # create dataset
        data.prepare_data()
        data.setup()
        input, target = data.val_set[subj_idx]['image'][tio.DATA], data.val_set[subj_idx]['label']

    elif map_type[0] in ['R1', 'MTsat']:
        root_dir = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI')
        md_df = pd.read_csv(this_path/'bids_3t.csv')
        
        data = HMRIDataModuleDownstream(root_dir=root_dir,
                                md_df=md_df,
                                **cfg['dataset'])
        # loading model from checkpoint
        chkpt_path = Path(cfg['model']['chkpt_path'])
        exp_dir = chkpt_path.parent.parent.parent
        with open(exp_dir /'config_dump.yml', 'r') as f:
            exp_cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

        pretrained_model = ContrastiveLearning.load_from_checkpoint(chkpt_path, hpdict=exp_cfg)
        model = ModelDownstream.load_from_checkpoint(ckpt_path, net=pretrained_model.model, **cfg['model'])

        network = copy.deepcopy(model)
        network.eval()
        # create dataset
        data.prepare_data()
        data.setup()
        input, target = data.val_set[subj_idx]
        input = input.as_tensor() # convert Monai Metatensor to torch tensor for captum compatibility

    input = input.to(device).unsqueeze(0)
    df = data.md_df_val

    subj_id = df.iloc[subj_idx]['id']
    print(f'Predicting for subject {subj_id} (with target {target.cpu().numpy()}')

    # prediction
    network.to(device)
    with torch.no_grad():
        output = network(input)
        # print(f"model's logits output: {output}")
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    # print(pred_label_idx, prediction_score)

    # occlusion
    if occlusion_att is not None:
        print('Performing Occlusion Sensitivity')
        occlusion = Occlusion(network)
        start = datetime.now()
        attributions_occ = occlusion.attribute(input,
                                            strides = (1, 5, 5, 5),
                                            target=pred_label_idx,
                                            sliding_window_shapes=(1, 10, 10, 10),
                                            baselines=0)
        print('___________________\n')
        print(f'Occlusion attribution took {datetime.now() - start}')
        # save the results
        out_dir = Path(f'/mrhome/alejandrocu/Documents/parkinson_classification/xai_outs/{subj_id}/occ_sens') / Path(ckpt_path).parent.parent.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)

        if save_img:
            # save the images for visualization
            save_nifti_from_array(subj_id=subj_id,
                                arr=attributions_occ[0, 0].cpu().detach().numpy(), 
                                path=out_dir / f'{subj_id}_{map_type}_{occlusion_att}_occ_result.nii.gz')
            save_nifti_from_array(subj_id=subj_id,
                                    arr=input[0, 0].cpu().detach().numpy(),
                                    path=out_dir / f'{subj_id}_{map_type}_{occlusion_att}_og_img.nii.gz')
            print(f'-------------- \n Images saved to {out_dir}')
        del occlusion
        del attributions_occ

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
                                    arr=attributions_ig[0, 0].cpu().detach().numpy(), 
                                    path=out_dir / f'{subj_id}_{map_type}_{grad_based_att}_ig_n_steps_{n_steps}_result.nii.gz') #n_step{n_steps}
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

    idxs = [1, 4, 6, 9, 11]
    maps = {
        # 'R2_WLS1': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4A_hMRI_R2s_WLS1_optim_adam_lr_0.01/version_0/checkpoints/epoch=60-val_auroc=0.9388.ckpt'),
        'MTsat': Path("/mrhome/alejandrocu/Documents/parkinson_classification/p4_downstream_outs/5B_hMRI_MTsat_optim_adam_lr_0.001_ufrz_15/version_0/checkpoints/epoch=76-val_auroc=tensor(0.7832, device='cuda:0').ckpt"),
        'R1': Path("/mrhome/alejandrocu/Documents/parkinson_classification/p4_downstream_outs/5B_hMRI_R1_optim_adam_lr_0.001_ufrz_15/version_0/checkpoints/epoch=54-val_auroc=tensor(0.8316, device='cuda:0').ckpt"),
        'PD_R2scorr': Path("/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4A_hMRI_PD_R2scorr_optim_adam_lr_0.001/version_0/checkpoints/epoch=65-val_auroc=0.8163.ckpt")
    }
    for idx in idxs:
        for map_type, ckpt_path in maps.items():
            print('\n')
            print('___________________\n')
            print(f'Predicting for subject {idx}, map {map_type}')
            obtain_xai_maps(subj_idx=idx, 
                            ckpt_path=ckpt_path,
                            grad_based_att=None,
                            save_img=True)

if __name__ == "__main__":
    main()