from datetime import datetime
import os
import tempfile
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch import optim, nn, utils, Tensor, as_tensor
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import monai
from monai.transforms import Compose
import torchmetrics
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import yaml
import torch
from dataset.hmri_dataset import HMRIDataModule, HMRIDataModuleDownstream
from models.pl_model import Model, ModelDownstream, ContrastiveLearning
from utils.utils import get_pretrained_model
from torchmetrics.functional import auroc, accuracy, f1_score
this_path = Path().resolve()

EXP_DIR_3 = {
    'MTsat': {
        '3A': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/3A_hMRI_MTsat_optim_adam_lr_0.01/version_0/checkpoints/epoch=69-val_auroc=0.9133.ckpt'),
        '3B': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/3B_hMRI_MTsat_optim_adam_lr_0.001/version_0/checkpoints/epoch=57-val_auroc=0.9235.ckpt'),
        '3C': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/3C_hMRI_MTsat_optim_adam_lr_0.001/version_0/checkpoints/epoch=33-val_auroc=0.8163.ckpt')
    },
    'R1': {
        '3A': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/3A_hMRI_R1_optim_adam_lr_0.001/version_0/checkpoints/epoch=124-val_auroc=0.9337.ckpt'),
        '3B': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/3B_hMRI_R1_optim_adam_lr_0.001/version_0/checkpoints/epoch=76-val_auroc=0.9286.ckpt'),
        '3C': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/3C_hMRI_R1_optim_adam_lr_0.01/version_0/checkpoints/epoch=133-val_auroc=0.9541.ckpt')
    },
    'R2s_WLS1': {
        '3A': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/3A_hMRI_R2s_WLS1_optim_adam_lr_0.001/version_0/checkpoints/epoch=19-val_auroc=0.6913.ckpt'),
        '3B': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/3B_hMRI_R2s_WLS1_optim_adam_lr_0.01/version_0/checkpoints/epoch=25-val_auroc=0.7500.ckpt'),
        '3C': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/3C_hMRI_R2s_WLS1_optim_adam_lr_0.001/version_0/checkpoints/epoch=50-val_auroc=0.6531.ckpt')
    },
    'PD_R2scorr': {
        '3A': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/3A_hMRI_PD_R2scorr_optim_adam_lr_0.001/version_0/checkpoints/epoch=38-val_auroc=0.7015.ckpt'),
        '3B': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/3B_hMRI_PD_R2scorr_optim_adam_lr_0.001/version_0/checkpoints/epoch=18-val_auroc=0.7730.ckpt'),
        '3C': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/3C_hMRI_PD_R2scorr_optim_adam_lr_0.001/version_0/checkpoints/epoch=41-val_auroc=0.7653.ckpt')
    }    
}

EXP_DIR_4 = {
    'MTsat': {
        '4A': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4A_hMRI_MTsat_optim_adam_lr_0.01/version_0/checkpoints/epoch=79-val_auroc=0.7372.ckpt'),
        '4B': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4B_hMRI_MTsat_optim_adam_lr_0.01/version_0/checkpoints/epoch=64-val_auroc=0.7066.ckpt'),
        '4C': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4C_hMRI_MTsat_optim_adam_lr_0.01/version_0/checkpoints/epoch=48-val_auroc=0.7015.ckpt')
    },
    'R1': {
        '4A': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4A_hMRI_R1_optim_adam_lr_0.001/version_0/checkpoints/epoch=84-val_auroc=0.7883.ckpt'),
        '4B': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4B_hMRI_R1_optim_adam_lr_0.001/version_0/checkpoints/epoch=65-val_auroc=0.7704.ckpt'),
        '4C': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4C_hMRI_R1_optim_adam_lr_0.01/version_0/checkpoints/epoch=88-val_auroc=0.7883.ckpt')
    },
    'R2s_WLS1': {
        '4A': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4A_hMRI_R2s_WLS1_optim_adam_lr_0.01/version_0/checkpoints/epoch=60-val_auroc=0.9388.ckpt'),
        '4B': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4B_hMRI_R2s_WLS1_optim_adam_lr_0.01/version_0/checkpoints/epoch=34-val_auroc=0.8163.ckpt'),
        '4C': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4C_hMRI_R2s_WLS1_optim_adam_lr_0.01/version_0/checkpoints/epoch=33-val_auroc=0.7832.ckpt')
    },
    'PD_R2scorr': {
        '4A': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4A_hMRI_PD_R2scorr_optim_adam_lr_0.001/version_0/checkpoints/epoch=65-val_auroc=0.8163.ckpt'),
        '4B': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4B_hMRI_PD_R2scorr_optim_adam_lr_0.001/version_0/checkpoints/epoch=119-val_auroc=0.7602.ckpt'),
        '4C': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4C_hMRI_PD_R2scorr_optim_adam_lr_0.01/version_0/checkpoints/epoch=99-val_auroc=0.7194.ckpt')
    }    
}

EXP_DIR_6 = {
    'MTsat': {
        '6A': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/6A_hMRI_MTsat_optim_adam_lr_0.01/version_0/checkpoints/epoch=123-val_auroc=0.7321.ckpt'),
        '6B': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/6B_hMRI_MTsat_optim_adam_lr_0.01/version_0/checkpoints/epoch=64-val_auroc=0.6888.ckpt'),
        '6C': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/6C_hMRI_MTsat_optim_adam_lr_0.01/version_0/checkpoints/epoch=77-val_auroc=0.6888.ckpt')
    },
    'R1': {
        '6A': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/6A_hMRI_R1_optim_adam_lr_0.01/version_0/checkpoints/epoch=70-val_auroc=0.6403.ckpt'),
        '6B': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/6B_hMRI_R1_optim_adam_lr_0.01/version_0/checkpoints/epoch=36-val_auroc=0.6939.ckpt'),
        '6C': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/6C_hMRI_R1_optim_adam_lr_0.01/version_0/checkpoints/epoch=46-val_auroc=0.6888.ckpt')
    },
    'R2s_WLS1': {
        '6A': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/6A_hMRI_R2s_WLS1_optim_adam_lr_0.01/version_0/checkpoints/epoch=42-val_auroc=0.8112.ckpt'),
        '6B': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/6B_hMRI_R2s_WLS1_optim_adam_lr_0.01/version_0/checkpoints/epoch=62-val_auroc=0.8138.ckpt'),
        '6C': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/6C_hMRI_R2s_WLS1_optim_adam_lr_0.01/version_0/checkpoints/epoch=54-val_auroc=0.8010.ckpt')
    },
    'PD_R2scorr': {
        '6A': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/6A_hMRI_PD_R2scorr_optim_adam_lr_0.01/version_0/checkpoints/epoch=59-val_auroc=0.6531.ckpt'),
        '6B': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/6B_hMRI_PD_R2scorr_optim_adam_lr_0.01/version_0/checkpoints/epoch=29-val_auroc=0.6658.ckpt'),
        '6C': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/6C_hMRI_PD_R2scorr_optim_adam_lr_0.01/version_0/checkpoints/epoch=64-val_auroc=0.6939.ckpt')
    }
}

EXP_DIR_5 = {
    'MTsat': {
        '5B-2': Path("/mrhome/alejandrocu/Documents/parkinson_classification/p4_downstream_outs/5B-2_hMRI_MTsat_optim_adam_lr_0.001/version_0/checkpoints/epoch=74-val_auroc=tensor(0.7602, device='cuda:0').ckpt")
    },
    'R1': {
        '5B-2': Path("/mrhome/alejandrocu/Documents/parkinson_classification/p4_downstream_outs/5B-2_hMRI_R1_optim_adam_lr_0.001/version_0/checkpoints/epoch=27-val_auroc=tensor(0.7449, device='cuda:0').ckpt")
    },
    'R2s_WLS1': {
        '5B-2': Path("/mrhome/alejandrocu/Documents/parkinson_classification/p4_downstream_outs/5B-2_hMRI_R2s_WLS1_optim_adam_lr_0.001/version_0/checkpoints/epoch=13-val_auroc=tensor(0.7602, device='cuda:0').ckpt")
    },
    'PD_R2scorr': {
        '5B-2': Path("/mrhome/alejandrocu/Documents/parkinson_classification/p4_downstream_outs/5B-2_hMRI_PD_R2scorr_optim_adam_lr_0.001/version_0/checkpoints/epoch=17-val_auroc=tensor(0.6837, device='cuda:0').ckpt")
    }
}

def perform_inference_exp5(chkpt_path: Path, phase: str = 'val'):
     # read model from checkpoint and set up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config file    
    exp_dir = chkpt_path.parent.parent.parent
    with open(exp_dir /'config_dump.yml', 'r') as f:
        exp_cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]
    
    print(exp_dir.name)
    # set random seed for reproducibility
    pl.seed_everything(exp_cfg['dataset']['random_state'],  workers=True)

    # load model
    ssl_chkpt_path = Path(exp_cfg['model']['chkpt_path'])
    ssl_exp_dir = ssl_chkpt_path.parent.parent.parent
    with open(ssl_exp_dir /'config_dump.yml', 'r') as f:
        ssl_exp_cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

    pretrained_model = ContrastiveLearning.load_from_checkpoint(ssl_chkpt_path, hpdict=ssl_exp_cfg)

    # for downstream task, use commented ModelDownstream class in pl_model.py

    model = ModelDownstream(net=pretrained_model.model, **exp_cfg['model'])

    model = model.to(device)
    model.eval()

    # create dataset
    exp_cfg['dataset']['val_batch_size'] = 1
    exp_cfg['dataset']['train_batch_size'] = 1
    exp_cfg['dataset']['shuffle'] = False

    root_dir = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI')
    md_df = pd.read_csv(this_path/'bids_3t.csv')

    augmentations = Compose([])

    data = HMRIDataModuleDownstream(root_dir=root_dir,
                            md_df=md_df,
                            augment=augmentations,
                            **exp_cfg['dataset'])
    data.prepare_data()
    data.setup()

    dataloaders = {'val': data.val_dataloader(),
                    'train': data.train_dataloader()}
    
    print(f'No. {phase} subjects: {len(dataloaders[phase])}')
    # Get predictions
    save_list = []
    logits_list = []
    preds_list = []
    sm_preds_list = []
    targets_list = []
    targets_int = []
    for i, batch in enumerate(dataloaders[phase]):
        subj_results = {}
        if phase == 'val':
            subj_results['subj_id'] = data.md_df_val['id'][i]
        else:
            subj_results['subj_id'] = data.md_df_train['id'][i]

        inputs, targets = batch['image'][tio.DATA], batch['label']
        inputs = inputs.to(device)
        targets = targets.to(device).long()
        targets_list.append(targets)
        targets_int.append(torch.argmax(targets, dim=1))
        logits = model.net(inputs)
        logits_list.append(logits)
        sm_preds = torch.softmax(logits, dim=1)
        sm_preds_list.append(sm_preds)
        preds = torch.argmax(sm_preds, dim=1)
        preds_list.append(preds)

        subj_results['preds'] = preds.detach().cpu().numpy()
        subj_results['logits'] = logits.detach().cpu().numpy()        
        subj_results['targets'] = targets.detach().cpu().numpy()
        subj_results['sm_preds'] = sm_preds.detach().cpu().numpy()
        subj_results['split'] = phase
        subj_results['exp_name'] = exp_dir.name
        save_list.append(subj_results)

    # save results
    save_df = pd.DataFrame(save_list)

    exp_results = {}
    exp_results['auroc_sm'] = auroc(torch.cat(sm_preds_list, dim=0), torch.cat(targets_list, dim=0), task='binary').cpu().numpy()
    exp_results['auroc_log'] = auroc(torch.cat(logits_list, dim=0), torch.cat(targets_list, dim=0), task='binary').cpu().numpy()
    exp_results['acc'] = accuracy(torch.cat(preds_list, dim=0), torch.cat(targets_int, dim=0), task='binary', num_classes=2).cpu().numpy()
    exp_results['f1'] = f1_score(torch.cat(preds_list, dim=0), torch.cat(targets_int, dim=0), task='binary', num_classes=2).cpu().numpy()
    exp_results['exp_name'] = exp_dir.name
    exp_results['split'] = phase
    exp_results['map_type'] = exp_cfg['dataset']['map_type']

    return save_df, pd.DataFrame(exp_results, index=[0])


def perform_inference(chkpt_path: Path, phase: str = 'val'):
    # read model from checkpoint and set up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config file    
    exp_dir = chkpt_path.parent.parent.parent
    with open(exp_dir /'config_dump.yml', 'r') as f:
        exp_cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]
    
    print(exp_dir.name)
    # set random seed for reproducibility
    pl.seed_everything(exp_cfg['dataset']['random_state'],  workers=True)

    # load model
    model = Model.load_from_checkpoint(chkpt_path, **exp_cfg['model'])

    model = model.to(device)
    model.eval()

    # create dataset
    exp_cfg['dataset']['val_batch_size'] = 1
    exp_cfg['dataset']['train_batch_size'] = 1
    exp_cfg['dataset']['shuffle'] = False

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

    dataloaders = {'val': data.val_dataloader(),
                    'train': data.train_dataloader()}
    
    print(f'No. {phase} subjects: {len(dataloaders[phase])}')
    # Get predictions
    save_list = []
    logits_list = []
    preds_list = []
    sm_preds_list = []
    targets_list = []
    targets_int = []
    for i, batch in enumerate(dataloaders[phase]):
        subj_results = {}
        if phase == 'val':
            subj_results['subj_id'] = data.md_df_val['id'][i]
        else:
            subj_results['subj_id'] = data.md_df_train['id'][i]

        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device).long()
        targets_list.append(targets)
        targets_int.append(torch.argmax(targets, dim=1))
        logits = model.net(inputs)
        logits_list.append(logits)
        sm_preds = torch.softmax(logits, dim=1)
        sm_preds_list.append(sm_preds)
        preds = torch.argmax(sm_preds, dim=1)
        preds_list.append(preds)

        subj_results['preds'] = preds.detach().cpu().numpy()
        subj_results['logits'] = logits.detach().cpu().numpy()        
        subj_results['targets'] = targets.detach().cpu().numpy()
        subj_results['sm_preds'] = sm_preds.detach().cpu().numpy()
        subj_results['split'] = phase
        subj_results['exp_name'] = exp_dir.name
        save_list.append(subj_results)

    # save results
    save_df = pd.DataFrame(save_list)

    exp_results = {}
    exp_results['auroc_sm'] = auroc(torch.cat(sm_preds_list, dim=0), torch.cat(targets_list, dim=0), task='binary').cpu().numpy()
    exp_results['auroc_log'] = auroc(torch.cat(logits_list, dim=0), torch.cat(targets_list, dim=0), task='binary').cpu().numpy()
    exp_results['acc'] = accuracy(torch.cat(preds_list, dim=0), torch.cat(targets_int, dim=0), task='binary', num_classes=2).cpu().numpy()
    exp_results['f1'] = f1_score(torch.cat(preds_list, dim=0), torch.cat(targets_int, dim=0), task='binary', num_classes=2).cpu().numpy()
    exp_results['exp_name'] = exp_dir.name
    exp_results['split'] = phase
    exp_results['map_type'] = exp_cfg['dataset']['map_type']

    return save_df, pd.DataFrame(exp_results, index=[0])

if __name__ == '__main__':
    phase = 'train'
    # set up experiment directory
    results_path = Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs')
    results_path = results_path / 'classification_results'
    results_path.mkdir(exist_ok=True, parents=True)
    # chkpt_path = Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/3A_hMRI_MTsat_optim_adam_lr_0.01/version_0/checkpoints/epoch=69-val_auroc=0.9133.ckpt')
    predictions = pd.DataFrame()
    results = pd.DataFrame()
    for map_type, exps_paths in EXP_DIR_3.items():
        for exp_name, chkpt_path in exps_paths.items():
            print(exp_name)
            save_df, exp_results = perform_inference(chkpt_path, phase)
            exp_results['exp_type'] = exp_name
            save_df['map_type'] = map_type
            save_df['exp_type'] = exp_name
            predictions = pd.concat([predictions, save_df], axis=0)
            results = pd.concat([results, exp_results], axis=0)
    predictions.to_csv(results_path/f'3_{phase}_inference_results.csv', index=False)
    results.to_csv(results_path/f'3_{phase}_exps_results.csv', index=False)
    # save_df.to_csv(exp_dir/'inference_results.csv', index=False)
    # print(save_df)
    print(results)



