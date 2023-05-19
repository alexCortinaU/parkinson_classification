from datetime import datetime
import os
import tempfile
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch import optim, nn, utils, Tensor, as_tensor
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import monai
import torchmetrics
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import torch
import yaml
from dataset.hmri_dataset import HMRIDataModule, HMRIDataModuleDownstream
from models.pl_model import Model, get_3dresnet, ContrastiveLearning, ModelDownstream
from utils.utils import get_pretrained_model
from monai.transforms import Compose, RandAffine, RandAdjustContrast
this_path = Path().resolve()

from pytorch_lightning.callbacks import BaseFinetuning

class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing `feature_extractor`
        self.freeze(pl_module.feature_extractor)

    def finetune_function(self, pl_module, current_epoch, optimizer, opt_idx):
        # When `current_epoch` is 10, feature_extractor will start training.
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor,
                optimizer=optimizer,
                train_bn=True,
                initial_denom_lr=10,
            )

def full_train_model(cfg):
    
     # Set data directory
    root_dir = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI')
    md_df = pd.read_csv(this_path/'bids_3t.csv')
    pd_number = len(md_df.loc[md_df.group == 1, :])
    hc_number = len(md_df.loc[md_df.group == 0, :])
    print(f'------------\n Data - PD: {pd_number}, HC: {hc_number}')

    # create PPMI dataset
    augmentations = Compose([
                        RandAffine(
                            rotate_range=((-np.pi/12, np.pi/12), 0, (-np.pi/12, np.pi/12)), 
                            scale_range=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)),
                            # translate_range=((-12, 12), (-12, 12), (-12, 12)),
                            padding_mode="zeros",
                            prob=1, 
                            mode='bilinear'),
                        RandAdjustContrast(prob=1, gamma=(0.5, 2.0)),
                        ])
    data = HMRIDataModuleDownstream(root_dir=root_dir,
                            md_df=md_df,
                            augment=augmentations,
                            **cfg['dataset'])
    data.prepare_data()
    data.setup()

    # create model
    # load config file
    chkpt_path = Path(cfg['model']['chkpt_path'])
    exp_dir = chkpt_path.parent.parent.parent
    with open(exp_dir /'config_dump.yml', 'r') as f:
        exp_cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

    pretrained_model = ContrastiveLearning.load_from_checkpoint(chkpt_path, hpdict=exp_cfg)

    # for downstream task, use commented ModelDownstream class in pl_model.py

    model = ModelDownstream(net=pretrained_model.model, **cfg['model'])

    # create callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=5,
                                          monitor=cfg['training']['monitor_ckpt'],
                                          mode="max",
                                          filename="{epoch:02d}-{val_auroc}")
    
    # finetune_callback = FeatureExtractorFreezeUnfreeze(unfreeze_at_epoch=cfg['training']['unfreeze_at_epoch'])
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    # early stopping doesn't work with monai's Meta tensors
    if cfg['training']['early_stopping']:
        early_stopping = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", 
                                                                mode='min', patience=70,
                                                                )
        callbacks = [checkpoint_callback, lr_monitor, early_stopping] # finetune_callback,
    else:
        print("---- \n ----- Early stopping is disabled \n ----")
        callbacks = [checkpoint_callback,lr_monitor] #  finetune_callback, 
    
    # create loggers
    tb_logger = TensorBoardLogger(save_dir=Path('./p4_downstream_outs'),
                               name=cfg['exp_name'],
                               version=0
                               )
    # save the config file to the output folder
    # for a given experiment
    dump_path = Path('./p4_downstream_outs').resolve() / f'{cfg["exp_name"]}'
    dump_path.mkdir(parents=True, exist_ok=True)
    dump_path = dump_path/'config_dump.yml'
    with open(dump_path, 'w') as f:
        yaml.dump(cfg, f)

    # create trainer
    trainer = pl.Trainer(**cfg['pl_trainer'],
                        callbacks=callbacks,
                        logger=[tb_logger],
                        )

    start = datetime.now()
    # print("Training started at", start)
    trainer.fit(model=model, datamodule=data)

    del trainer

    return datetime.now() - start, dump_path

def main():

    with open('./config/config_downstream.yaml', 'r') as f:
            cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

    # set random seed for reproducibility
    pl.seed_everything(cfg['dataset']['random_state'],  workers=True)

    maps = ['MTsat', 'R1'] # 'MTsat', 'R1', 'R2s_WLS1', 'PD_R2scorr'
    optimizers = ['adam'] # , 'sgd'
    lrates = [0.01, 0.001]
    # unfreeze_at_epochs = [15]
    # chckpt_paths = {'R2s_WLS1': "/mrhome/alejandrocu/Documents/parkinson_classification/p3_ssl_hmri/rs120_ssl_simclr_resnet_DA02_bs5_v2epochs400/version_0/checkpoints/epoch=326-val_loss=tensor(0.8306, device='cuda:0').ckpt",
    #                 'MTsat': "/mrhome/alejandrocu/Documents/parkinson_classification/p3_ssl_hmri/ssl_simclr_MTsat_optim_adam_lr_0.001/version_1/checkpoints/epoch=240-val_loss=tensor(1.0347, device='cuda:0').ckpt",
    #                 'R1': "/mrhome/alejandrocu/Documents/parkinson_classification/p3_ssl_hmri/ssl_simclr_R1_optim_adam_lr_0.001/version_0/checkpoints/epoch=351-val_loss=tensor(0.8743, device='cuda:0').ckpt",
    #                 'PD_R2scorr': "/mrhome/alejandrocu/Documents/parkinson_classification/p3_ssl_hmri/ssl_simclr_PD_R2scorr_optim_adam_lr_0.001/version_0/checkpoints/epoch=371-val_loss=tensor(0.5705, device='cuda:0').ckpt"}
    chckpt_paths = {'R2s_WLS1': "",
                    'MTsat': "/mrhome/alejandrocu/Documents/parkinson_classification/p3_ssl_hmri/ssl_hmri_MTsat_optim_adam_lr_0.001/version_0/checkpoints/epoch=119-val_loss=tensor(1.0069, device='cuda:0').ckpt",
                    'R1': "/mrhome/alejandrocu/Documents/parkinson_classification/p3_ssl_hmri/ssl_hmri_R1_optim_sgd_lr_0.01/version_0/checkpoints/epoch=319-val_loss=tensor(1.0122, device='cuda:0').ckpt",
                    'PD_R2scorr': ""}
    
    exps = 'new5_hMRI'
    exc_times = []
    for optim in optimizers:
        for map_type in maps:  
            for lr in lrates:                                         
                times = {}
                cfg['model']['chkpt_path'] = chckpt_paths[map_type]  
                cfg['model']['learning_rate'] = lr
                cfg['model']['optimizer_class'] = optim
                cfg['dataset']['map_type'] = [map_type]
                cfg['exp_name'] = f'{exps}_{map_type}_optim_{optim}_lr_{lr}'

                exc_time, dump_path = full_train_model(cfg)   

                times['exp_name'] = cfg['exp_name']  
                times['time'] = exc_time    
                exc_times.append(times)
        
    pd.DataFrame(exc_times).to_csv(dump_path.parent.parent/f'{exps}_execution_times_.csv', index=False)

if __name__ == "__main__":
    main()

