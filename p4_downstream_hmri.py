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

def main():

    with open('./config/config_downstream.yaml', 'r') as f:
            cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

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
    model = ModelDownstream(net=pretrained_model.model, **cfg['model'])

    # create callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=5,
                                          monitor=cfg['training']['monitor_ckpt'],
                                          mode="max",
                                          filename="{epoch:02d}-{val_auroc}")
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    if cfg['training']['early_stopping']:
        early_stopping = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", 
                                                                mode='min', patience=60,
                                                                )
        callbacks = [checkpoint_callback, lr_monitor, early_stopping]
    else:
        print("---- \n ----- Early stopping is disabled \n ----")
        callbacks = [checkpoint_callback, lr_monitor]
    
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
    print("Training duration:", datetime.now() - start)

if __name__ == "__main__":
    main()

