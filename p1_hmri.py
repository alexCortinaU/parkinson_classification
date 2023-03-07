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
import torchmetrics
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import yaml
from dataset.hmri_dataset import HMRIDataModule
from models.pl_model import Model
from utils.utils import get_pretrained_model

def main():

    this_path = Path().resolve()

    # read the config file
    with open('config.yaml', 'r') as f:
        cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

    # Set data directory
    root_dir = Path('/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI')
    md_df = pd.read_csv(this_path/'bids_3t.csv')
    pd_number = len(md_df.loc[md_df.group == 1, :])
    hc_number = len(md_df.loc[md_df.group == 0, :])
    print(f'------------\n Data - PD: {pd_number}, HC: {hc_number}')

    # create PPMI dataset
    augmentations = tio.Compose([                                        
                                tio.RandomAffine(scales=(0.15, 0.15, 0.15), 
                                                degrees=(15, 0, 15),
                                                # isotropic=True,
                                                # center='image',
                                                default_pad_value=0),
                                # # tio.RandomElasticDeformation(p=0.1, num_control_points=7, max_displacement=10),
                                # tio.RandomGamma(p=0.5),
                                # # tio.RandomNoise(p=0.5, mean=0.5, std=0.05), # p=0.5
                                # # tio.RandomMotion(p=0.1), #, degrees=20, translation=20),
                                # # tio.RandomBiasField(p=0.25),
                                ])
    
    # save augmentations to config file                                   
    cfg['aug'] = str(augmentations)

    data = HMRIDataModule(md_df=md_df,
                        root_dir=root_dir,
                        augment=augmentations,
                        **cfg['dataset'])
    data.prepare_data()
    data.setup()
    print("Training:  ", len(data.train_set))
    print("Validation: ", len(data.val_set))
    print("Test:      ", len(data.test_set))

    # create model
    pretrained_model = get_pretrained_model(chkpt_path=Path(cfg['model']['chkpt_path']),
                                 input_channels=cfg['model']['in_channels'])
    model = Model(net=pretrained_model.net, **cfg['model'])
    
    print(f"--- \n --- {cfg['exp_name']}")

    # create callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=5,
                                          monitor=cfg['training']['monitor_ckpt'],
                                          mode="max",
                                          filename="{epoch:02d}-{val_f1:.4f}")

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    if cfg['training']['early_stopping']:
        early_stopping = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", 
                                                                mode='min', patience=15,
                                                                )
        callbacks = [checkpoint_callback, lr_monitor, early_stopping]
    else:
        print("---- \n ----- Early stopping is disabled \n ----")
        callbacks = [checkpoint_callback, lr_monitor]

    # create loggers
    tb_logger = TensorBoardLogger(save_dir=Path('./p1_hmri_outs'),
                               name=cfg['exp_name'],
                               version=0
                               )
    
    csv_logger = CSVLogger(save_dir=Path('./p1_hmri_outs'),
                            flush_logs_every_n_steps=10,
                            name=cfg['exp_name'],
                            version=0
                            )
                            
    # save the config file to the output folder
    # for a given experiment
    dump_path = Path('./p1_hmri_outs').resolve() / f'{cfg["exp_name"]}'
    dump_path.mkdir(parents=True, exist_ok=True)
    dump_path = dump_path/'config_dump.yml'
    with open(dump_path, 'w') as f:
        yaml.dump(cfg, f)

    # create trainer
    trainer = pl.Trainer(**cfg['pl_trainer'],
                        callbacks=callbacks,
                        logger=[tb_logger, csv_logger],
                        )

    # # find optimal learning rate
    print('Default LR: ', model.lr)
    trainer.tune(model, datamodule=data)
    print('Tuned LR: ', model.lr)

    start = datetime.now()
    print("Training started at", start)
    trainer.fit(model=model, datamodule=data)
    print("Training duration:", datetime.now() - start)

if __name__ == "__main__":
    main()