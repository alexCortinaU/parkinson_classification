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
from dataset.ppmi_dataset import PPMIDataModule
from models.pl_model import Model

def main():
    # read the config file
    with open('../config/config_ppmi.yaml', 'r') as f:
        cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

    # read metadata file and get the first scan for each subject
    root_dir = Path("/mnt/scratch/7TPD/mpm_run_acu/PPMI")
    md_df = pd.read_csv(root_dir/'t1_3d_3t_1mm_pdhc_2_16_2023.csv')
    md_df['Acq Date'] = md_df['Acq Date'].apply(pd.to_datetime)
    md_df.sort_values(by='Acq Date', inplace=True)
    first_acq_idx = md_df.duplicated(subset=['Subject'])
    md_df_first = md_df.loc[~first_acq_idx, :]

    # create PPMI dataset
    augmentations = tio.Compose([
                                        
                                        tio.RandomAffine(scales=(0.1, 0.1, 0.1), 
                                                        degrees=(10, 0, 10),
                                                        # isotropic=True,
                                                        # center='image',
                                                        default_pad_value=0)
                                        # tio.RandomElasticDeformation(p=0.1, num_control_points=7, max_displacement=10),
                                        # tio.RandomGamma(p=0.5),
                                        # tio.RandomNoise(p=0.5, mean=0.5, std=0.05), # p=0.5
                                        # tio.RandomMotion(p=0.1), #, degrees=20, translation=20),
                                        # tio.RandomBiasField(p=0.25),
                                        ])
    
    # save augmentations to config file                                   
    cfg['aug'] = str(augmentations)

    data = PPMIDataModule(md_df=md_df_first, 
                            root_dir=root_dir, 
                            augment=augmentations, 
                            **cfg['dataset'])
    data.prepare_data()
    data.setup()
    # print("Training:  ", len(data.train_set))
    # print("Validation: ", len(data.val_set))
    # print("Test:      ", len(data.test_set))

    # create model
    model = Model(**cfg['model'])
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
    tb_logger = TensorBoardLogger(save_dir=Path('./p1_ppmi_outs'),
                               name=cfg['exp_name'],
                               version=0
                               )
    
    csv_logger = CSVLogger(save_dir=Path('./p1_ppmi_outs'),
                            flush_logs_every_n_steps=10,
                            name=cfg['exp_name'],
                            version=0
                            )
                            
    # save the config file to the output folder
    # for a given experiment
    dump_path = Path('./p1_ppmi_outs').resolve() / f'{cfg["exp_name"]}'
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