from datetime import datetime
import os
import tempfile
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
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
this_path = Path().resolve()

def full_train_model(cfg, data): 

    # Set data directory
    # root_dir = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI')
    # pd_number = len(md_df.loc[md_df.group == 1, :])
    # hc_number = len(md_df.loc[md_df.group == 0, :])
    # print(f'------------\n Data - PD: {pd_number}, HC: {hc_number}')

    # create model
    pretrained_model = get_pretrained_model(chkpt_path=Path(cfg['model']['chkpt_path']),
                                 input_channels=cfg['model']['in_channels'])
    model = Model(net=pretrained_model.net, **cfg['model']) #net=pretrained_model.net, 
    
    # create callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=5,
                                          monitor=cfg['training']['monitor_ckpt'],
                                          mode="max",
                                          filename="{epoch:02d}-{val_auroc:.4f}")

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
    tb_logger = TensorBoardLogger(save_dir=Path('./new_p1_hmri_outs'),
                               name=cfg['exp_name'],
                               version=0
                               )
    
    csv_logger = CSVLogger(save_dir=Path('./new_p1_hmri_outs'),
                            flush_logs_every_n_steps=10,
                            name=cfg['exp_name'],
                            version=0
                            )
                            
    # save the config file to the output folder
    # for a given experiment
    dump_path = Path('./new_p1_hmri_outs').resolve() / f'{cfg["exp_name"]}'
    dump_path.mkdir(parents=True, exist_ok=True)
    dump_path = dump_path/'config_dump.yml'
    with open(dump_path, 'w') as f:
        yaml.dump(cfg, f)

    # create trainer
    trainer = pl.Trainer(**cfg['pl_trainer'],
                        callbacks=callbacks,
                        logger=[tb_logger, csv_logger],
                        )

    start = datetime.now()
    # print("Training started at", start)
    trainer.fit(model=model, datamodule=data)
    print("Training duration:", datetime.now() - start)
    del trainer

    return datetime.now() - start, dump_path

def main():

    print('-------\n HPT-2\n-------')
    # read the config file
    with open('../config/config.yaml', 'r') as f:
        cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

    # set random seed for reproducibility
    pl.seed_everything(cfg['dataset']['random_state'],  workers=True)

    # Set data directory
    root_dir = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI')
    md_df = pd.read_csv(this_path/'bids_3t.csv')
    subjs_to_drop = ['sub-058', 'sub-016']

    for drop_id in subjs_to_drop:
        md_df.drop(md_df[md_df.id == drop_id].index, inplace=True)
    md_df.reset_index(drop=True, inplace=True)

    augmentations = tio.Compose([                                        
                                tio.RandomAffine(scales=(0.15, 0.15, 0.15), 
                                                degrees=(15, 0, 15),
                                                # isotropic=True,
                                                # center='image',
                                                default_pad_value=0),
                                # # tio.RandomElasticDeformation(p=0.1, num_control_points=7, max_displacement=10),
                                tio.RandomGamma(p=0.5),
                                # # tio.RandomNoise(p=0.5, mean=0.5, std=0.05), # p=0.5
                                # # tio.RandomMotion(p=0.1), #, degrees=20, translation=20),
                                # # tio.RandomBiasField(p=0.25),
                                ])
    
    # save augmentations to config file                                   
    cfg['aug'] = str(augmentations)

    maps = ['R2s_WLS1'] # 'MTsat', 'R1', 'R2s_WLS1', 'PD_R2scorr'
    optimizers = ['adam'] #, 'sgd'] # , 'sgd'
    lrates = [0.01] #, 0.001]
    
    exps = 'cv2-4A_hMRI'
    exc_times = []
    for optim in optimizers:
        for map_type in maps:  
            for lr in lrates:                         
                times = {}
                cfg['model']['learning_rate'] = lr
                cfg['model']['optimizer_class'] = optim
                cfg['dataset']['map_type'] = [map_type]
                
                skf = StratifiedKFold(n_splits=5, random_state=cfg['dataset']['random_state'], shuffle=True)

                for i, (train_index, test_index) in enumerate(skf.split(md_df.id.values, md_df.group.values)):
                    
                    md_df_train = md_df.iloc[train_index, :]
                    md_df_test = md_df.iloc[test_index, :]
                    data = HMRIDataModule(md_df=md_df,
                        root_dir=root_dir,
                        augment=augmentations,
                        **cfg['dataset'])
                    data.prepare_data(md_df_train, md_df_test)
                    data.setup()

                    cfg['train_idxs'] = str(train_index)
                    cfg['test_idxs'] = str(test_index)
                    cfg['exp_name'] = f'{exps}_{map_type}_optim_{optim}_lr_{lr}_cv_{i}'
                    exc_time, dump_path = full_train_model(cfg, data)   

                    times['exp_name'] = cfg['exp_name']  
                    times['time'] = exc_time    
                exc_times.append(times)
        
    pd.DataFrame(exc_times).to_csv(dump_path.parent.parent/f'{exps}_execution_times_.csv', index=False)

if __name__ == "__main__":
    main()