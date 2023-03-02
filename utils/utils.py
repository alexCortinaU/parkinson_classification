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

from dataset.ppmi_dataset import PPMIDataModule
from models.pl_model import Model

def predict_from_ckpt(ckpt_path: Path, dataloader: str = 'test', return_preds: bool = True):

    # read config file
    exp_dir = ckpt_path.parent.parent.parent
    with open(exp_dir.parent /'config_dump.yml', 'r') as f:
        cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]
    
    # create PPMI dataset

    # read metadata file and get the first scan for each subject
    root_dir = Path("/mnt/scratch/7TPD/mpm_run_acu/PPMI")
    md_df = pd.read_csv(root_dir/'t1_3d_3t_1mm_pdhc_2_16_2023.csv')
    md_df['Acq Date'] = md_df['Acq Date'].apply(pd.to_datetime)
    md_df.sort_values(by='Acq Date', inplace=True)
    first_acq_idx = md_df.duplicated(subset=['Subject'])
    md_df_first = md_df.loc[~first_acq_idx, :]

    # create dataset
    data = PPMIDataModule(md_df=md_df_first, root_dir=root_dir, **cfg['dataset'])
    data.prepare_data()
    data.setup()
    print("Training:  ", len(data.train_set))
    print("Validation: ", len(data.val_set))
    print("Test:      ", len(data.test_set))

    # obtain the dataloader
    if dataloader == 'test':
        dl = data.test_dataloader()
    elif dataloader == 'val':
        dl = data.val_dataloader()
    elif dataloader == 'train':
        dl = data.train_dataloader()
    
    # create model from checkpoint
    model = Model.load_from_checkpoint(ckpt_path, **cfg['model'])
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
