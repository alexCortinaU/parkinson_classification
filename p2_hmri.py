from datetime import datetime
from glob import glob
from pathlib import Path
import pandas as pd
import torchio as tio
import torch
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from dataset.hmri_dataset import HMRIDataModule, HMRIControlsDataModule, HMRIPDDataModule
from models.pl_model import Model, Model_AE, GenerateReconstructions, ComputeRE
from utils.utils import get_pretrained_model


def main():
    torch.autograd.set_detect_anomaly(True)
    this_path = Path().resolve()

    # read the config file
    with open('config_patches.yaml', 'r') as f:
        cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

    # set random seed for reproducibility
    pl.seed_everything(cfg['dataset']['random_state'])

    # Set data directory
    root_dir = Path('/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI')
    md_df = pd.read_csv(this_path/'bids_3t.csv')
    md_df_hc = md_df[md_df['group'] == 0]
    md_df_pd = md_df[md_df['group'] == 1]
    print(f'Number of controls: {len(md_df_hc)}')

    # create augmentations
    augmentations = tio.Compose([])                              
    cfg['aug'] = str(augmentations) # save augmentations to config file     

    # create controls dataset
    data = HMRIControlsDataModule(md_df=md_df_hc,
                        root_dir=root_dir,
                        augment=augmentations,
                        **cfg['dataset'])
    data.prepare_data()
    data.setup()
    print("Training:  ", len(data.train_set))
    print("Validation: ", len(data.val_set))

    hc_patches, hc_locations, hc_sampler, hc_subject = data.get_grid()

    # create PD dataset (for RE callback)
    data_pd = HMRIPDDataModule(md_df=md_df_pd,
                               root_dir=root_dir,
                               augment=augmentations,
                               **cfg['dataset'])
    data_pd.prepare_data()
    data_pd.setup()

    pd_patches, pd_locations, pd_sampler, pd_subject = data_pd.get_grid()

    # create model
    model = Model_AE(patch_size=cfg['dataset']['patch_size'],
                      **cfg['model'])
    
    print(f"--- \n --- {cfg['exp_name']}")

    # create callbacks
    if 'vae' in cfg['model']['net']:
        is_vae = True
    else:
        is_vae = False

    if cfg['training']['display_recons']:
        train_display = GenerateReconstructions(data.get_images(num=4, mode='train'),
                                                every_n_epochs=10,
                                                split='train',
                                                vae=is_vae)
        val_display = GenerateReconstructions(data.get_images(num=4, mode='val'),
                                                every_n_epochs=10,
                                                split='val',
                                                vae=is_vae)
    # create reconstruction error callback

    hc_re_callback = ComputeRE(input_imgs=hc_patches, 
                               locations=hc_locations, 
                               sampler=hc_sampler, 
                               subject=hc_subject,
                               every_n_epochs=1,
                               cohort='controls',
                               vae=is_vae)
    
    pd_re_callback = ComputeRE(input_imgs=pd_patches,
                                locations=pd_locations,
                                sampler=pd_sampler,
                                subject=pd_subject,
                                every_n_epochs=1,
                                cohort='pd',
                                vae=is_vae)
    # create other callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=5,
                                          monitor=cfg['training']['monitor_ckpt'],
                                          mode="min",
                                          filename="{epoch:02d}-{val_loss:.4f}-{val_mse:.4f}")

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    callbacks = [checkpoint_callback, lr_monitor, train_display, val_display,
                 hc_re_callback, pd_re_callback]

    if cfg['training']['early_stopping']:
        early_stopping = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", 
                                                                mode='min', patience=15,
                                                                )
        callbacks.append(early_stopping)
    else:
        print("---- \n ----- Early stopping is disabled \n ----")

    # create loggers
    tb_logger = TensorBoardLogger(save_dir=Path('./p2_hmri_outs'),
                               name=cfg['exp_name'],
                               version=0
                               )
                            
    # save the config file to the output folder
    # for a given experiment
    dump_path = Path('./p2_hmri_outs').resolve() / f'{cfg["exp_name"]}'
    dump_path.mkdir(parents=True, exist_ok=True)
    dump_path = dump_path/'config_dump.yml'
    with open(dump_path, 'w') as f:
        yaml.dump(cfg, f)

    # create trainer
    trainer = pl.Trainer(**cfg['pl_trainer'],
                        callbacks=callbacks,
                        logger=tb_logger,
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