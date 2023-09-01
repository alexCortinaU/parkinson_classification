from datetime import datetime
from glob import glob
from pathlib import Path
import pandas as pd
import torchio as tio
import torch
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split, StratifiedKFold
import yaml
from dataset.hmri_dataset import HMRIDataModule, HMRIControlsDataModule, HMRIPDDataModule
from models.pl_model import Model, Model_AE, GenerateReconstructions, ComputeRE
from utils.utils import get_pretrained_model


def full_train_model(cfg, data, data_pd):    

    hc_patches, hc_locations, hc_sampler, hc_subject = data.get_grid()

    pd_patches, pd_locations, pd_sampler, pd_subject = data_pd.get_grid()

    # create model
    model = Model_AE(patch_size=cfg['dataset']['patch_size'],
                      **cfg['model'])
    
    print(f"--- \n --- {cfg['exp_name']}")

    # create callbacks
    
    if cfg['training']['display_recons']:
        train_display = GenerateReconstructions(data.get_images(num=4, mode='train'),
                                                every_n_epochs=10,
                                                split='train',
                                                ae_type=cfg['model']['net'])
        val_display = GenerateReconstructions(data.get_images(num=4, mode='val'),
                                                every_n_epochs=10,
                                                split='val',
                                                ae_type=cfg['model']['net'])
    # create reconstruction error callback

    hc_re_callback = ComputeRE(input_imgs=hc_patches, 
                               locations=hc_locations, 
                               sampler=hc_sampler, 
                               subject=hc_subject,
                               every_n_epochs=1,
                               cohort='controls',
                               ae_type=cfg['model']['net'])
    
    pd_re_callback = ComputeRE(input_imgs=pd_patches,
                                locations=pd_locations,
                                sampler=pd_sampler,
                                subject=pd_subject,
                                every_n_epochs=1,
                                cohort='pd',
                                ae_type=cfg['model']['net'])
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
    tb_logger = TensorBoardLogger(save_dir=Path('./p2_hmri_outs/cv'),
                               name=cfg['exp_name'],
                               version=0
                               )
                            
    # save the config file to the output folder
    # for a given experiment
    dump_path = Path('./p2_hmri_outs/cv').resolve() / f'{cfg["exp_name"]}'
    dump_path.mkdir(parents=True, exist_ok=True)
    dump_path = dump_path/'config_dump.yml'
    with open(dump_path, 'w') as f:
        yaml.dump(cfg, f)

    # create trainer
    trainer = pl.Trainer(**cfg['pl_trainer'],
                        callbacks=callbacks,
                        logger=tb_logger,
                        )

    start = datetime.now()
    print("Training started at", start)
    trainer.fit(model=model, datamodule=data)
    print("Training duration:", datetime.now() - start)

    del trainer

    return datetime.now() - start, dump_path

def main():

    # torch.autograd.set_detect_anomaly(True)
    this_path = Path().resolve()

    # read the config file
    with open('config/config_patches.yaml', 'r') as f:
        cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

    # set random seed for reproducibility
    pl.seed_everything(cfg['dataset']['random_state'])

    # Set data directory
    root_dir = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI')
    md_df = pd.read_csv(this_path/'bids_3t.csv')
    # subjs_to_drop = ['sub-058', 'sub-016']

    # for drop_id in subjs_to_drop:
    #     md_df.drop(md_df[md_df.id == drop_id].index, inplace=True)
    # md_df.reset_index(drop=True, inplace=True)

    md_df_hc = md_df[md_df['group'] == 0]
    md_df_pd = md_df[md_df['group'] == 1]
    print(f'Number of controls: {len(md_df_hc)}')

    # create augmentations
    augmentations = tio.Compose([])                              
    cfg['aug'] = str(augmentations) # save augmentations to config file     

    # create PD dataset (for RE callback)
    data_pd = HMRIPDDataModule(md_df=md_df_pd,
                               root_dir=root_dir,
                               augment=augmentations,
                               **cfg['dataset'])
    data_pd.prepare_data()
    data_pd.setup()

    maps = ['MTsat', 'R1', 'R2s_WLS1', 'PD_R2scorr'] # 'MTsat', 'R1', 'R2s_WLS1', 'PD_R2scorr'
    nets = ['svae'] # , 'autoencoder'
    # losses = [''] #, 0.001]
    
    exps = 'cv-normative_hMRI'
    exc_times = []
    for net in nets:
        for map_type in maps:  
            # for lr in lrates:                         
                times = {}
                cfg['model']['net'] = net
                if net == 'svae':
                    cfg['model']['loss'] = 'l1kld'
                else:
                    cfg['model']['loss'] = 'l1'

                cfg['dataset']['map_type'] = [map_type]
                
                skf = StratifiedKFold(n_splits=5, random_state=cfg['dataset']['random_state'], shuffle=True)

                for i, (train_index, test_index) in enumerate(skf.split(md_df_hc.id.values, md_df_hc.sex.values)):
                    
                    md_df_train = md_df_hc.iloc[train_index, :]
                    md_df_test = md_df_hc.iloc[test_index, :]
                    # create controls dataset
                    data = HMRIControlsDataModule(md_df=md_df_hc,
                                        root_dir=root_dir,
                                        augment=augmentations,
                                        **cfg['dataset'])
                    data.prepare_data(md_df_train, md_df_test)
                    data.setup()

                    cfg['train_idxs'] = str(train_index)
                    cfg['train_ids'] = str(md_df_train.id.values)
                    cfg['test_idxs'] = str(test_index)
                    cfg['test_ids'] = str(md_df_test.id.values)
                    cfg['exp_name'] = f'{exps}_{map_type}_{net}_cv_{i}'

                    print(f'---  {cfg["exp_name"]}')
                    print(f"train: {cfg['train_ids']}")
                    print(f"test: {cfg['test_ids']} \n ---")

                    exc_time, dump_path = full_train_model(cfg, data, data_pd)   
                    
                    times['exp_name'] = cfg['exp_name']  
                    times['time'] = exc_time    
                exc_times.append(times)
        
    pd.DataFrame(exc_times).to_csv(dump_path.parent.parent/f'{exps}_execution_times_.csv', index=False)

if __name__ == "__main__":
    main()