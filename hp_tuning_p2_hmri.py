from datetime import datetime
from glob import glob
from pathlib import Path
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from dataset.hmri_dataset import HMRIDataModule, HMRIControlsDataModule, HMRIPDDataModule
from models.pl_model import Model, Model_AE, GenerateReconstructions, ComputeRE
from utils.utils import get_pretrained_model, reconstruct
this_path = Path().resolve()

def full_train_model(cfg):

    # Set data directory
    # root_dir = Path('/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI')
    root_dir = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI')
    md_df = pd.read_csv(this_path/'bids_3t.csv')
    md_df_hc = md_df[md_df['group'] == 0]
    md_df_pd = md_df[md_df['group'] == 1]

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
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_last=True,
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
    # else:
    #     print("---- \n ----- Early stopping is disabled \n ----")

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
                        deterministic=True
                        )

    # # find optimal learning ratedeterministic=True
    # print('Default LR: ', model.lr)
    # trainer.tune(model, datamodule=data)
    # print('Tuned LR: ', model.lr)

    start = datetime.now()
    # print("Training started at", start)
    trainer.fit(model=model, datamodule=data)
    print("Training duration:", datetime.now() - start)
    del trainer
    return model, data, data_pd, datetime.now() - start, dump_path

def compute_recons(model, data_hc, data_pd, out_dir, ovlap=6, vae=False):
    re_all = []
    for i in range(len(data_hc.md_df_val)):    
        for overlap in ['hann']:    
            re_subj = {}    
            hc_patches, hc_locations, hc_sampler, hc_subject = data_hc.get_grid(subj=i, overlap=ovlap, mode='val')
            hc_subj_id = data_hc.md_df_train.iloc[i]['id']
            hc_data = [hc_patches, hc_locations, hc_sampler, hc_subject, hc_subj_id]
            if overlap == 'hann':
                if hc_subj_id == 'sub-027':
                    hc_rerror = reconstruct(hc_data, model, 
                                            overlap_mode=overlap, 
                                            save_img=True, 
                                            type='hn_hc', 
                                            out_dir=out_dir, 
                                            vae=vae)
            hc_rerror = reconstruct(hc_data, model, overlap_mode=overlap, save_img=False, type='hc', vae=vae)
            re_subj['overlap'] = overlap
            re_subj['mean_re'] = np.mean(hc_rerror)
            re_subj['id'] = hc_subj_id
            re_subj['group'] = 'hc'
            re_all.append(re_subj)
    
    for i in range(len(data_pd.md_df)):    
        for overlap in ['hann']:    
            re_subj = {}    
            pd_patches, pd_locations, pd_sampler, pd_subject = data_pd.get_grid(subj=i, overlap=ovlap)
            pd_subj_id = data_pd.md_df.iloc[i]['id']
            pd_data = [pd_patches, pd_locations, pd_sampler, pd_subject, pd_subj_id]
            if overlap == 'hann':
                if pd_subj_id == 'sub-004':
                    pd_rerror = reconstruct(pd_data, 
                                            model, 
                                            overlap_mode=overlap, 
                                            save_img=True, 
                                            type='hn_pd', 
                                            out_dir=out_dir,
                                            vae=vae)
            pd_rerror = reconstruct(pd_data, model, overlap_mode=overlap, save_img=False, type='pd', vae=vae)
            re_subj['overlap'] = overlap
            re_subj['mean_re'] = np.mean(pd_rerror)
            re_subj['id'] = pd_subj_id
            re_subj['group'] = 'pd'
            re_all.append(re_subj)

    pd.DataFrame(re_all).to_csv(out_dir/'reconstruction_errors.csv', index=False)

def main():

    print('-------\n HPT-2\n-------')
    # read the config file
    with open('config/config_patches.yaml', 'r') as f:
        cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]
    
    # set random seed for reproducibility
    pl.seed_everything(cfg['dataset']['random_state'],  workers=True)

    # parameters to tune
    # psizes = [96, 128]
    # lrates = [0.008, 0.001]

    maps = ['MTsat', 'R1', 'R2s_WLS1', 'PD_R2scorr']
    gammas = [0.95]

    model_net = 'autoencoder'

    exc_times = []
    exps = f'normative_{model_net}'
    for gamma in gammas:
        for map_type in maps: 
            times = {}   
            cfg['model']['gamma'] = gamma
            cfg['model']['net'] = model_net
            # cfg['dataset']['patch_size'] = ps
            cfg['dataset']['map_type'] = [map_type]
            cfg['exp_name'] = f'{exps}_{map_type}' #_gamma_{gamma}'
            model, data_hc, data_pd, exc_time, dump_path = full_train_model(cfg)
            # # load model from checkpoint
            # last_ckpt_path = dump_path/'version_0'/ 'checkpoints'/ 'last.ckpt'
            # model = Model_AE.load_from_checkpoint(last_ckpt_path, **cfg['model'])
            # model = model.to('cuda')
            # model.eval()
            # compute_recons(model, data_hc, data_pd, dump_path, vae=True)
            del model, data_hc, data_pd
            times['exp_name'] = cfg['exp_name']  
            times['time'] = exc_time    
            exc_times.append(times)
    
    pd.DataFrame(exc_times).to_csv(dump_path.parent/f'{exps}_sae_execution_times_.csv', index=False)

if __name__ == '__main__':
    main()