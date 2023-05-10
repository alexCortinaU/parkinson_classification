from datetime import datetime
from glob import glob
from pathlib import Path
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import numpy as np
import torch
from torch import optim, nn
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from dataset.hmri_dataset import HMRIDataModule, HMRIControlsDataModule, HMRIPDDataModule
from models.pl_model import Model, Model_AE, GenerateReconstructions, ComputeRE
from GenerativeModels.generative.networks.nets import VQVAE
from utils.utils import get_pretrained_model, reconstruct
import torchvision
import torchmetrics
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
this_path = Path().resolve()

def full_train_model(cfg):

    root_dir = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI')
    md_df = pd.read_csv(this_path/'bids_3t.csv')
    md_df_hc = md_df[md_df['group'] == 0]
    md_df_pd = md_df[md_df['group'] == 1]

    # create augmentations
    augmentations = tio.Compose([])                              
    cfg['aug'] = str(augmentations) # save augmentations to config file
    exps = cfg['exp_name']     

    # create controls dataset
    data = HMRIControlsDataModule(md_df=md_df_hc,
                        root_dir=root_dir,
                        augment=augmentations,
                        **cfg['dataset'])
    data.prepare_data()
    data.setup()
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")    

    channels = cfg['model']['channels']
    latent_size = cfg['model']['latent_size']
    # create model
    vqvae_model = VQVAE(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                num_res_layers=2,
                downsample_parameters=((2, 3, 1, 1), (2, 3, 1, 1)),
                upsample_parameters=((2, 3, 1, 1, 1), (2, 3, 1, 1, 1)),
                num_channels=channels, #(96, 96),
                num_res_channels=channels, #,
                num_embeddings=latent_size, # 256,
                embedding_dim=32,
                act='LEAKYRELU'
                )
    vqvae_model = vqvae_model.to(device)
    optimizer = optim.Adam(params=vqvae_model.parameters(), lr=cfg['model']['learning_rate'])
    l1_loss = nn.L1Loss()
    mse_error = torchmetrics.MeanSquaredError().to(device)

    n_epochs = cfg['pl_trainer']['max_epochs']
    val_interval = 5
    # epoch_recon_loss_list = []
    # epoch_quant_loss_list = []
    # val_recon_epoch_loss_list = []
    # intermediary_images = []
    n_example_images = 4

    model_path = this_path / 'vqvae_models' / exps
    model_path.mkdir(parents=True, exist_ok=True)
    log_path = model_path / 'logs'
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_path)
    dump_path = model_path/'config_dump.yml'
    with open(dump_path, 'w') as f:
        yaml.dump(cfg, f)

    total_start = datetime.now()
    for epoch in range(n_epochs):
        vqvae_model.train()
        epoch_loss = 0
        epoch_mse = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"][tio.DATA].to(device)
            optimizer.zero_grad(set_to_none=True)

            # model outputs reconstruction and the quantization error
            reconstruction, quantization_loss = vqvae_model(images=images)

            recons_loss = l1_loss(reconstruction.float(), images.float())
            mse = mse_error(reconstruction.float(), images.float())

            loss = recons_loss + quantization_loss

            loss.backward()
            optimizer.step()

            epoch_loss += recons_loss.item()
            epoch_mse += mse.item()

            progress_bar.set_postfix(
                {"recons_loss": epoch_loss / (step + 1), "quantization_loss": quantization_loss.item() / (step + 1)}
            )
        # epoch_recon_loss_list.append(epoch_loss / (step + 1))
        # epoch_quant_loss_list.append(quantization_loss.item() / (step + 1))

        writer.add_scalars('loss_recon',{'train': epoch_loss / (step + 1)}, epoch)
        writer.add_scalars('loss_quant',{'train': quantization_loss.item() / (step + 1)}, epoch)
        writer.add_scalars('error_mse',{'train': epoch_mse / (step + 1)}, epoch)

        if (epoch + 1) % val_interval == 0:
            vqvae_model.eval()
            val_loss = 0
            val_mse = 0
            with torch.no_grad():
                # k = 0
                for val_step, batch in enumerate(val_loader): # , start=1
                    # k += 1
                    # if k == 3:
                    #     break
                    images = batch["image"][tio.DATA].to(device)

                    reconstruction, quantization_loss = vqvae_model(images=images)

                    # get the first sample from the first validation batch for
                    # visualizing how the training evolves
                    if val_step == 1:
                    #     intermediary_images.append(reconstruction[:n_example_images, 0])
                        slice = images.shape[-1] // 2
                        imgs = torch.stack([images[:n_example_images, ..., slice].detach(), reconstruction[:n_example_images, ..., slice].detach()], dim=1).flatten(0, 1)
                        grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
                        writer.add_image('reconstruction', grid, epoch)

                    recons_loss = l1_loss(reconstruction.float(), images.float())
                    mse = mse_error(reconstruction.float(), images.float())

                    val_loss += recons_loss.item()
                    val_mse += mse.item()

            val_loss /= val_step
            writer.add_scalars('loss_recon',{'val': val_loss}, epoch)
            writer.add_scalars('loss_quant',{'val': quantization_loss.item() / (step + 1)}, epoch)
            writer.add_scalars('error_mse',{'val': val_mse / (step + 1)}, epoch)
            # val_recon_epoch_loss_list.append(val_loss)
            # Plot and add to tensorboard

    total_time = datetime.now() - total_start
    print(f"train completed, total time: {total_time}.")

    # save the model
    torch.save(vqvae_model.state_dict(), model_path / f'{exps}_vqvae_model.pt')
    writer.flush()

    return total_time, dump_path

def main():
    run_no = 3
    # read the config file
    with open('config/config_patches.yaml', 'r') as f:
        cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]
    
    maps = ['MTsat', 'R1', 'R2s_WLS1', 'PD_R2scorr']
    model_net = cfg['model']['net']
    exps = f"normative_{model_net}_run{run_no}"
    print(exps)
    exc_times = []
    for map_type in maps: 
        times = {}   
        # cfg['model']['gamma'] = gamma        
        # cfg['dataset']['patch_size'] = ps
        cfg['dataset']['map_type'] = [map_type]
        cfg['exp_name'] = f'{exps}_{map_type}' #_gamma_{gamma}'
        exc_time, dump_path = full_train_model(cfg)
        times['exp_name'] = cfg['exp_name']  
        times['time'] = exc_time    
        exc_times.append(times)    
        pd.DataFrame(exc_times).to_csv(dump_path.parent.parent / f'{exps}_vqvae_execution_times_.csv', index=False)

if __name__ == '__main__':
    main()