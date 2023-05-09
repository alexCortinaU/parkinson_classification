from datetime import datetime
import os
import tempfile
from glob import glob
from pathlib import Path
from typing import Optional, Tuple, List

from torch import optim, nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import monai
import torchmetrics
import torchio as tio
import pytorch_lightning as pl
import numpy as np
import yaml
import kornia.losses as losses
from models.medicalnet.model import generate_model
from models.medicalnet.setting import get_def_args
# from utils.utils import get_pretrained_model
from pytorch_lightning.callbacks import Callback
import torchvision
from models.svae import spatialVAE
from monai.networks.layers import Act
from GenerativeModels.generative.networks.nets import VQVAE

# class ReconstructionError(torchmetrics.Metric):
#     def __init__(self, maps: list, **kwargs):
#         super().__init__()
#         for map in maps:
#             self.add_state(map, default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, x: torch.Tensor, x_hat: torch.Tensor):
#         preds, target = self._input_format(x, x_hat)
#         assert preds.shape == target.shape

#         self.correct += torch.sum(preds == target)
#         self.total += target.numel()

#     def compute(self):
#         return self.correct.float() / self.total
#     # Set to True if the metric is differentiable else set to False
#     is_differentiable: Optional[bool] = None

#     # Set to True if the metric reaches it optimal value when the metric is maximized.
#     # Set to False if it when the metric is minimized.
#     higher_is_better: Optional[bool] = True

#     # Set to True if the metric during 'update' requires access to the global metric
#     # state for its calculations. If not, setting this to False indicates that all
#     # batch states are independent and we will optimize the runtime of 'forward'
#     full_state_update: bool = True

class ComputeRE(Callback):
    def __init__(self,
                 input_imgs,
                 locations,
                 sampler,
                 subject,
                 every_n_epochs=1,
                 cohort="control",
                 ae_type='vae'):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.locations = locations
        self.sampler = sampler
        self.every_n_epochs = every_n_epochs
        self.cohort = cohort
        self.ae_type = ae_type
        self.aggregator = tio.data.GridAggregator(sampler)
        self.og_img = subject['image'][tio.DATA]

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                if self.ae_type == 'vae':
                    reconst_imgs, _, _, _ = pl_module(input_imgs)
                elif self.ae_type == 'vqvae':
                    reconst_imgs, _ = pl_module(input_imgs)
                else:
                    reconst_imgs = pl_module(input_imgs)
                pl_module.train()

            # Aggregate patches into image
            self.aggregator.add_batch(reconst_imgs, self.locations)
            reconstructed = self.aggregator.get_output_tensor()

            # Compute reconstruction error
            diff = [torch.pow(self.og_img[i] - reconstructed[i], 2) for i in range(self.og_img.shape[0])]
            rerror = torch.sqrt(torch.sum(torch.stack(diff), dim=0))
            trainer.logger.experiment.add_scalar(f"RE {self.cohort}", torch.mean(rerror), global_step=trainer.global_step)

class LossL1KLD(nn.Module):

    def __init__(self, gamma: float = 0.9):
        super(LossL1KLD, self).__init__()
        self.gamma = gamma
        self.l1criterion = nn.L1Loss(reduction='sum')

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor):

        # Reconstruction loss
        l1 = self.l1criterion(x_hat, x) #_hat, x)
        # KLD loss
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = self.gamma * l1 + (1 - self.gamma) * kld

        return loss

class LossVQVAE(nn.Module):
    def __init__(self):
        super(LossVQVAE, self).__init__()
        self.l1criterion = nn.L1Loss(reduction='mean')

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor, quantization_loss: torch.Tensor):
        l1 = self.l1criterion(x_hat, x)
        # VQVAE loss
        loss = l1/10000 + quantization_loss
        return l1 #loss
    
def get_criterions(name: str, gamma: float = 0.9):

    if name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    # elif name == 'dice':
    #     return losses.DiceLoss()
    elif name == 'bin_cross_entropy':
        return nn.BCEWithLogitsLoss()
    elif name == 'focal':
        return losses.BinaryFocalLossWithLogits(0.5)
    elif name == 'tversky':
        return losses.TverskyLoss(0.4, 0.4)
    
    # reconstruction losses will need to be changed in order to work with multiple channels
    elif name == 'l1':
        return nn.L1Loss()
    elif name == 'mse':
        return nn.MSELoss()
    elif name == 'l1kld':
        return LossL1KLD(gamma=gamma)
    elif name == 'vqvaeloss':
        return LossVQVAE()
    else:
        raise ValueError(f'Unknown loss name: {name}')

def get_optimizer(name: str):
    if name == 'adam':
        return optim.Adam
    if name == 'sgd':
        return optim.SGD
    if name == 'rmsprop':
        return optim.RMSprop
    else:
        raise ValueError(f'Unknown loss name: {name}')

def get_monai_net(name: str, in_channels: int = 1, n_classes: int = 2):
    if name == 'densenet':
        return monai.networks.nets.DenseNet121(spatial_dims=3, 
                                                in_channels=in_channels, 
                                                out_channels=n_classes,
                                                pretrained=True)
    elif name == 'efficient':
        return monai.networks.nets.EfficientNetBN('efficientnet-b0', 
                                                    spatial_dims=3, 
                                                    in_channels=in_channels, 
                                                    num_classes=n_classes,
                                                    pretrained=True)
    elif name == 'resnet':
        return monai.networks.nets.ResNet(block='bottleneck', 
                                            layers=[3, 4, 6, 3], 
                                            spatial_dims=3, 
                                            n_input_channels=in_channels, 
                                            num_classes=n_classes,
                                            block_inplanes=[64, 128, 256, 512],
                                            pretrained=True)
def get_3dresnet(n_classes: int = 2):
    args = get_def_args()
    model, _ = generate_model(args) 
    model.conv_seg = nn.Sequential(
                                nn.AdaptiveMaxPool3d(output_size=(1, 1, 1)),
                                nn.Flatten(start_dim=1),
                                nn.Dropout(0.1),
                                # the last Conv3d layer has out_channels = 512
                                nn.Linear(512, n_classes)
                                )
    return model

def get_autoencoder(net, 
                    channels, 
                    in_channels: int = 1, 
                    ps: int = 128,
                    latent_size: int = 128):
    
    if net == 'autoencoder':
        strides = [2 for _ in range(len(channels))]
        net = monai.networks.nets.AutoEncoder(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=in_channels,
            channels=channels,
            strides=strides,
            norm='BATCH',
            bias=True)
        
        # add activation to the last layer
        net.decode[-1].conv.add_module('adn', monai.networks.blocks.ADN('NDA', 1, act='sigmoid'))

        return net
    
    elif net == 'vae':
        strides = [2 for _ in range(len(channels))]
        net = monai.networks.nets.VarAutoEncoder(
            spatial_dims=3,
            in_shape=(in_channels, ps, ps, ps),
            out_channels=in_channels,
            channels=channels,
            strides=strides,
            latent_size=latent_size,
            norm='BATCH',
            bias=True,
            use_sigmoid=True)
        
        return net
    
    elif net == 'svae':
        # strides = [2 for _ in range(len(channels))]
        # net = spatialVAE(spatial_dims=3,
        #         in_shape=[in_channels, ps, ps, ps],      
        #         out_channels=in_channels,
        #         latent_size=latent_size,
        #         channels=channels,
        #         strides=strides,
        #         norm='BATCH',
        #         bias=True,
        #         use_sigmoid=True)

        net = spatialVAE(spatial_dims=3,
                in_shape=[1, 128, 128, 128],      
                out_channels=1,
                latent_size=128,
                channels=(32, 64, 128),
                strides=(2, 2, 2),
                norm='BATCH',
                bias=True,
                use_sigmoid=True)
        
        return net

    elif net == 'vqvae':
        net = VQVAE(
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

        return net


class Model(pl.LightningModule):
    def __init__(self,
                    net,                  
                    loss,
                    learning_rate, 
                    optimizer_class,
                    gamma=0.9,
                    chkpt_path=None, 
                    n_classes = 2, 
                    in_channels = 1, 
                    sch_patience = 15, 
                    weight_decay = 0.0001,
                    momentum = 0):
        super().__init__()
        self.lr = learning_rate
        self.loss = loss
        self.criterion = get_criterions(loss, gamma=gamma)
        self.optimizer_class = get_optimizer(optimizer_class)
        self.chkpt_path = chkpt_path
        self.sch_patience = sch_patience
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.train_acc = torchmetrics.Accuracy(task='binary', validate_args=True)
        self.val_acc = torchmetrics.Accuracy(task='binary', validate_args=True)
        self.train_auroc = torchmetrics.AUROC(task='binary')
        self.val_auroc = torchmetrics.AUROC(task='binary')
        self.train_f1 = torchmetrics.F1Score(task='binary', num_classes=2, average='macro')
        self.val_f1 = torchmetrics.F1Score(task='binary', num_classes=2, average='macro')

        # create network
        if not isinstance(net, str):
            self.net = net
        else:   
            if net == '3dresnet':
                self.net = get_3dresnet(n_classes)
                # print('Pretrained 3D resnet has a single input channel')
            if net == None:
                self.net = None
            else:
                self.net = get_monai_net(net, in_channels, n_classes)
            
    def configure_optimizers(self):
        if self.optimizer_class == optim.Adam:
            optimizer = self.optimizer_class(self.parameters(), 
                                             lr=self.lr, 
                                             weight_decay=self.weight_decay)
        else:
            optimizer = self.optimizer_class(self.parameters(), 
                                             lr=self.lr, 
                                             momentum=self.momentum, 
                                             weight_decay=self.weight_decay)
        
        if self.sch_patience > 0:
            sch = ReduceLROnPlateau(optimizer, 'min',
                                    factor=0.1, patience=self.sch_patience)
            #learning rate scheduler
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": sch,
                                    "monitor":"val_loss"}}
        else:
            return optimizer

    def prepare_batch(self, batch):
        return batch["image"][tio.DATA], batch["label"]

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        x_hat = self.net(x)
        return x_hat, y

    def training_step(self, batch, batch_idx):
        x_hat, y = self.infer_batch(batch)
        batch_size = len(y)
        loss = self.criterion(x_hat, y).mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        self.train_acc(x_hat, y)
        self.train_auroc(x_hat, y)
        self.train_f1(x_hat, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True ,prog_bar=False, logger=True, batch_size=batch_size)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
           
        return loss

    def validation_step(self, batch, batch_idx):
        x_hat, y = self.infer_batch(batch)
        batch_size = len(y)
        loss = self.criterion(x_hat, y).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        self.val_acc(x_hat, y)
        self.val_auroc(x_hat, y)
        self.val_f1(x_hat, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True ,prog_bar=False, logger=True, batch_size=batch_size)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)

        return loss
    
    def forward(self, x):
        x, _ = self.prepare_batch(x)
        return self.net(x)
    
    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     # this calls forward
    #     x_hat, y = self.infer_batch(batch)
    #     return self(batch)

class Model_AE(Model):
    def __init__(self, 
                    net, 
                    loss, 
                    learning_rate, 
                    optimizer_class, 
                    gamma = 0.1,
                    channels = [32, 64, 128, 256],
                    n_classes = 2, 
                    in_channels = 1, 
                    sch_patience = 15, 
                    weight_decay = 0.0001,
                    momentum = 0,
                    patch_size = 64,
                    latent_size = 128):
        
        super().__init__(net=None,
                         loss=loss, 
                         gamma=gamma,
                         learning_rate=learning_rate, 
                         optimizer_class=optimizer_class,
                         chkpt_path=None,
                         n_classes=n_classes, 
                         in_channels=in_channels, 
                         sch_patience=sch_patience, 
                         weight_decay=weight_decay, 
                         momentum=momentum)
        
        self.channels = channels
        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.net = get_autoencoder(net, 
                                   self.channels, 
                                   in_channels, 
                                   patch_size,
                                   latent_size)

    def infer_batch(self, batch):
        x = batch["image"][tio.DATA]
        x_hat = self.net(x)
        # x_hat = self.net(x) # , mu, logvar, _ 
        return x_hat, x
    
    def vae_step(self, batch):
        x = batch["image"][tio.DATA]
        x_hat, mu, logvar, _ = self.net(x)
        loss = self.criterion(x_hat, x, mu, logvar)
        # loss = self.criterion(x_hat, x)
        return x, x_hat, loss
    
    def vqvae_step(self, batch):
        x = batch['image'][tio.DATA]
        x_hat, quant_loss = self.net(images=x)
        loss = self.criterion(x_hat, x, quant_loss)
        return x, x_hat, loss, quant_loss

    def training_step(self, batch, batch_idx):
        
        if self.loss == "l1kld":
            x, x_hat, loss = self.vae_step(batch)
        elif self.loss == "vqvaeloss":
            x, x_hat, loss, quant_loss = self.vqvae_step(batch)
            batch_size = len(x)  
            self.log("train_quant_loss", quant_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        else:
            x_hat, x = self.infer_batch(batch)                      
            loss = self.criterion(x_hat, x).mean()

        batch_size = len(x)            
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.train_mse(x_hat, x)
        self.log("train_mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
           
        return loss

    def validation_step(self, batch, batch_idx):

        if self.loss == "l1kld":
            x, x_hat, loss = self.vae_step(batch)
        elif self.loss == "vqvaeloss":
            x, x_hat, loss, quant_loss = self.vqvae_step(batch)
            batch_size = len(x)  
            self.log("val_quant_loss", quant_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        else:
            x_hat, x = self.infer_batch(batch)          
            loss = self.criterion(x_hat, x).mean()

        batch_size = len(x)  
        self.log("val_loss", loss, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        self.val_mse(x_hat, x)
        self.log("val_mse", self.val_mse, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)

        return loss
    
    def forward(self, x):
        return self.net(x)


class GenerateReconstructions(Callback):
    def __init__(self, input_imgs, every_n_epochs=1, split="train", ae_type='vae'):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs
        self.split = split
        self.ae_type = ae_type

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                if self.ae_type == 'vae':
                    reconst_imgs, _, _, _ = pl_module(input_imgs)
                elif self.ae_type == 'vqvae':
                    reconst_imgs, _ = pl_module(input_imgs)
                else:
                    reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            slice = input_imgs.shape[-1] // 2
            imgs = torch.stack([input_imgs[..., slice].detach(), reconst_imgs[..., slice].detach()], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image(f"Reconstructions {self.split}", grid, global_step=trainer.global_step)