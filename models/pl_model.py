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
from monai.losses import ContrastiveLoss

class ComputeRE(Callback):
    def __init__(self,
                 input_imgs,
                 locations,
                 sampler,
                 subject,
                 every_n_epochs=1,
                 cohort="control",
                 vae=False):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.locations = locations
        self.sampler = sampler
        self.every_n_epochs = every_n_epochs
        self.cohort = cohort
        self.vae = vae
        self.aggregator = tio.data.GridAggregator(sampler)
        self.og_img = subject['image'][tio.DATA]

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                if self.vae:
                    reconst_imgs, _, _, _ = pl_module(input_imgs)
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

def get_criterions(name: str, gamma: float = 0.9, alpha: float = 0.5):

    if name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    # elif name == 'dice':
    #     return losses.DiceLoss()
    elif name == 'bin_cross_entropy':
        return nn.BCEWithLogitsLoss()
    elif name == 'focal':
        return losses.BinaryFocalLossWithLogits(alpha=alpha, reduction='mean')
    elif name == 'tversky':
        return losses.TverskyLoss(0.4, 0.4)
    
    # reconstruction losses will need to be changed in order to work with multiple channels
    elif name == 'l1':
        return nn.L1Loss()
    elif name == 'mse':
        return nn.MSELoss()
    elif name == 'l1kld':
        return LossL1KLD(gamma=gamma)
    else:
        raise ValueError(f'Unknown loss name: {name}')

def get_optimizer(name: str):
    if name == 'adam':
        return optim.Adam
    if name == 'adamw':
        return optim.AdamW
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
    for m in model.conv_seg.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

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

class Model(pl.LightningModule):
    def __init__(self,
                    net='3dresnet',                  
                    loss='focal',
                    learning_rate=0.001, 
                    optimizer_class='adam',
                    group_params=False,
                    gamma=0.9,
                    alpha=0.5,
                    chkpt_path=None, 
                    n_classes = 2, 
                    in_channels = 1, 
                    sch_patience = 15, 
                    weight_decay = 0.0001,
                    momentum = 0):
        super().__init__()
        self.lr = learning_rate
        self.loss = loss
        self.criterion = get_criterions(loss, gamma=gamma, alpha=alpha)
        self.group_params = group_params
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
                print('Pretrained 3D resnet has a single input channel')
            elif net == None:
                self.net = None
                # print('debugging this this')
            else:
                self.net = get_monai_net(net, in_channels, n_classes)
            
    def configure_optimizers(self):

        params = list(self.named_parameters())
        def is_head(n): return 'conv_seg' in n

        # set different learning rates for the last layer
        if self.group_params:
            group_parameters = [{'params': [p for n, p in params if is_head(n)], 'lr': self.lr * 10},
                                {'params': [p for n, p in params if not is_head(n)], 'lr': self.lr}]
            parameters = group_parameters
        else:
            parameters = self.parameters()
        
        # set optimizer
        if self.optimizer_class == optim.Adam:
            optimizer = self.optimizer_class(parameters, 
                                             lr=self.lr, 
                                             weight_decay=self.weight_decay)
        elif self.optimizer_class == optim.AdamW:
            optimizer = self.optimizer_class(parameters, 
                                             lr=self.lr, 
                                             weight_decay=self.weight_decay)
        else:
            optimizer = self.optimizer_class(parameters, 
                                             lr=self.lr, 
                                             momentum=self.momentum, 
                                             weight_decay=self.weight_decay)
        
        # set learning rate scheduler
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

    def training_step(self, batch, batch_idx):
        
        if self.loss == "l1kld":
            x, x_hat, loss = self.vae_step(batch)
        else:
            x_hat, x = self.infer_batch(batch)                      
            loss = self.criterion(x_hat, x).mean()

        batch_size = len(x)  
        self.log("train_loss", loss, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        self.train_mse(x_hat, x)
        self.log("train_mse", self.train_mse, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
           
        return loss

    def validation_step(self, batch, batch_idx):

        if self.loss == "l1kld":
            x, x_hat, loss = self.vae_step(batch)
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

    def __init__(self, input_imgs, every_n_epochs=1, split="train", vae=False):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs
        self.split = split
        self.vae = vae

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                if self.vae:
                    reconst_imgs, _, _, _ = pl_module(input_imgs)
                else:
                    reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            slice = input_imgs.shape[-1] // 2
            imgs = torch.stack([input_imgs[..., slice].detach(), reconst_imgs[..., slice].detach()], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image(f"Reconstructions {self.split}", grid, global_step=trainer.global_step)

# Contrastive Learning and Downstream Tasks

def get_net(net):
    if net == 'resnet_monai':
        net = monai.networks.nets.ResNet('basic', layers=[1, 1, 1, 1], block_inplanes=[64, 128, 256, 512],
                                    spatial_dims=3, n_input_channels=1, num_classes=2)
    return net

class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, projection_dim):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = self.encoder.fc.in_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = nn.Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j

class ContrastiveLearning(pl.LightningModule):

    def __init__(self, hpdict):
        
        super().__init__()

        self.hpdict = hpdict

        # initialize ResNet
        # self.encoder = get_net(self.hpdict['model']['net'])
        # self.n_features = self.encoder.fc.in_features  # get dimensions of fc layer
        self.model = SimCLR(get_net(self.hpdict['model']['net']), self.hpdict['model']['projection_dim'])
        self.criterion = ContrastiveLoss(self.hpdict['model']['temperature'])

    def forward(self, x_i, x_j):
        h_i, h_j, z_i, z_j = self.model(x_i, x_j)
        loss = self.criterion(z_i, z_j)
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x_i, x_j = batch
        loss = self.forward(x_i, x_j)
        self.log("train_loss", 
                 loss, on_step=False, 
                 on_epoch=True,
                 prog_bar=True, 
                 logger=True, 
                 batch_size=self.hpdict['dataset']['train_batch_size'])
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x_i, x_j = batch
        loss = self.forward(x_i, x_j)
        self.log("val_loss", 
                 loss, on_step=False, 
                 on_epoch=True,
                 prog_bar=True, 
                 logger=True, 
                 batch_size=self.hpdict['dataset']['train_batch_size'])
        return loss

    def configure_criterion(self):
        # criterion = NT_Xent(self.hpdict['dataset']['batch_size'], self.hpdict['model']['temperature'])
        criterion = ContrastiveLoss(temperature=self.hpdict['model']['temperature'])
        return criterion

    def configure_optimizers(self):
        scheduler = None
        if self.hpdict['model']['optimizer_class'] == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hpdict['model']['learning_rate'])
                    # "decay the learning rate with the cosine decay schedule without restarts"
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.hpdict['pl_trainer']['max_epochs'], eta_min=0, last_epoch=-1
            )
        else:
            raise NotImplementedError

        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}

class ClassifierSimCLR(nn.Module):
    def __init__(self, net, num_classes):
        super().__init__()

        n_features = net.projector[0].in_features
        self.feature_extractor = nn.Sequential(*list(net.children())[:-1])

        self.classifier = nn.Sequential(
                            nn.Dropout(0.5),
                            nn.Linear(n_features, 256),
                            nn.ReLU(),
                            nn.Linear(256, 2)
                        )
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
    
class ModelDownstream(Model):
    def __init__(self, 
                 net=None,
                 **kwargs):
        super().__init__(net=net, **kwargs)

        # self.save_hyperparameters(ignore=['net'])

        n_features = self.net.projector[0].in_features
        layers = list(self.net.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
                            nn.Dropout(0.5),
                            nn.Linear(n_features, 256),
                            nn.ReLU(),
                            nn.Linear(256, 2)
                        )

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        # set optimizer
        if self.optimizer_class == optim.Adam:
            optimizer = self.optimizer_class(filter(lambda p: p.requires_grad, self.parameters()), 
                                             lr=self.lr, 
                                             weight_decay=self.weight_decay)
        else:
            optimizer = self.optimizer_class(filter(lambda p: p.requires_grad, self.parameters()), 
                                             lr=self.lr, 
                                             momentum=self.momentum, 
                                             weight_decay=self.weight_decay)
        
        # set learning rate scheduler
        if self.sch_patience > 0:
            sch = ReduceLROnPlateau(optimizer, 'min',
                                    factor=0.1, patience=self.sch_patience)
            #learning rate scheduler
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": sch,
                                    "monitor":"val_loss"}}
        else:
            return optimizer
        
    def forward(self, x):
            x = self.feature_extractor(x)
            y_hat = self.classifier(x)
            return y_hat

    def training_step(self, batch, batch_idx):
            x, y = batch
            batch_size = len(y)
            y_hat = self.forward(x)
            loss = self.criterion(y_hat, y)
            self.log("train_loss", loss, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
            self.train_acc(y_hat, y)
            self.train_auroc(y_hat, y)
            self.train_f1(y_hat, y)
            self.log("train_acc", self.train_acc, on_step=False, on_epoch=True ,prog_bar=False, logger=True, batch_size=batch_size)
            self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
            self.log("train_f1", self.train_f1, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
            
            return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = len(y)
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        self.val_acc(y_hat, y)
        self.val_auroc(y_hat, y)
        self.val_f1(y_hat, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True ,prog_bar=False, logger=True, batch_size=batch_size)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        return loss

# class ModelDownstream(Model):
#     def __init__(self, 
#                  get_encoder=None,
#                  net=None,
#                  **kwargs):
#         super().__init__(net=net, **kwargs)

#         self.save_hyperparameters(ignore=['net'])
#         if get_encoder is not None:
#             self.feature_extractor, n_features = get_encoder(self.net)
#         else:
#             n_features = self.net.projector[0].in_features
#             layers = list(self.net.children())[:-1]
#             self.feature_extractor = nn.Sequential(*layers)
        
#         # self.classifier = nn.Linear(n_features, 2)
#         # nn.init.xavier_uniform_(self.classifier.weight)
#         # nn.init.zeros_(self.classifier.bias)

#         self.classifier = nn.Sequential(
#                             nn.Dropout(0.5),
#                             nn.Linear(n_features, 256),
#                             nn.ReLU(),
#                             nn.Linear(256, 2)
#                         )

#         for m in self.classifier.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0)
    
#     def forward(self, x):
#         self.feature_extractor.eval()
#         with torch.no_grad():
#             representations = self.feature_extractor(x)
#         y_hat = self.classifier(representations)
#         return y_hat
    
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         batch_size = len(y)
#         y_hat = self.forward(x)
#         loss = self.criterion(y_hat, y)
#         self.log("train_loss", loss, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
#         self.train_acc(y_hat, y)
#         self.train_auroc(y_hat, y)
#         self.train_f1(y_hat, y)
#         self.log("train_acc", self.train_acc, on_step=False, on_epoch=True ,prog_bar=False, logger=True, batch_size=batch_size)
#         self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
#         self.log("train_f1", self.train_f1, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
           
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         batch_size = len(y)
#         y_hat = self.forward(x)
#         loss = self.criterion(y_hat, y)
#         self.log("val_loss", loss, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
#         self.val_acc(y_hat, y)
#         self.val_auroc(y_hat, y)
#         self.val_f1(y_hat, y)
#         self.log("val_acc", self.val_acc, on_step=False, on_epoch=True ,prog_bar=False, logger=True, batch_size=batch_size)
#         self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
#         self.log("val_f1", self.val_f1, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
#         return loss

