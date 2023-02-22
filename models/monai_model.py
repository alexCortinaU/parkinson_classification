from datetime import datetime
import os
import tempfile
from glob import glob
from pathlib import Path

from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import monai
import torchmetrics
import torchio as tio
import pytorch_lightning as pl
import numpy as np
import yaml

def get_criterions(name: str):
    if name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    # elif name == 'dice':
    #     return losses.DiceLoss()
    # elif name == 'focal':
    #     return losses.FocalLoss(0.5)
    # elif name == 'tversky':
    #     return losses.TverskyLoss(0.4, 0.4)
    else:
        raise ValueError(f'Unknown loss name: {name}')

def get_optimizer(name: str):
    if name == 'adam':
        return optim.Adam
    # elif name == 'dice':
    #     return losses.DiceLoss()
    # elif name == 'focal':
    #     return losses.FocalLoss(0.5)
    # elif name == 'tversky':
    #     return losses.TverskyLoss(0.4, 0.4)
    else:
        raise ValueError(f'Unknown loss name: {name}')

def get_monai_net(name: str):
    if name == 'densenet':
        return monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)
    elif name == 'efficient':
        return monai.networks.nets.EfficientNetBN('efficientnet-b0', spatial_dims=3, in_channels=1, num_classes=2)
    elif name == 'resnet':
        return monai.networks.nets.ResNet(block='bottleneck', 
                                            layers=[3, 4, 6, 3], spatial_dims=3, n_input_channels=1, num_classes=2)

class Model(pl.LightningModule):
    def __init__(self, net, loss, learning_rate, optimizer_class):
        super().__init__()
        self.lr = learning_rate
        self.net = get_monai_net(net)
        self.criterion = get_criterions(loss)
        self.optimizer_class = get_optimizer(optimizer_class)
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.train_auroc = torchmetrics.AUROC(task='binary')
        self.val_auroc = torchmetrics.AUROC(task='binary')
        self.train_f1 = torchmetrics.F1Score(task='binary', num_classes=2)
        self.val_f1 = torchmetrics.F1Score(task='binary', num_classes=2)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        sch = ReduceLROnPlateau(optimizer, 'min',
                                factor=0.1, patience=10)
         #learning rate scheduler
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": sch,
                                 "monitor":"val_loss"}}

    def prepare_batch(self, batch):
        return batch["image"][tio.DATA], batch["label"]

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        batch_size = len(y)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        self.train_acc(y_hat, y)
        self.train_auroc(y_hat, y)
        self.train_f1(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
           
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        batch_size = len(y)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        self.val_acc(y_hat, y)
        self.val_auroc(y_hat, y)
        self.val_f1(y_hat, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)

        return loss
