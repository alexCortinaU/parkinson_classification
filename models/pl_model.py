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
import kornia.losses as losses
from models.medicalnet.model import generate_model
from models.medicalnet.setting import get_def_args

def get_criterions(name: str):
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
    else:
        raise ValueError(f'Unknown loss name: {name}')

def get_optimizer(name: str):
    if name == 'adam':
        return optim.Adam
    if name == 'sgd':
        return optim.SGD
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


class Model(pl.LightningModule):
    def __init__(self, 
                    net, 
                    loss, 
                    learning_rate, 
                    optimizer_class, 
                    n_classes = 2, 
                    in_channels = 1, 
                    sch_patience = 15, 
                    weight_decay = 0.0001,
                    momentum = 0):
        super().__init__()
        self.lr = learning_rate
        self.criterion = get_criterions(loss)
        self.optimizer_class = get_optimizer(optimizer_class)
        self.sch_patience = sch_patience
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.train_acc = torchmetrics.Accuracy(task='binary', validate_args=True)
        self.val_acc = torchmetrics.Accuracy(task='binary', validate_args=True)
        self.train_auroc = torchmetrics.AUROC(task='binary')
        self.val_auroc = torchmetrics.AUROC(task='binary')
        self.train_f1 = torchmetrics.F1Score(task='binary', num_classes=2, average='macro')
        self.val_f1 = torchmetrics.F1Score(task='binary', num_classes=2, average='macro')
        if net == '3dresnet':
            self.net = get_3dresnet(n_classes)
            print('Pretrained 3D resnet has a single input channel')
        else:
            self.net = get_monai_net(net, in_channels, n_classes)
            
    def configure_optimizers(self):
        if self.optimizer_class == optim.Adam:
            optimizer = self.optimizer_class(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = self.optimizer_class(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        sch = ReduceLROnPlateau(optimizer, 'min',
                                factor=0.1, patience=self.sch_patience)
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
        loss = self.criterion(y_hat, y).mean()
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
        loss = self.criterion(y_hat, y).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        self.val_acc(y_hat, y)
        self.val_auroc(y_hat, y)
        self.val_f1(y_hat, y)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True ,prog_bar=True, logger=True, batch_size=batch_size)

        return loss
    
    def forward(self, x):
        x, _ = self.prepare_batch(x)
        return self.net(x)
    
    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     # this calls forward
    #     y_hat, y = self.infer_batch(batch)
    #     return self(batch)
