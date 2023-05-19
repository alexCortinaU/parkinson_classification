from datetime import datetime
import os
import tempfile
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch import optim, nn, utils, Tensor, as_tensor
from torch.utils.data import random_split
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
import torch
import yaml
from dataset.hmri_dataset import HMRIDataModule
from models.pl_model import Model, get_3dresnet
from utils.utils import get_pretrained_model
from monai.data import ImageDataset, DataLoader
from monai.transforms import (
    LoadImage,
    ResizeWithPadOrCrop,
    RandCoarseShuffle,
    RandCoarseDropout,
    RandSpatialCropSamples,
    RandFlip,
    RandAffine,
    OneOf,
    EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
)
this_path = Path().resolve()
from monai.losses import ContrastiveLoss

# SimCLR
# from simclr import SimCLR
# from simclr.modules import NT_Xent

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
        scheduler = True
        if self.hpdict['model']['optimizer_class'] == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hpdict['model']['learning_rate'])
                    # "decay the learning rate with the cosine decay schedule without restarts"
        elif self.hpdict['model']['optimizer_class'] == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hpdict['model']['learning_rate'], momentum=0.9)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.hpdict['pl_trainer']['max_epochs'], eta_min=0, last_epoch=-1
            )
            print("Using SGD and scheduler")
        else:
            raise NotImplementedError

        if scheduler and self.hpdict['model']['optimizer_class'] == "sgd":
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}

class HMRIDataCLR(pl.LightningDataModule):
    def __init__(self, 
                md_df, 
                root_dir,
                train_batch_size = 4,
                val_batch_size = 4,
                train_num_workers = 4,
                val_num_workers = 4, 
                reshape_size = (128, 128, 128), 
                test_split = 0.2, 
                random_state = 42,
                map_type = ['MTsat'],
                windowed_dataset = False,
                masked = 'brain_masked',
                weighted_sampler = False,
                augment = None,
                shuffle = True):
        super().__init__()
        self.md_df = md_df
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.root_dir = root_dir
        self.reshape_size = reshape_size
        self.test_split = test_split
        self.random_state = random_state
        self.map_type = map_type
        self.windowed_dataset = windowed_dataset
        self.masked = masked
        self.weighted_sampler = weighted_sampler
        self.augment = augment
        self.shuffle = shuffle
        self.subjects = None
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def get_subjects_list(self, md_df):

        subjects_list = []
        # subjects_labels = []
        md_df.reset_index(inplace=True, drop=True)

        for i in range(len(md_df)):

            # select the correct folder of masked volumes
            subj_dir = self.root_dir / md_df['id'][i] / 'Results' / self.masked
            subj_dir.exists()
            # get all windowed nifti volumes
            hmri_files = sorted(list(subj_dir.glob('*_w.nii')), key=lambda x: x.stem)

            # get only maps of interest
            hmri_files = [x for x in hmri_files if any(sub in x.stem for sub in self.map_type)]

            subjects_list.append(str(hmri_files[0]))
            # subjects_labels.append(md_df['group'][i])

        return subjects_list #, subjects_labels

    def prepare_data(self):

        # split ratio train = 0.6, val = 0.2, test = 0.2

        # drop subject 058 because it doesn't have maps
        # 'sub-016' has PD* map completely black
        # sub-025 has no brain mask
        subjs_to_drop = ['sub-058', 'sub-016']
        # if self.brain_masked:
        #     subjs_to_drop.append('sub-025')

        for drop_id in subjs_to_drop:
            self.md_df.drop(self.md_df[self.md_df.id == drop_id].index, inplace=True)
        self.md_df.reset_index(drop=True, inplace=True)
        print(f'Drop subjects {subjs_to_drop}')

        self.md_df_train, self.md_df_val = train_test_split(self.md_df, test_size=self.test_split, 
                                                            random_state=42, stratify=self.md_df.loc[:, 'group'].values)
        # self.md_df_train, self.md_df_val = train_test_split(md_df_train_, test_size=0.25,
        #                                         random_state=self.random_state, stratify=md_df_train_.loc[:, 'group'].values)
                                                
        self.train_subjects = self.get_subjects_list(self.md_df_train)
        self.val_subjects = self.get_subjects_list(self.md_df_val)

        # self.test_subjects = []
        # for image_path, label in zip(image_test_paths, labels_test):
        #     subject = tio.Subject(image=tio.ScalarImage(image_path), label=nn.functional.one_hot(as_tensor(label), num_classes=2).float())
        #     self.test_subjects.append(subject)

    def get_preprocessing_transform(self):

        preprocess = Compose([
                # LoadImage(),
                EnsureChannelFirst(),
                ScaleIntensity(minv=0, maxv=1),
                ResizeWithPadOrCrop(self.reshape_size, mode='minimum')
            ]
        )
        return preprocess

    def get_augmentation_transform(self):

        # If no augmentation is specified, use the default one
        if self.augment == None:
            self.augment = tio.Compose([
                                        tio.RandomAffine(),
                                        tio.RandomGamma(p=0.5),
                                        tio.RandomNoise(p=0.5),
                                        tio.RandomMotion(p=0.1),
                                        tio.RandomBiasField(p=0.25),
                                        ])

    def setup(self, stage=None):
        
        # Assign train/val datasets for use in dataloaders
        self.preprocess = self.get_preprocessing_transform()
        self.get_augmentation_transform()
        self.transform = self.augment #Compose([self.preprocess, self.augment])

        self.train_set = ImageDataset(image_files=self.train_subjects, 
                                      transform=self.transform,
                                      image_only=True)
        self.val_set = ImageDataset(image_files=self.val_subjects,
                                    transform=self.transform,
                                    image_only=True)
        # self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.preprocess)

    def train_dataloader(self):
        return DataLoader(self.train_set, 
                            batch_size=self.train_batch_size, 
                            num_workers=self.train_num_workers,
                            shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                            batch_size=self.val_batch_size, 
                            num_workers=self.val_num_workers,
                            shuffle=False)

class TransformsSimCLR:

    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, reshape_size):
        
        self.preprocess = Compose([
                # LoadImage(),
                EnsureChannelFirst(),
                ScaleIntensity(minv=0, maxv=1),
                Resize(spatial_size=(reshape_size, reshape_size, reshape_size), mode='trilinear'),
                # ResizeWithPadOrCrop(reshape_size, mode='minimum')
            ]
        )
        self.augment = Compose([
                RandFlip(prob=0.5, spatial_axis=0),
                RandFlip(prob=0.5, spatial_axis=1),
                RandFlip(prob=0.5, spatial_axis=2),
                RandAffine(
                        # rotate_range=((-np.pi/12, np.pi/12), (-np.pi/12, np.pi/12), (-np.pi/12, np.pi/12)), 
                    #    scale_range=((1, 1.1), (1, 1.1), (1, 1.1)),
                       translate_range=((-12, 12), (-12, 12), (-12, 12)),
                       padding_mode="zeros",
                       prob=1, 
                       mode='bilinear'),
                OneOf(
                    transforms=[
                    RandCoarseDropout(prob=1.0, holes=8, spatial_size=5, dropout_holes=True, max_spatial_size=32),
                    RandCoarseDropout(prob=1.0, holes=12, spatial_size=20, dropout_holes=False, max_spatial_size=64),
                    ]
                ),
                RandCoarseShuffle(prob=0.8, holes=20, spatial_size=20)
            ]
        )

        self.train_transform = Compose([self.preprocess, self.augment])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
    
def full_train(cfg):

    # Set data directory
    root_dir = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI')
    md_df = pd.read_csv(this_path/'bids_3t.csv')
    pd_number = len(md_df.loc[md_df.group == 1, :])
    hc_number = len(md_df.loc[md_df.group == 0, :])
    print(f'------------\n Data - PD: {pd_number}, HC: {hc_number}')

    # create PPMI dataset
    augmentations = TransformsSimCLR(reshape_size=cfg['dataset']['reshape_size'])
    data = HMRIDataCLR(root_dir=root_dir,
                            md_df=md_df,
                            augment=augmentations,
                            **cfg['dataset'])
    data.prepare_data()
    data.setup()

    # create model
    clmodel = ContrastiveLearning(cfg)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=5,
                                            monitor=cfg['training']['monitor_ckpt'],
                                            mode="min",
                                            filename="{epoch:02d}-{val_loss}" # problem with val_loss format using Monai's meta tensor
                                            )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    # create loggers
    tb_logger = TensorBoardLogger(save_dir=Path('./p3_ssl_hmri'),
                                name=cfg['exp_name'])

    # save the config file to the output folder
    # for a given experiment
    dump_path = Path('./p3_ssl_hmri').resolve() / f'{cfg["exp_name"]}'
    dump_path.mkdir(parents=True, exist_ok=True)
    dump_path = dump_path/'config_dump.yml'
    with open(dump_path, 'w') as f:
        yaml.dump(cfg, f)
    
    # create trainer
    trainer = pl.Trainer(**cfg['pl_trainer'],
                        callbacks=[checkpoint_callback, lr_monitor],
                        logger=[tb_logger],
                        )

    start = datetime.now()
    # print("Training started at", start)
    trainer.fit(model=clmodel, datamodule=data)
    print("Training duration:", datetime.now() - start)

    return datetime.now() - start, dump_path

def main():

    with open('./config/configssl.yaml', 'r') as f:
        cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

     # set random seed for reproducibility
    pl.seed_everything(cfg['dataset']['random_state'],  workers=True)

    maps = ['MTsat', 'R1', 'R2s_WLS1', 'PD_R2scorr'] # 'MTsat', 'R1', 'R2s_WLS1', 'PD_R2scorr'
    optimizers = ['sgd'] # , 'sgd'
    lrates = [0.01]
    
    exps = 'ssl_hmri'
    exc_times = []
    for optim in optimizers:
        for map_type in maps:  
            for lr in lrates:                                         
                times = {}
                cfg['model']['learning_rate'] = lr
                cfg['model']['optimizer_class'] = optim
                cfg['dataset']['map_type'] = [map_type]
                cfg['exp_name'] = f'{exps}_{map_type}_optim_{optim}_lr_{lr}'

                exc_time, dump_path = full_train(cfg)   

                times['exp_name'] = cfg['exp_name']  
                times['time'] = exc_time    
                exc_times.append(times)
        
    pd.DataFrame(exc_times).to_csv(dump_path.parent.parent/f'{exps}_execution_times_.csv', index=False)


if __name__ == "__main__":
    main()