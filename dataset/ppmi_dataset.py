from datetime import datetime
import os
import tempfile
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch import optim, nn, utils, Tensor, as_tensor
from torch.utils.data import random_split, DataLoader

import monai
import torchmetrics
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import yaml

class PPMIDataModule(pl.LightningDataModule):
    def __init__(self, 
                md_df, 
                root_dir,
                train_batch_size = 4,
                val_batch_size = 4,
                train_num_workers = 4,
                val_num_workers = 4, 
                reshape_size = (128, 128, 128), 
                val_test_split = 0.4, 
                random_state = 42,
                augment = None):
        super().__init__()
        self.md_df = md_df
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.root_dir = root_dir
        self.reshape_size = reshape_size
        self.val_test_split = val_test_split
        self.random_state = random_state
        self.augment = augment
        self.subjects = None
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def get_subjects_list(self, md_df):
        subjects_list = []
        subjects_labels = []
        for i in range(len(md_df)):
            subj = md_df.iloc[i]
            if 'w/' in subj['Description']:
                subj_desc = subj['Description'].replace('w/', 'w_').replace(' ', '_')
            else:
                subj_desc = subj['Description'].replace(' ', '_')
                
            subj_dir = self.root_dir / str(subj['Subject']) / subj_desc
            for f in subj_dir.iterdir():
                if str(subj['Acq Date'].date()) in f.name:
                    subj_dir = f
                    break
            for f in subj_dir.iterdir(): 
                subj_dir = [f for f in f.iterdir()][0]
            subjects_list.append(str(subj_dir))
            if subj['Group'] == 'PD':
                subjects_labels.append(1)
            else:
                subjects_labels.append(0)

        return subjects_list, subjects_labels

    def prepare_data(self):

        # split ratio train = 0.6, val = 0.2, test = 0.2
        self.md_df_train, md_df_rest = train_test_split(self.md_df, test_size=self.val_test_split, 
                                                            random_state=self.random_state, stratify=self.md_df.loc[:, 'Group'].values)
        self.md_df_val, self.md_df_test = train_test_split(md_df_rest, test_size=0.5,
                                                random_state=self.random_state, stratify=md_df_rest.loc[:, 'Group'].values)
                                                
        image_training_paths, labels_train = self.get_subjects_list(self.md_df_train)
        image_val_paths, labels_val = self.get_subjects_list(self.md_df_val)
        image_test_paths, labels_test = self.get_subjects_list(self.md_df_test)

        self.train_subjects = []
        for image_path, label in zip(image_training_paths, labels_train):
            # 'image' and 'label' are arbitrary names for the images
            subject = tio.Subject(image=tio.ScalarImage(image_path), label=nn.functional.one_hot(as_tensor(label), num_classes=2).float())
            self.train_subjects.append(subject)

        self.val_subjects = []
        for image_path, label in zip(image_val_paths, labels_val):
            # 'image' and 'label' are arbitrary names for the images
            subject = tio.Subject(image=tio.ScalarImage(image_path), label=nn.functional.one_hot(as_tensor(label), num_classes=2).float())
            self.val_subjects.append(subject)

        self.test_subjects = []
        for image_path, label in zip(image_test_paths, labels_test):
            subject = tio.Subject(image=tio.ScalarImage(image_path), label=nn.functional.one_hot(as_tensor(label), num_classes=2).float())
            self.test_subjects.append(subject)

    def get_preprocessing_transform(self):
        # Rescales intensities to [0, 1] and adds a channel dimension,
        # then resizes to the desired shape

        # preprocess = Compose(
        #     [ScaleIntensity(), 
        #     EnsureChannelFirst(), 
        #     ResizeWithPadOrCrop(self.reshape_size)
        #     ])
        preprocess = tio.Compose(
            [
                tio.RescaleIntensity((-1, 1)),
                tio.CropOrPad(self.reshape_size),
                # tio.EnsureShapeMultiple(8),  # for the U-Net
                # tio.OneHot(),
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
                            # Compose(
                            # [RandAffine(prob=0.5,
                            #             translate_range=(5, 5, 5),
                            #             rotate_range=(np.pi * 4, np.pi * 4, np.pi *4 ),
                            #             scale_range=(0.15, 0.15, 0.15),
                            #             padding_mode='zeros')
                            # ])

    def setup(self, stage=None):
        
        # Assign train/val datasets for use in dataloaders
        self.preprocess = self.get_preprocessing_transform()
        self.get_augmentation_transform()
        self.transform = tio.Compose([self.preprocess, self.augment])

        self.train_set = tio.SubjectsDataset(self.train_subjects, transform=self.transform)
        self.val_set = tio.SubjectsDataset(self.val_subjects, transform=self.preprocess)
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.preprocess)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.train_batch_size, num_workers=self.train_num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.val_batch_size, num_workers=self.val_num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.val_batch_size, num_workers=self.val_num_workers)