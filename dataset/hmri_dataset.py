from datetime import datetime
import os
import tempfile
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch import optim, nn, utils, Tensor, as_tensor
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler

import torch
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import yaml

class HMRIDataModule(pl.LightningDataModule):
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
                windowed_dataset = False,
                brain_masked = False,
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
        self.windowed_dataset = windowed_dataset
        self.brain_masked = brain_masked
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
        subjects_labels = []
        for i in range(len(md_df)):
            subj_dir = self.root_dir / md_df.iloc[i, 0] / 'Results'
            if self.windowed_dataset:
                if self.brain_masked:
                    hmri_files = sorted(list(subj_dir.glob('*w_masked.nii')), key=lambda x: x.stem)
                else:
                    hmri_files = sorted(list(subj_dir.glob('*_w.nii')), key=lambda x: x.stem)
            else:
                hmri_files = sorted(list(subj_dir.glob('*.nii')), key=lambda x: x.stem)
                hmri_files = [file for file in hmri_files if '_w' not in file.stem]
            subjects_list.append(hmri_files)
            subjects_labels.append(md_df.iloc[i, -1])

        return subjects_list, subjects_labels

    def prepare_data(self):

        # split ratio train = 0.6, val = 0.2, test = 0.2

        # drop subject 058 because it doesn't have maps
        subjs_to_drop = ['sub-058', 'sub-016']
        if self.brain_masked:
            subjs_to_drop.append('sub-025')
        for drop_id in subjs_to_drop: # 'sub-016'
            self.md_df.drop(self.md_df[self.md_df.id == drop_id].index, inplace=True)
        self.md_df.reset_index(drop=True, inplace=True)
        print(f'Drop subjects {subjs_to_drop}')

        md_df_train_, self.md_df_test = train_test_split(self.md_df, test_size=self.test_split, 
                                                            random_state=42, stratify=self.md_df.loc[:, 'group'].values)
        self.md_df_train, self.md_df_val = train_test_split(md_df_train_, test_size=0.25,
                                                random_state=self.random_state, stratify=md_df_train_.loc[:, 'group'].values)
                                                
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
            [   tio.ToCanonical(),
                tio.RescaleIntensity((0, 1)),
                tio.CropOrPad(self.reshape_size, 
                              padding_mode='minimum')
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
        self.transform = tio.Compose([self.preprocess, self.augment])

        self.train_set = tio.SubjectsDataset(self.train_subjects, transform=self.transform)
        self.val_set = tio.SubjectsDataset(self.val_subjects, transform=self.preprocess)
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.preprocess)

    def train_dataloader(self):

        if self.weighted_sampler:
            targets = self.md_df_train['group'].values
            class_sample_count = np.array(
                [len(np.where(targets == t)[0]) for t in np.unique(targets)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in targets])
            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            return DataLoader(self.train_set,
                            self.train_batch_size,
                            num_workers=self.train_num_workers,
                            sampler=sampler)
        else:
            return DataLoader(self.train_set, 
                            self.train_batch_size, 
                            num_workers=self.train_num_workers,
                            shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_set, 
                            self.val_batch_size, 
                            num_workers=self.val_num_workers,
                            shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, 
                            self.val_batch_size, 
                            num_workers=self.val_num_workers,
                            shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.val_set, 
                            self.val_batch_size, 
                            num_workers=self.val_num_workers,
                            shuffle=False)