from datetime import datetime
import os
import tempfile
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch import optim, nn, utils, Tensor, as_tensor
from torch.utils.data import random_split, DataLoader
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
                agument = None):
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
        self.agument = agument
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
                tio.RescaleIntensity((0, 1)),
                tio.CropOrPad(self.reshape_size),
                # tio.EnsureShapeMultiple(8),  # for the U-Net
                # tio.OneHot(),
            ]
        )
        return preprocess

    def get_augmentation_transform(self):

        # If no augmentation is specified, use the default one
        if self.agument == None:
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

def main():
    # read the config file
    with open('config2.yaml', 'r') as f:
        cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

    # read metadata file and get the first scan for each subject
    root_dir = Path("/mnt/scratch/7TPD/mpm_run_acu/PPMI")
    md_df = pd.read_csv(root_dir/'t1_3d_3t_1mm_pdhc_2_16_2023.csv')
    md_df['Acq Date'] = md_df['Acq Date'].apply(pd.to_datetime)
    md_df.sort_values(by='Acq Date', inplace=True)
    first_acq_idx = md_df.duplicated(subset=['Subject'])
    md_df_first = md_df.loc[~first_acq_idx, :]

    # create PPMI dataset
    data = PPMIDataModule(md_df=md_df_first, root_dir=root_dir, **cfg['dataset'])
    data.prepare_data()
    data.setup()
    print("Training:  ", len(data.train_set))
    print("Validation: ", len(data.val_set))
    print("Test:      ", len(data.test_set))

    model = Model(**cfg['model'])

    # create callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=5,
                                          monitor="val_acc",
                                          mode="max",
                                          filename="{epoch:02d}-{val_acc:.4f}")

    early_stopping = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", 
                                                                mode='min', patience=15,
                                                                )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    # create loggers
    tb_logger = TensorBoardLogger(save_dir=Path('./outputs'),
                               name=cfg['exp_name'],
                               version=0
                               )
    
    csv_logger = CSVLogger(save_dir=Path('./outputs'),
                            flush_logs_every_n_steps=10,
                            name=cfg['exp_name'],
                            version=0
                            )
                            
    # save the config file to the output folder
    # for a given experiment
    dump_path = Path('./outputs').resolve() / f'{cfg["exp_name"]}'
    dump_path.mkdir(parents=True, exist_ok=True)
    dump_path = dump_path/'config_dump.yml'
    with open(dump_path, 'w') as f:
        yaml.dump(cfg, f)

    # create trainer
    trainer = pl.Trainer(**cfg['pl_trainer'],
                        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
                        logger=[tb_logger, csv_logger],
                        )

    # # find optimal learning rate
    print('Default LR: ', model.lr)
    trainer.tune(model, datamodule=data)
    print('Tuned LR: ', model.lr)

    start = datetime.now()
    print("Training started at", start)
    trainer.fit(model=model, datamodule=data)
    print("Training duration:", datetime.now() - start)

if __name__ == "__main__":
    main()