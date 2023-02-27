from datetime import datetime
import os
from glob import glob
from pathlib import Path
this_path = Path().resolve()
import torch
import monai
import torchmetrics

import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml

from dataset.ppmi_dataset import PPMIDataModule
from models.pl_model import Model

def predict_from_ckpt(ckpt_path: Path):

    exp_dir = ckpt_path.parent.parent.parent
    with open(exp_dir.parent /'config_dump.yml', 'r') as f:
        cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]