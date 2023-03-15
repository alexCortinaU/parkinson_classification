from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import nibabel as nib
import SimpleITK as sitk
from typing import List

def plot_imgs(images: List, labels= None, slice = 100):
    n_imgs = len(images)
    if labels == None:
        labels = [f'img_{i}' for i in range(n_imgs)]
    f,axs = plt.subplots(1, n_imgs, figsize=(4*n_imgs, 8))
    for i,ax in enumerate(axs.flat):
        ax.imshow(sitk.GetArrayFromImage(images[i])[slice, :, :], cmap='gray', origin='lower')
        ax.set_title(labels[i], fontdict={'size':26})
        ax.axis('off')
        plt.tight_layout()
    plt.show()

def plot_sitk_img(img: sitk.Image, 
                  slice: int = 80,
                  title: str = None,
                  cmap: str = 'gray'):
    """
    Plot a SimpleITK image
    """
    plt.imshow(sitk.GetArrayFromImage(img)[slice], cmap=cmap)
    plt.title(title)
    plt.show()

def save_sitk_from_nda(nda: np.ndarray,
                       path: Path,
                       img: sitk.Image):
    """
    Save a nifti file from a numpy array and data from original nifti
    """
    nifti = sitk.GetImageFromArray(nda)
    nifti.SetSpacing(img.GetSpacing())
    nifti.SetOrigin(img.GetOrigin())
    nifti.SetDirection(img.GetDirection())
    sitk.WriteImage(nifti, str(path))