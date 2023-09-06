# Normative modeling strategy

### This strategy comprises the investigation of different autoencoder models, the reconstruction of images and the generation of reconstruction error (RE) maps. As  well as the analysis of the RE maps for the classification task. 

- **ae_train.py**, training script for sAE and sVAE normative models (training using only healthy controls), uses config_patches.yaml (p2_hmri)

- **ae_train_cv.py**, training script for cross-validation sAE and sVAE models, uses config_patches.yaml (p2_hmri_cv)

- **vqvae_train.py**, specific training script for vqvae model **[¹]**, since pytorch lightning didn't work for it.

- **get_re_maps.py**, script to generate reconstructions from healthy controls and PD patients, from a checkpoint path. Also, it computes the defined RE maps per subject and saves all results as nifti files. **[¹]**

- **proc_re_maps.py**, notebook for RE maps processing and data analysis. It contains utilitary functions and the intermediate steps of the pipeline for single split and cross-validation analysis.

**[¹]** It requires the installation (or repo copy) from MONAI's Generative Models package.