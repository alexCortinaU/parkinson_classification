# Binary classification strategy

### This strategy comprises the investigation of different pre-training strategies at different volume masking levels. 

- **train.py**, training script for 3 and 4 experiments A,B, or C. (p1_hmri), uses config.yaml

- **train_cv.py**, training script for 4A experiment using cross-validation, uses config.yaml

- **ppmi_pretrain.py**, training script for PPMI pre-training (p1_ppmi). Paths are wrong!, uses config_ppmi.yaml

- **ssl_pretrain.py**, training script for SimCLR pre-training, uses configssl.yaml

- **ssl_downstream_train.py**, training script for classification task using SLL pre-trained models (p4_downstream_hmri), uses config_downstream.yaml

- **inference.py**, from pre-trained models perform classification inference for the train or validation set, and compute performance metrics (auroc, acc, f1).

- **xai_captum.py**, with a pre-trained model perform inference for a determined subject and compute occlusion sensitivity and/or integrated gradients, with tunable parameters each.

- **proc_xai_maps.py**, notebook for group analysis of XAI maps.