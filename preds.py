# import torch
# import torch.nn.functional as F

# from PIL import Image
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
# import torchio as tio

# import os
# import json
# import numpy as np
# import copy
# from matplotlib.colors import LinearSegmentedColormap

# import torchvision
# from torchvision import models
# from torchvision import transforms

# from captum.attr import IntegratedGradients
# from captum.attr import GradientShap
# from captum.attr import Occlusion
# from captum.attr import NoiseTunnel
# from captum.attr import visualization as viz
# from captum.attr import LRP

# import yaml
# from dataset.hmri_dataset import HMRIDataModule, HMRIDataModuleDownstream
# from models.pl_model import Model, ContrastiveLearning, ModelDownstream
# from utils.utils import get_pretrained_model
# from utils.utils import save_nifti_from_array
# this_path = Path().resolve()

# def get_preds(map_type):

#     maps = {
#         'R2_WLS1': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4A_hMRI_R2s_WLS1_optim_adam_lr_0.01/version_0/checkpoints/epoch=60-val_auroc=0.9388.ckpt'),
#         'MTsat': Path("/mrhome/alejandrocu/Documents/parkinson_classification/p4_downstream_outs/5B_hMRI_MTsat_optim_adam_lr_0.001_ufrz_15/version_0/checkpoints/epoch=76-val_auroc=tensor(0.7832, device='cuda:0').ckpt"),
#         'R1': Path("/mrhome/alejandrocu/Documents/parkinson_classification/p4_downstream_outs/5B_hMRI_R1_optim_adam_lr_0.001_ufrz_15/version_0/checkpoints/epoch=54-val_auroc=tensor(0.8316, device='cuda:0').ckpt"),
#         'PD_R2scorr': Path("/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4A_hMRI_PD_R2scorr_optim_adam_lr_0.001/version_0/checkpoints/epoch=65-val_auroc=0.8163.ckpt")
#         }
    
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#     ckpt_path = maps[map_type]
#     exp_dir = ckpt_path.parent.parent.parent
#     with open(exp_dir /'config_dump.yml', 'r') as f:
#         cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

#     root_dir = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI')
#     md_df = pd.read_csv(this_path/'bids_3t.csv')

#     data = HMRIDataModuleDownstream(root_dir=root_dir,
#                             md_df=md_df,
#                             **cfg['dataset'])
#     data.prepare_data()
#     data.setup()

#     # model = Model.load_from_checkpoint(ckpt_path, **cfg['model'])

#     chkpt_path = Path(cfg['model']['chkpt_path'])
#     exp_dir = chkpt_path.parent.parent.parent
#     with open(exp_dir /'config_dump.yml', 'r') as f:
#         exp_cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

#     pretrained_model = ContrastiveLearning.load_from_checkpoint(chkpt_path, hpdict=exp_cfg)
#     model = ModelDownstream.load_from_checkpoint(ckpt_path, net=pretrained_model.model, **cfg['model'])

#     input, target = data.val_set[0] #[11]['image'][tio.DATA], data.val_set[11]['label']
#     input = input.to(device).unsqueeze(0)
#     df = data.md_df_val

#     model = model.to(device)
#     # with torch.no_grad():
#     #     output = model(input)
    
#     # print(output)
#     preds_final = []
#     preds_scores = []
#     for i in range(len(df)):
#         input, target = data.val_set[i] # ['image'][tio.DATA], data.val_set[i]['label']
#         print(df.iloc[i]['id'], target)
#         input = input.unsqueeze(0).to(device)

#         with torch.no_grad():
#             output = model(input)
#             # print(f"model's logits output: {output}")
#         output = F.softmax(output, dim=1)
#         prediction_score, pred_label_idx = torch.topk(output, 1)
#         print(pred_label_idx, prediction_score)
#         preds_final.append(prediction_score.cpu().numpy()[0][0])
#         preds_scores.append(pred_label_idx.cpu().numpy()[0][0])
#         # break
#     df['preds'] = preds_final
#     df['preds_scores'] = preds_scores

#     df.to_csv(f'{map_type}_preds.csv', index=False)

# def main():
#     maps = ['MTsat', 'R1']
#     for map_type in maps:
#         get_preds(map_type)

# if __name__ == '__main__':
#     main()

import torch
import torch.nn.functional as F

from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torchio as tio

import os
import json
import numpy as np
import copy
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms

import yaml
from dataset.hmri_dataset import HMRIDataModule
from models.pl_model import Model
from utils.utils import get_pretrained_model
from utils.utils import save_nifti_from_array
this_path = Path().resolve()

def main():
    # subj_idx = 11
    map_type = 'PD_R2scorr'
    maps = {
        'R2_WLS1': Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4A_hMRI_R2s_WLS1_optim_adam_lr_0.01/version_0/checkpoints/epoch=60-val_auroc=0.9388.ckpt'),
        'MTsat': Path("/mrhome/alejandrocu/Documents/parkinson_classification/p4_downstream_outs/5B_hMRI_MTsat_optim_adam_lr_0.001_ufrz_15/version_0/checkpoints/epoch=76-val_auroc=tensor(0.7832, device='cuda:0').ckpt"),
        'R1': Path("/mrhome/alejandrocu/Documents/parkinson_classification/p4_downstream_outs/5B_hMRI_R1_optim_adam_lr_0.001_ufrz_15/version_0/checkpoints/epoch=54-val_auroc=tensor(0.8316, device='cuda:0').ckpt"),
        'PD_R2scorr': Path("/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4A_hMRI_PD_R2scorr_optim_adam_lr_0.001/version_0/checkpoints/epoch=65-val_auroc=0.8163.ckpt")
        }
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ckpt_path = maps[map_type]
    device = 'cuda:0'
    # ckpt_path = Path('/mrhome/alejandrocu/Documents/parkinson_classification/new_p1_hmri_outs/4A_hMRI_R2s_WLS1_optim_adam_lr_0.01/version_0/checkpoints/epoch=60-val_auroc=0.9388.ckpt')
    exp_dir = ckpt_path.parent.parent.parent
    with open(exp_dir /'config_dump.yml', 'r') as f:
        cfg = list(yaml.load_all(f, yaml.SafeLoader))[0]

    root_dir = Path('/mnt/projects/7TPD/bids/derivatives/hMRI_acu/derivatives/hMRI')
    md_df = pd.read_csv(this_path/'bids_3t.csv')
    data = HMRIDataModule(md_df=md_df, root_dir=root_dir, **cfg['dataset']) #shuffle=False

    # loading model from checkpoint
    model = Model.load_from_checkpoint(ckpt_path, **cfg['model'])
    network = copy.deepcopy(model.net)
    network.eval()
    # create dataset
    data.prepare_data()
    data.setup()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ckpt_path = maps[map_type]
    df = data.md_df_val

    # prediction of all
    network = network.to(device)
    preds_final = []
    preds_scores = []
    for i in range(len(df)):
        input, target = data.val_set[i]['image'][tio.DATA], data.val_set[i]['label']
        print(df.iloc[i]['id'], target)
        input = input.unsqueeze(0).to(device)

        with torch.no_grad():
            output = network(input)
            # print(f"model's logits output: {output}")
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)
        print(pred_label_idx, prediction_score)
        preds_final.append(prediction_score.cpu().numpy()[0][0])
        preds_scores.append(pred_label_idx.cpu().numpy()[0][0])
        # break
    df['preds'] = preds_final
    df['preds_scores'] = preds_scores

    df.to_csv(f'{map_type}_preds.csv', index=False)


if __name__ == '__main__':
    main()