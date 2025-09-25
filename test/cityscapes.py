import torch
from torch.utils.data import DataLoader

from torchvision.datasets import Cityscapes
from torchvision import transforms

from tqdm import tqdm

import datetime
import os
import sys
import yaml

# #{ include this project packages

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)

# #}

from models.ddrnets23slim import DDRNetS23slim, load_weights
from datasets.cityscapes import Cityscapes
from utils.model_evaluator import ModelEvaluator


if __name__=='__main__':

    home_path = os.path.expanduser('~')
    root_path = os.path.join(home_path, '../app/data/cityscapes')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ddrnets23slim = DDRNetS23slim(num_classes=19)
    ddrnets23slim = load_weights(ddrnets23slim, '../train/ddrnets23slim_dummy.pth')

    val_dataset = Cityscapes(root=root_path, mode='val')

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )

    ddrnets23slim = ddrnets23slim.to(device)
