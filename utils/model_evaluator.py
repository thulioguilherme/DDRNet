import numpy as np

from pathlib import Path

import sys

import torch
from torch.utils.data import DataLoader

# #{ include this project packages

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# #}

from models.ddrnets23slim import DDRNetS23slim
from utils.datasets.cityscapes import Cityscapes

class_id_mapping = [7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

# #{ ModelEvaluator

class ModelEvaluator(object):

    def __init__(self, num_classes, device, ignore_label=255):
        self.num_classes = num_classes
        self.device = device
        self.ignore_label = ignore_label

    def __call__(self, model, dataset_loader):
        # set model to evaluation mode
        model.eval()

        hist = torch.zeros(self.num_classes, self.num_classes).to(self.device).detach()

        with torch.no_grad():
            for i, (images, labels) in enumerate(dataset_loader):

                labels = labels.squeeze(1).to(self.device)
                size = labels.size()[-2:]

                images = images.to(self.device)

                probs = model(images)

                if isinstance(probs, list):
                    probs = torch.softmax(probs[0], dim=1) # ignore auxiliary loss
                else:
                    probs = torch.softmax(probs, dim=1)

                preds = torch.argmax(probs, dim=1)

                keep = labels != self.ignore_label
                hist += torch.bincount(
                    labels[keep] * self.num_classes + preds[keep],
                    minlength=self.num_classes ** 2
                    ).view(self.num_classes, self.num_classes).float()

        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())

        ious_by_class = {}
        for class_id, iou in enumerate(ious.cpu().numpy()):
            class_train_id = class_id_mapping[class_id]
            if np.isnan(iou):
                ious_by_class[class_train_id] = float('nan')
            else:
                ious_by_class[class_train_id] = iou

        valid_ious = ious[~torch.isnan(ious)]

        if valid_ious.numel() > 0:
            mean_iou = valid_ious.mean().item()
        else:
            mean_iou = 0.0

        return {'mean_iou': mean_iou, 'per_class_iou': ious_by_class}

# #}

if __name__ == '__main__':

    home_path = Path.home()
    data_path = home_path / '../app/data/cityscapes'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} as device')

    ddrnets23slim = DDRNetS23slim(num_classes=19, mode='train')
    ddrnets23slim = ddrnets23slim.to(device)

    val_dataset = Cityscapes(root=data_path, partition='val')

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=8,
        drop_last=True
    )

    model_evaluator = ModelEvaluator(num_classes=19, device=device)
    res = model_evaluator(ddrnets23slim, val_loader)
    print('Mean IoU:', res['mean_iou'])
