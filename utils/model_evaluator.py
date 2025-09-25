import numpy as np

from pathlib import Path

import os, sys

import torch

from torch.utils.data import DataLoader


# #{ include this project packages

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.append(project_root)

# #}

from models.ddrnets23slim import DDRNetS23slim, load_weights
from datasets.cityscapes import Cityscapes

class_id_mapping = [7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]

# #{ ModelEvaluator

class ModelEvaluator(object):

    def __init__(self, device=None, ignore_label=255):
        self.device = device
        self.ignore_label = ignore_label

    def __call__(self, model, dataset_loader, num_classes):
        # Set model to evaluation mode
        model.eval()

        hist = torch.zeros(num_classes, num_classes).to(self.device).detach()

        with torch.no_grad():
            for i, (images, labels) in enumerate(dataset_loader):

                labels = labels.squeeze(1).to(self.device)
                size = labels.size()[-2:]

                images = images.to(self.device)

                probs = model(images)

                probs = torch.softmax(probs, dim=1)
                preds = torch.argmax(probs, dim=1)

                keep = labels != self.ignore_label
                hist += torch.bincount(
                    labels[keep] * num_classes + preds[keep],
                    minlength=num_classes ** 2
                    ).view(num_classes, num_classes).float()

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
    model_weights_path = Path('../train/ddrnets23slim_dummy.pth')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ddrnets23slim = DDRNetS23slim(num_classes=19, mode='val')
    ddrnets23slim = load_weights(ddrnets23slim, model_weights_path, is_backbone_weights=False)
    ddrnets23slim = ddrnets23slim.to(device)

    val_dataset = Cityscapes(root=data_path, mode='val')

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    model_evaluator = ModelEvaluator(device=device)

    res = model_evaluator(ddrnets23slim, val_loader, num_classes=19)
    print('Mean IoU:', res['mean_iou'])
    print('IoU per class:', res['per_class_iou'])
