import torch
import torch.nn as nn
import torch.nn.functional as F


class OhemCrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index=255, threshold=0.7, min_kept=100000):
        super(OhemCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.min_kept = min_kept
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, pred, target):
        pixel_loss = self.criterion(pred, target).view(-1)

        valid_mask = (target.view(-1) != self.ignore_index)
        valid_loss = pixel_loss[valid_mask]

        if valid_loss.numel() == 0:
            return torch.tensor(0.0).to(pred.device)

        hard_pixels_mask = valid_loss > self.threshold

        if torch.sum(hard_pixels_mask) < self.min_kept:
            k = min(self.min_kept, valid_loss.numel())
            hard_loss, _ = torch.topk(valid_loss, k=k, largest=True)
        else:
            hard_loss = valid_loss[hard_pixels_mask]

        loss = torch.mean(hard_loss)

        return loss
