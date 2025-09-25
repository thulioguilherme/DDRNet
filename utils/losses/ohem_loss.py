import torch
import torch.nn as nn
import torch.nn.functional as F


# #{ OhemCrossEntropyLoss

class OhemCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255, threshold=0.7, min_kept=100000):
        super(OhemCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.min_kept = min_kept
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, pred, target):
        pixel_loss = self.criterion(pred, target)

        pixel_loss = pixel_loss.contiguous().view(-1)

        valid_mask = (target != self.ignore_index).view(-1)
        valid_loss = pixel_loss[valid_mask]

        if valid_loss.numel() > self.min_kept:
            hard_loss, _ = torch.topk(valid_loss, k=self.min_kept, largest=True)
            loss = torch.mean(hard_loss)
        elif valid_loss.numel() > 0:
            loss = torch.mean(valid_loss)
        else:
            loss = torch.tensor(0.0).to(pred.device) # No valid pixels to train on

        return loss

# #}
