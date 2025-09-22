from thop import profile, clever_format

import torch

from .segmentation.DDRNet23s import DDRNet23s, SequentialResidualBlock


class DDRNetFactory():
    def getModel(self, model_name):
        model = None
        if model_name == 'DDRNet23s':
            model = DDRNet23s(
                SequentialResidualBlock,
                [2, 2, 2, 2],
                num_classes=19,
                channels=32,
                spp_channels=128,
                head_channels=64
            )

        return model
