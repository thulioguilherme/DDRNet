from thop import profile, clever_format

import torch

from models.ddrnets23slim import DDRNetS23slim, load_weights


class DDRNetFactory():
    def getModel(self, model_name):
        model = None
        if model_name == 'DDRNetS23slim':
            model = DDRNet23s(num_classes=19)

        return model
