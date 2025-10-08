from segcore.utils import profile_model

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddrnetc23slim import BasicResidual, BilateralFusion, BottleneckResidual, Conv2dBNReLU


# #{ init_weights()

def init_weights(module):
    for m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Sequential,
                                Conv2dBNReLU,
                                BasicResidual,
                                BottleneckResidual,
                                BilateralFusion,
                                SegmentationHead,
                                DAPPM)):
                init_weights(m)
            elif isinstance(m, (nn.ReLU, nn.ReLU6)):
                pass
            else:
                pass

# #}


# #{ load_weights()

def load_weights(model, weights_path):
    weights = torch.load(weights_path)
    model.load_state_dict(weights, strict=False)

    return model

# #}


# #{ SegmentationHead

class SegmentationHead(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, scale_factor=1):
        super(SegmentationHead, self).__init__()

        self.scale_factor = scale_factor

        self.conv0 = Conv2dBNReLU(
            in_channels,
            mid_channels,
            kernel_size=3,
            padding=0,
            bias=False
        )

        self.conv1 = Conv2dBNReLU(
            mid_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=False
        )

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)

        if self.scale_factor != 1:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(
                out,
                size=[height, width],
                mode='bilinear'
            )

        return out

# #}


# #{ Deep Aggregation Pyramid Pooling (DAPPM)

# NOTE: DAPPM is implemented with the sequence BN-ReLU-Conv

class DAPPM(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DAPPM, self).__init__()

        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=1,
                bias=False
            )
        )

        self.scale1_process = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=5,
                stride=2,
                padding=2
            ),

            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=1,
                bias=False
            )
        )

        self.scale1_fuse = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        )

        self.scale2_process = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=9,
                stride=4,
                padding=4
            ),

            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=1,
                bias=False
            )
        )

        self.scale2_fuse = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        )

        self.scale3_process = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=17,
                stride=8,
                padding=8
            ),

            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=1,
                bias=False
            )
        )

        self.scale3_fuse = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        )

        self.scale4_process = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),

            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=1,
                bias=False
            )
        )

        self.scale4_fuse = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        )

        self.compression = nn.Sequential(
            nn.BatchNorm2d(mid_channels * 5),
            nn.ReLU(),
            nn.Conv2d(
                mid_channels * 5,
                out_channels,
                kernel_size=1,
                bias=False
            )
        )

        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            )
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]

        y0 = self.scale0(x)
        # print('out y0', y0.size())

        y1 = self.scale1_process(x)
        y1 = F.interpolate(y1, size=[height, width], mode='bilinear')
        y1 = y1 + y0
        y1 = self.scale1_fuse(y1)
        # print('out y1', y1.size())

        y2 = self.scale2_process(x)
        y2 = F.interpolate(y2, size=[height, width], mode='bilinear')
        y2 = y2 + y1
        y2 = self.scale2_fuse(y2)
        # print('out y2', y2.size())

        y3 = self.scale3_process(x)
        y3 = F.interpolate(y3, size=[height, width], mode='bilinear')
        y3 = y3 + y2
        y3 = self.scale3_fuse(y3)
        # print('out y3', y3.size())

        y4 = self.scale4_process(x)
        y4 = F.interpolate(y4, size=[height, width], mode='bilinear')
        y4 = y4 + y3
        y4 = self.scale3_fuse(y4)
        # print('out y4', y4.size())

        y = torch.cat([y0, y1, y2, y3, y4], dim=1)
        # print('out y', y.size())

        y = self.compression(y)
        # print('out y compression', y.size())

        out = y + self.shortcut(x)

        return out

# #}


# #{ Deep Dual-Resolution Network (DDRNet) for semantic segmentation

class DDRNetS23slim(nn.Module):

    def __init__(
        self,
        num_classes=19,
        mode='train',
        num_channels=32,
        dappm_channels=64,
        segmentation_channels=64
    ):

        super(DDRNetS23slim, self).__init__()

        assert mode in ('train', 'eval')
        self.mode = mode

        # #{ Backbone

        self.stage_conv1 = nn.Sequential(
            Conv2dBNReLU(
                3,
                num_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.stage_conv2 = nn.Sequential(
            Conv2dBNReLU(
                num_channels,
                num_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            BasicResidual(num_channels, num_channels),
            BasicResidual(num_channels, num_channels)
        )

        self.stage_conv3 = nn.Sequential(
            BasicResidual(num_channels, num_channels * 2, stride=2),
            BasicResidual(num_channels * 2, num_channels * 2),
        )

        self.stage_conv4_low_res = BasicResidual(num_channels * 2, num_channels * 4, stride=2)
        self.stage_conv4_high_res = BasicResidual(num_channels * 2, num_channels * 2)
        self.stage_conv4_bilateral_fusion = BilateralFusion(num_channels * 4, num_channels * 2)

        self.stage_conv5_1_low_res = BasicResidual(num_channels * 4, num_channels * 8, stride=2)
        self.stage_conv5_1_high_res = BasicResidual(num_channels * 2, num_channels * 2)
        self.stage_conv5_1_bilateral_fusion = BilateralFusion(
            num_channels * 8,
            num_channels * 2,
            ratio=4
        )
        self.stage_conv5_1_low_bottleneck = BottleneckResidual(num_channels * 8, num_channels * 16)
        self.stage_conv5_1_high_bottleneck = BottleneckResidual(num_channels * 2, num_channels * 4)

        # #}

        self.dappm = DAPPM(num_channels * 16, dappm_channels, num_channels * 4)

        if self.mode == 'train':
            self.extra_segmentation_head = SegmentationHead(
                num_channels * 2,
                segmentation_channels,
                num_classes,
                scale_factor=8
            )

        self.segmentation_head = SegmentationHead(
            num_channels * 4,
            segmentation_channels,
            num_classes,
            scale_factor=8
        )

    def forward(self, x):
        in_width = x.shape[-1]
        in_height = x.shape[-2]

        out = self.stage_conv1(x)
        # print('stage conv1', out.size())

        out = self.stage_conv2(out)
        # print('stage conv2', out.size())

        out = self.stage_conv3(out)
        # print('stage conv3', out.size())

        out_l = self.stage_conv4_low_res(out)
        # print('stage conv4 low res', out_l.size())

        out_h = self.stage_conv4_high_res(out)
        # print('stage conv4 high res', out_h.size())

        out_l, out_h = self.stage_conv4_bilateral_fusion(out_l, out_h)
        # print('stage conv4 low bilateral fusion', out_l.size())
        # print('stage conv4 high bilateral fusion', out_h.size())

        if self.mode == 'train':
            out_extra = self.extra_segmentation_head(out_h)
            # print('out extra segmentation head', out_extra.size())

        out_l = self.stage_conv5_1_low_res(out_l)
        # print('stage conv5_1 low res', out_l.size())

        out_h = self.stage_conv5_1_high_res(out_h)
        # print('stage conv5_1 high res', out_h.size())

        out_l, out_h = self.stage_conv5_1_bilateral_fusion(out_l, out_h)
        # print('stage conv5_1 low bilateral fusion', out_l.size())
        # print('stage conv5_1 high bilateral fusion', out_h.size())

        out_l = self.stage_conv5_1_low_bottleneck(out_l)
        # print('stage conv5_1 low bottleneck', out_l.size())

        out_h = self.stage_conv5_1_high_bottleneck(out_h)
        # print('stage conv5_1 high bottleneck', out_h.size())

        out_dappm = self.dappm(out_l)
        # print('dappm', out_dappm.size())

        out_dappm = F.interpolate(
            out_dappm,
            size=[in_height // 8, in_width // 8],
            mode='bilinear'
        )

        out = self.segmentation_head(out_dappm + out_h)
        # print('out segment head', out.size())

        # out = F.interpolate(
        #     out,
        #     size=[in_height, in_width],
        #     mode='bilinear'
        # )
        # print('out', out.size())

        if self.mode == 'train':
            out = [out, out_extra]

        return out

# #}


if __name__ == '__main__':

    model = DDRNetS23slim(num_classes=19)
    init_weights(model)

    flops, params = profile_model(model, inputs=(1, 3, 1024, 2048))
    print(f'FLOPs: {flops}')
    print(f'Parameters: {params}')

    sys.exit(0)
