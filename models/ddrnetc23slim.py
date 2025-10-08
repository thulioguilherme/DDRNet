from segcore.utils import profile_model

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                                BilateralFusion)):
                init_weights(m)
            elif isinstance(m, (nn.ReLU, nn.ReLU6)):
                pass
            else:
                pass

# #}


# #{ Conv2dBNReLU

class Conv2dBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(Conv2dBNReLU, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=False
        )

        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

# #}


# #{ Basic Residual Block

class BasicResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, use_relu=False):
        super(BasicResidual, self).__init__()

        self.use_relu = use_relu
        self.relu = nn.ReLU()

        self.conv0 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn0 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)

        out += residual

        if self.use_relu:
            out = self.relu(out)

        return out

# #}


# #{ Bottleneck Residual Block

class BottleneckResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, use_relu=True):
        super(BottleneckResidual, self).__init__()
        self.use_relu = use_relu
        self.relu = nn.ReLU()

        self.conv0 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            bias=False
        )
        self.bn0 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        if self.use_relu:
            return out
        else:
            return self.relu(out)

# #}


# #{ Bilateral Fusion

class BilateralFusion(nn.Module):

    def __init__(self, low_channels, high_channels, ratio=2):
        super(BilateralFusion, self).__init__()

        self.ratio = ratio

        self.relu = nn.ReLU()

        self.low_basic_residual = BasicResidual(low_channels, low_channels)
        self.high_basic_residual = BasicResidual(high_channels, high_channels)

        self.low_downsample = nn.Sequential(
            nn.Conv2d(
                low_channels,
                high_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(high_channels)
            # no ReLU
        )

        self.high_conv = nn.Sequential(
            nn.Conv2d(
                high_channels,
                low_channels,
                kernel_size=1,
                stride=ratio,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(low_channels)
            # no ReLU
        )

    def forward(self, x_low, x_high):
        low_residual = self.low_basic_residual(x_low)
        out_low = self.low_downsample(low_residual)
        out_low = F.interpolate(
            out_low,
            scale_factor=self.ratio,
            mode='bilinear',
            align_corners=True
        )

        high_residual = self.high_basic_residual(x_high)
        out_high = self.high_conv(high_residual)

        tmp = out_low
        out_low = low_residual + out_high
        out_high = high_residual + tmp

        out_low = self.relu(out_low)
        out_high = self.relu(out_high)

        return out_low, out_high

# #}


# #{ Deep Dual-Resolution Network (DDRNet) for classification

class DDRNetC23slim(nn.Module):

    def __init__(self, num_channels=32, num_classes=1000):
        super(DDRNetC23slim, self).__init__()

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

        self.stage_conv5_2_high_downsample = nn.Sequential(
            Conv2dBNReLU(
                num_channels * 4,
                num_channels * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.Conv2d(
                num_channels * 8,
                num_channels * 16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_channels * 16)
        )

        self.stage_conv5_2_conv = nn.Sequential(
            Conv2dBNReLU(
                num_channels * 16,
                num_channels * 32,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
        )

        self.stage_conv5_2_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.stage_conv5_2_linear = nn.Linear(num_channels * 32, num_classes)

    def forward(self, x):
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

        out_h = self.stage_conv5_2_high_downsample(out_h)
        # print('stage conv5_2 high downsample', out_h.size())

        out = out_l + out_h
        # print('stage conv5_2 cat', out.size())

        out = self.stage_conv5_2_conv(out)
        # print('stage conv5_2 conv', out.size())

        out = self.stage_conv5_2_avgpool(out)
        # print('stage conv5_2 GAP', out.size())

        out = out.view(out.size(0), -1)
        out = self.stage_conv5_2_linear(out)
        # print('stage conv5_2 linear', out.size())

        return out
# #}


if __name__ == '__main__':

    model = DDRNetC23slim(num_classes=1000)
    init_weights(model)

    flops, params = profile_model(model, inputs=(1, 3, 224, 224))
    print(f'FLOPs: {flops}')
    print(f'Parameters: {params}')

    sys.exit(0)
