from thop import profile, clever_format

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from collections import OrderedDict


# #{ Sequential Residual Block (RB)

class SequentialResidualBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, no_relu=False):
        super(SequentialResidualBlock, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        # self.conv1 = conv3x3(in_channels, channels, stride)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # self.conv2 = conv3x3(channels, channels)
        # self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv_layers(x)

        #out = self.conv1(x)
        ##out = self.bn1(out)
        #out = self.relu(out)

        #out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

# #}

# #{ Residual Bottleneck Block (RBB)

class ResidualBottleneckBlock(nn.Module):

    expansion = 2

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, no_relu=True):
        super(ResidualBottleneckBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=True
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

# #}

# #{ DAPPM

class DAPPM(nn.Module):
    def __init__(self, in_channels, branch_channels, out_channels):
        super(DAPPM, self).__init__()

        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=1,
                bias=False
            )
        )

        self.scale1 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=1,
                bias=False
            )
        )

        self.scale2 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=9,
                stride=4,
                padding=4
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=1,
                bias=False
            )
        )

        self.scale3 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=17,
                stride=8,
                padding=8
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=1,
                bias=False)
        )

        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=1,
                bias=False
            )
        )

        self.process1 = nn.Sequential(
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        )

        self.process2 = nn.Sequential(
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        )

        self.process3 = nn.Sequential(
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        )

        self.process4 = nn.Sequential(
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        )

        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_channels * 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_channels * 5,
                out_channels,
                kernel_size=1,
                bias=False)
        )

        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False)
        )

    def forward(self, x):

        # x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear')+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear')+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear')+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear')+x_list[3])))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out

# #}

# #{ SegmentHead

class SegmentHead(nn.Module):

    def __init__(self, in_channels, inter_channels, out_channels, scale_factor=8):
        super(SegmentHead, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            inter_channels,
            kernel_size=3,
            padding=1,
            bias=True
        )
        #self.bn2 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            inter_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=True
        )
        self.scale_factor = scale_factor

    def forward(self, x):

        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(x))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(
                out,
                size=[height, width],
                mode='bilinear'
            )

        return out

# #}

# #{ DDRNet23s

class DDRNet23s(nn.Module):

    def __init__(self, block, layers, num_classes=19, channels=64, spp_channels=128, head_channels=128, augment=False):
        super(DDRNet23s, self).__init__()

        highres_channels = channels * 2
        self.augment = augment

        self.conv1 =  nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(block, channels, channels, layers[0])
        self.layer2 = self._make_layer(block, channels, channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels * 2, channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels * 4, channels * 8, layers[3], stride=2)

        self.compression3 = nn.Sequential(
            nn.Conv2d(
                channels * 4,
                highres_channels,
                kernel_size=1,
                bias=True
            ),
            # nn.BatchNorm2d(highres_channels)
        )

        self.compression4 = nn.Sequential(
            nn.Conv2d(
                channels * 8,
                highres_channels,
                kernel_size=1,
                bias=True),
            # nn.BatchNorm2d(highres_channels)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(
                highres_channels,
                channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),
            # nn.BatchNorm2d(channels * 4)
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(
                highres_channels,
                channels * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),
            # nn.BatchNorm2d(channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels * 4,
                channels * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),
            # nn.BatchNorm2d(channels * 8),
        )

        self.layer3_ = self._make_layer(block, channels * 2, highres_channels, 2)

        self.layer4_ = self._make_layer(block, highres_channels, highres_channels, 2)

        self.layer5_ = self._make_layer(ResidualBottleneckBlock, highres_channels, highres_channels, 1)

        self.layer5 =  self._make_layer(ResidualBottleneckBlock, channels * 8, channels * 8, 1, stride=2)

        self.spp = DAPPM(channels * 16, spp_channels, channels * 4)

        if self.augment:
            self.seghead_extra = SegmentHead(highres_channels, head_channels, num_classes)

        self.final_layer = SegmentHead(channels * 4, head_channels, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, in_channels, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                # nn.BatchNorm2d(channels * block.expansion),
            )

        layers = []
        layers.append(block(in_channels, channels, stride, downsample))
        in_channels = channels * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(in_channels, channels, stride=1, no_relu=True))
            else:
                layers.append(block(in_channels, channels, stride=1, no_relu=False))

        return nn.Sequential(*layers)


    def forward(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        layers = []

        x = self.conv1(x)

        x = self.layer1(x)
        layers.append(x)

        x = self.layer2(self.relu(x))
        layers.append(x)

        x = self.layer3(self.relu(x))
        layers.append(x)
        x_ = self.layer3_(self.relu(layers[1]))

        x = x + self.down3(self.relu(x_))
        x_ = x_ + F.interpolate(
                        self.compression3(self.relu(layers[2])),
                        size=[height_output, width_output],
                        mode='bilinear')
        if self.augment:
            temp = x_

        x = self.layer4(self.relu(x))
        layers.append(x)
        x_ = self.layer4_(self.relu(x_))

        x = x + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(
                        self.compression4(self.relu(layers[3])),
                        size=[height_output, width_output],
                        mode='bilinear')

        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
                        self.spp(self.layer5(self.relu(x))),
                        size=[height_output, width_output],
                        mode='bilinear')

        x_ = self.final_layer(x + x_)

        if self.augment:
            x_extra = self.seghead_extra(temp)
            return [x_, x_extra]
        else:
            return x_

# #}

#def DDRNet23s_imagenet(pretrained=False):
#    model = DDRNet23s(SequentialResidualBlock, [2, 2, 2, 2], num_classes=19, channels=32, spp_channels=128, head_channels=64, augment=True)
#    if pretrained:
#        checkpoint = torch.load('/home/user1/hyd/HRNet/' + "DDRNet23s_imagenet.pth", map_location='cpu')
#        '''
#        new_state_dict = OrderedDict()
#        for k, v in checkpoint['state_dict'].items():
#            name = k[7:]
#            new_state_dict[name] = v
#        #model_dict.update(new_state_dict)
#        #model.load_state_dict(model_dict)
#        '''
#        model.load_state_dict(new_state_dict, strict = False)
#    return model

#def get_seg_model(cfg, **kwargs):

#    model = DDRNet23s_imagenet(pretrained=False)
#    return model

if __name__ == '__main__':

    # import time

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dummy_input = torch.randn(1, 3, 1024, 2048)
    # dummy_input = dummy_input.to(device)

    ddrnet23s = DDRNet23s(
        SequentialResidualBlock,
        [2, 2, 2, 2],
        num_classes=19,
        channels=32,
        spp_channels=128,
        head_channels=64
    )

    # ddrnet23s.eval()
    # ddrnet23s = ddrnet23s.to(device)

    flops, params = profile(ddrnet23s, inputs=(dummy_input, ))
    flops, params = clever_format([flops, params], '%.3f')

    print(f'FLOPs: {flops}')
    print(f'Parameters: {params}')

    # iterations = None
    # with torch.no_grad():
    #     for _ in range(10):
    #         model(input)

    #     if iterations is None:
    #         elapsed_time = 0
    #         iterations = 100
    #         while elapsed_time < 1:
    #             torch.cuda.synchronize()
    #             torch.cuda.synchronize()
    #             t_start = time.time()
    #             for _ in range(iterations):
    #                 model(input)
    #             torch.cuda.synchronize()
    #             torch.cuda.synchronize()
    #             elapsed_time = time.time() - t_start
    #             iterations *= 2
    #         FPS = iterations / elapsed_time
    #         iterations = int(FPS * 6)

    #     print('=========Speed Testing=========')
    #     torch.cuda.synchronize()
    #     torch.cuda.synchronize()
    #     t_start = time.time()
    #     for _ in range(iterations):
    #         model(input)
    #     torch.cuda.synchronize()
    #     torch.cuda.synchronize()
    #     elapsed_time = time.time() - t_start
    #     latency = elapsed_time / iterations * 1000
    # torch.cuda.empty_cache()
    # FPS = 1000 / latency
    # print(FPS)
