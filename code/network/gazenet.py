"""
@author: Dongze Lian
@contact: liandz@shanghaitech.edu.cn
@software: PyCharm
@file: gazenet.py
@time: 2020/1/12 15:21
"""

import torch.nn as nn
import torch.nn.functional as F
import math
import torch

from .resnet import resnet18, resnet34, resnet50, resnet101

import pdb

class GazeNet(nn.Module):
    def __init__(self, backbone, view, pretrained=False):
        super(GazeNet, self).__init__()
        if view == 'multiview':
            self.view = 'multi'
        else:
            self.view = 'single'
        if backbone == 'ResNet-18':
            self.expansion = 1
            self.backbone = resnet18(pretrained=pretrained)
        elif backbone == 'ResNet-34':
            self.expansion = 1
            self.backbone = resnet34(pretrained=pretrained)
        elif backbone == 'ResNet-50':
            self.expansion = 4
            self.backbone = resnet50(pretrained=pretrained)
        elif backbone == 'ResNet-101':
            self.expansion = 4
            self.backbone = resnet101(pretrained=pretrained)
        else:
            raise TypeError('Currently only support backbones '
                            '(ResNet-18, ResNet-34, ResNet-50, ResNet-101)')

        if self.view == 'single':
            self.features = nn.Sequential(
                self.backbone,
                nn.AvgPool2d(7),
                nn.Flatten())
            self.eyelocation = nn.Linear(24, 128)
            self.fc = nn.Sequential(
                nn.Linear(512 * self.expansion * 2 + 128, 128),
                nn.Linear(128, 2)
            )
        elif self.view == 'multi':
            self.svffn_fc1 = nn.Linear(1024, 256)
            self.fc_eye_loc = nn.Linear(24 * 3, 128)
            self.svffn_fc2 = nn.Linear(256 + 128, 128)
            self.cvffn_conv1 = nn.Sequential(
                nn.Conv2d(1024, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            )
            self.cvffn_fc1 = nn.Linear(64 * 7 *7, 256)
            self.cvffn_fc2 = nn.Linear(256 + 128, 128)
            self.fc = nn.Sequential(
                nn.Linear(128 * 4, 128),
                nn.Linear(128, 2)
            )


    def single_view(self, input):
        out = []
        for i, input_i in enumerate(input):
            # the last item of x is eyelocation
            if i != len(input) - 1:
                out.append(self.features(input_i))
            else:
                out.append(self.eyelocation(input_i))

        out = torch.cat(out, 1)
        out = self.fc(out)

        return out


    def svffn(self, eye_feature, eye_loc):
        eye_feature = F.adaptive_avg_pool2d(eye_feature, 1)
        eye_feature = torch.flatten(eye_feature, 1)
        eye_feature = self.svffn_fc1(eye_feature)
        eye_loc_feature = self.fc_eye_loc(eye_loc)
        eye_feature_fusion = torch.cat((eye_feature, eye_loc_feature), 1)
        eye_feature_fusion = self.svffn_fc2(eye_feature_fusion)

        return eye_feature_fusion

    def cvffn(self, cross_eye_feature, eye_loc):
        cross_eye_feature = self.cvffn_conv1(cross_eye_feature)
        cross_eye_feature = torch.flatten(cross_eye_feature, 1)
        cross_eye_feature = self.cvffn_fc1(cross_eye_feature)
        eye_loc_feature = self.fc_eye_loc(eye_loc)
        cross_eye_feature_fusion = torch.cat((cross_eye_feature, eye_loc_feature), 1)
        cross_eye_feature_fusion = self.cvffn_fc2(cross_eye_feature_fusion)

        return cross_eye_feature_fusion

    def multi_view(self, input):
        # parsing input
        # TODO: elegant implementation
        lc_le = input[0]
        lc_re = input[1]
        lc_eye_loc = input[2]

        mc_le = input[3]
        mc_re = input[4]
        mc_eye_loc = input[5]

        rc_le = input[6]
        rc_re = input[7]
        rc_eye_loc = input[8]

        lc_feature = torch.cat((self.backbone(lc_le), self.backbone(lc_re)), 1)
        mc_feature = torch.cat((self.backbone(mc_le), self.backbone(mc_re)), 1)
        rc_feature = torch.cat((self.backbone(rc_le), self.backbone(rc_re)), 1)
        eye_loc_feature = torch.cat((lc_eye_loc, mc_eye_loc, rc_eye_loc), 1)

        #pdb.set_trace()
        lc_feature_fusion = self.svffn(lc_feature, eye_loc_feature)
        mc_feature_fusion = self.svffn(mc_feature, eye_loc_feature)
        rc_feature_fusion = self.svffn(rc_feature, eye_loc_feature)

        cross_feature = torch.max(torch.max(lc_feature, mc_feature), rc_feature)
        cross_feature_fusion = self.cvffn(cross_feature, eye_loc_feature)

        out = torch.cat((lc_feature_fusion, mc_feature_fusion, rc_feature_fusion, cross_feature_fusion), 1)
        out = self.fc(out)

        return out

    # The length of x is not fixed: 3 for single_view and 9 for multi_view
    def forward(self, *input):
        if self.view == 'single':
            out = self.single_view(input)
        elif self.view == 'multi':
            out = self.multi_view(input)
        else:
            raise TypeError('Cannot parse the self.view (single or multi)')

        return out