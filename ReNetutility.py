# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:19:13 2020

@author: Jack
"""

import torch
import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class MyReNet1D(nn.Module):
    
    def __init__(self, input_channel, layers, num_classes = 4):
        self.inplanes = 64
        super(MyReNet1D, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer1 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer2 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer3 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        
        self.layer4 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer5 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer6 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer7 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        
        self.layer8 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer9 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer10 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer11 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        #self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)
        
        #self.fc1 = nn.Linear(256*13, num_classes) # 50000
        self.fc1 = nn.Linear(256, 200) # 5000
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)
        self.fc4 = nn.Linear(10, bear   )
        self.drop_out1 = nn.Dropout(0.5)
        self.drop_out2 = nn.Dropout(0.2)
        
    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
            
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.drop_out1(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        
        x = self.maxpool(x)
        
        x = x.view(x.size(0), -1)
        #x = self.drop_out2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        
        return x
        

class ResNet50000(nn.Module):
    # for 50000 input
    def __init__(self, input_channel, layers, regression_out = 1):
        self.inplanes = 64
        super(ResNet50000, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer0 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer1 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer2 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer3 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)


        self.layer4 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer5 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2) 
        self.layer6 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer7 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)


        self.layer8 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer9 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer10 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer11 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)

        
        # length == 50000
        self.fc0 = nn.Linear(512, 400)
        self.fc1 = nn.Linear(400, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)
        self.fc4 = nn.Linear(10, 1)
        '''
        self.fc0 = nn.Linear(512, 200)
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 10)        
        self.fc3 = nn.Linear(10, 1)
        '''
        
    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        
         
        return x


class ResNet5000(nn.Module):
    # for 5000 input
    def __init__(self, input_channel, layers, regression_out = 1):
        self.inplanes = 64
        super(ResNet5000, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer0 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer1 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer2 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer3 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)


        self.layer4 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer5 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2) 
        self.layer6 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer7 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)


        self.layer8 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer9 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer10 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer11 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)

        
        # length == 50000
        self.fc0 = nn.Linear(256, 200)
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 10)        
        self.fc3 = nn.Linear(10, 1)
        
    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
         
        return x
        


class MyReNetReg50000(nn.Module):
    # for 50000 input
    def __init__(self, input_channel, layers, regression_out = 1):
        self.inplanes = 64
        super(MyReNetReg50000, self).__init__()
        
        self.conv3x3_0 = nn.Conv1d(input_channel, 2, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv3x3_1 = nn.Conv1d(2,             4, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.conv3x3_2 = nn.Conv1d(4,             8, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.bnx0 = nn.BatchNorm1d(8)
        
        self.conv5x5_0 = nn.Conv1d(input_channel, 2, kernel_size = 5, stride = 1, padding = 2, bias = False)
        self.conv5x5_1 = nn.Conv1d(2,             4, kernel_size = 5, stride = 2, padding = 2, bias = False)
        self.conv5x5_2 = nn.Conv1d(4,             8, kernel_size = 5, stride = 2, padding = 2, bias = False)
        self.bnx1 = nn.BatchNorm1d(8)
        
        self.conv7x7_0 = nn.Conv1d(input_channel, 2, kernel_size = 7, stride = 1, padding = 3, bias = False)
        self.conv7x7_1 = nn.Conv1d(2,             4, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.conv7x7_2 = nn.Conv1d(4,             8, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bnx2 = nn.BatchNorm1d(8)
        
        self.conv1 = nn.Conv1d(24, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 1, padding = 1)
        
        self.layer0 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer1 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer2 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer3 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)


        self.layer4 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer5 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2) 
        self.layer6 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer7 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)


        self.layer8 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer9 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer10 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer11 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)

        
        # length == 50000
        self.fc0 = nn.Linear(512, 200)
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 1)
        
    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
        
    def forward(self, x):
        
        x2 = self.conv3x3_0(x)
        x2 = self.conv3x3_1(x2)
        x2 = self.conv3x3_2(x2)
        x2 = self.bnx0(x2)
        
        x3 = self.conv5x5_0(x)
        x3 = self.conv5x5_1(x3)
        x3 = self.conv5x5_2(x3)
        x3 = self.bnx1(x3)
        
        x4 = self.conv7x7_0(x)
        x4 = self.conv7x7_1(x4)
        x4 = self.conv7x7_2(x4)
        x4 = self.bnx2(x4)
        
        x = torch.cat([x2, x3, x4], 1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x
        

class MyReNetReg5000(nn.Module):
    
    def __init__(self, input_channel, layers, regression_out = 1):
        self.inplanes = 64
        super(MyReNetReg5000, self).__init__()
        
        
        self.conv3x3_0 = nn.Conv1d(input_channel, 2, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv3x3_1 = nn.Conv1d(2,             4, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv3x3_2 = nn.Conv1d(4,             8, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bnx0 = nn.BatchNorm1d(8)
        
        self.conv5x5_0 = nn.Conv1d(input_channel, 2, kernel_size = 5, stride = 1, padding = 2, bias = False)
        self.conv5x5_1 = nn.Conv1d(2,             4, kernel_size = 5, stride = 1, padding = 2, bias = False)
        self.conv5x5_2 = nn.Conv1d(4,             8, kernel_size = 5, stride = 1, padding = 2, bias = False)
        self.bnx1 = nn.BatchNorm1d(8)
        
        self.conv7x7_0 = nn.Conv1d(input_channel, 2, kernel_size = 7, stride = 1, padding = 3, bias = False)
        self.conv7x7_1 = nn.Conv1d(2,             4, kernel_size = 7, stride = 1, padding = 3, bias = False)
        self.conv7x7_2 = nn.Conv1d(4,             8, kernel_size = 7, stride = 1, padding = 3, bias = False)
        self.bnx2 = nn.BatchNorm1d(8)
        
        self.conv1 = nn.Conv1d(24, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 1, padding = 1)
        
        self.layer0 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer1 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer2 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer3 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)


        self.layer4 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer5 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2) 
        self.layer6 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer7 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)


        self.layer8 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer9 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer10 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer11 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)

        
        # length == 5000
        self.fc0 = nn.Linear(256, 200)
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 10)        
        self.fc3 = nn.Linear(10, 1)
        #self.drop_out1 = nn.Dropout(0.2)
        #self.drop_out2 = nn.Dropout(0.2)
        
    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
        
    def forward(self, x):
        
        x2 = self.conv3x3_0(x)
        x2 = self.conv3x3_1(x2)
        x2 = self.conv3x3_2(x2)
        x2 = self.bnx0(x2)
        
        x3 = self.conv5x5_0(x)
        x3 = self.conv5x5_1(x3)
        x3 = self.conv5x5_2(x3)
        x3 = self.bnx1(x3)
        
        x4 = self.conv7x7_0(x)
        x4 = self.conv7x7_1(x4)
        x4 = self.conv7x7_2(x4)
        x4 = self.bnx2(x4)
        
        x = torch.cat([x2, x3, x4], 1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x
        

        
class Conv_Cat_Residual2(nn.Module):
    
    def __init__(self, input_channel, layers, regression_out = 1):
        self.inplanes = 64
        super(Conv_Cat_Residual2, self).__init__()
        
        self.conv3x3_0 = nn.Conv1d(input_channel, 2, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv3x3_1 = nn.Conv1d(2,             4, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv3x3_2 = nn.Conv1d(4,             8, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bnx0 = nn.BatchNorm1d(8)
        
        self.conv5x5_0 = nn.Conv1d(input_channel, 2, kernel_size = 5, stride = 1, padding = 2, bias = False)
        self.conv5x5_1 = nn.Conv1d(2,             4, kernel_size = 5, stride = 1, padding = 2, bias = False)
        self.conv5x5_2 = nn.Conv1d(4,             8, kernel_size = 5, stride = 1, padding = 2, bias = False)
        self.bnx1 = nn.BatchNorm1d(8)
        
        self.conv7x7_0 = nn.Conv1d(input_channel, 2, kernel_size = 7, stride = 1, padding = 3, bias = False)
        self.conv7x7_1 = nn.Conv1d(2,             4, kernel_size = 7, stride = 1, padding = 3, bias = False)
        self.conv7x7_2 = nn.Conv1d(4,             8, kernel_size = 7, stride = 1, padding = 3, bias = False)
        self.bnx2 = nn.BatchNorm1d(8)
        
        self.conv3y3_0 = nn.Conv1d(input_channel, 2, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv3y3_1 = nn.Conv1d(2,             4, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv3y3_2 = nn.Conv1d(4,             8, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bny0 = nn.BatchNorm1d(8)
        
        self.conv5y5_0 = nn.Conv1d(input_channel, 2, kernel_size = 5, stride = 1, padding = 2, bias = False)
        self.conv5y5_1 = nn.Conv1d(2,             4, kernel_size = 5, stride = 1, padding = 2, bias = False)
        self.conv5y5_2 = nn.Conv1d(4,             8, kernel_size = 5, stride = 1, padding = 2, bias = False)
        self.bny1 = nn.BatchNorm1d(8)
        
        self.conv7y7_0 = nn.Conv1d(input_channel, 2, kernel_size = 7, stride = 1, padding = 3, bias = False)
        self.conv7y7_1 = nn.Conv1d(2,             4, kernel_size = 7, stride = 1, padding = 3, bias = False)
        self.conv7y7_2 = nn.Conv1d(4,             8, kernel_size = 7, stride = 1, padding = 3, bias = False)
        self.bny2 = nn.BatchNorm1d(8)

        self.conv10 = nn.Conv1d(48, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 1, padding = 1)
        
        
        self.layer0 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer1 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer2 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer3 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        
        self.layer4 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer5 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer6 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer7 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        
        self.layer8 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2) 
        self.layer9 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer10 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer11 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        
        # length == 5000
        self.fc0 = nn.Linear(256, 200)
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 1)
        
        self.drop_out = nn.Dropout(0.2)
        
    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x, y):
   
        x2 = self.conv3x3_0(x)
        x2 = self.conv3x3_1(x2)
        x2 = self.conv3x3_2(x2)
        x2 = self.bnx0(x2)
        
        x3 = self.conv5x5_0(x)
        x3 = self.conv5x5_1(x3)
        x3 = self.conv5x5_2(x3)
        x3 = self.bnx1(x3)
        
        x4 = self.conv7x7_0(x)
        x4 = self.conv7x7_1(x4)
        x4 = self.conv7x7_2(x4)
        x4 = self.bnx2(x4)
        
        x = torch.cat([x2, x3, x4], 1)
       
        
        y2 = self.conv3y3_0(y)
        y2 = self.conv3y3_1(y2)
        y2 = self.conv3y3_2(y2)
        y2 = self.bny0(y2)
        
        y3 = self.conv5y5_0(y)
        y3 = self.conv5y5_1(y3)
        y3 = self.conv5y5_2(y3)
        y3 = self.bny1(y3)
        
        y4 = self.conv7y7_0(y)
        y4 = self.conv7y7_1(y4)
        y4 = self.conv7y7_2(y4)
        y4 = self.bny2(y4)
        
        y = torch.cat([y2, y3, y4], 1)
        
        
        z = torch.cat([x, y], 1)
        
        
        z = self.conv10(z)
        z = self.bn3(z)
        z = self.relu(z)
        z = self.maxpool(z)
        
        z = self.layer0(z)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        
        z = self.layer4(z)
        z = self.layer5(z)
        z = self.layer6(z)
        z = self.layer7(z)
        
        z = self.layer8(z)
        z = self.layer9(z)
        z = self.layer10(z)
        z = self.layer11(z)
        z = self.maxpool(z)
        
        z = z.view(z.size(0), -1)
        
        z = self.fc0(z)
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)

        
        return z

        
class Conv_Cat_Residual3(nn.Module):
    
    def __init__(self, input_channel, layers, regression_out = 1):
        self.inplanes = 64
        super(Conv_Cat_Residual3, self).__init__()
        
        self.conv3x3_0 = nn.Conv1d(input_channel, 2, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv3x3_1 = nn.Conv1d(2,             4, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.conv3x3_2 = nn.Conv1d(4,             8, kernel_size = 3, stride = 2, padding = 1, bias = False)
        
        self.conv5x5_0 = nn.Conv1d(input_channel, 2, kernel_size = 5, stride = 1, padding = 2, bias = False)
        self.conv5x5_1 = nn.Conv1d(2,             4, kernel_size = 5, stride = 2, padding = 2, bias = False)
        self.conv5x5_2 = nn.Conv1d(4,             8, kernel_size = 5, stride = 2, padding = 2, bias = False)
        
        self.conv7x7_0 = nn.Conv1d(input_channel, 2, kernel_size = 7, stride = 1, padding = 3, bias = False)
        self.conv7x7_1 = nn.Conv1d(2,             4, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.conv7x7_2 = nn.Conv1d(4,             8, kernel_size = 7, stride = 2, padding = 3, bias = False)
        
        
        self.conv3y3_0 = nn.Conv1d(input_channel, 2, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.conv3y3_1 = nn.Conv1d(2,             4, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.conv3y3_2 = nn.Conv1d(4,             8, kernel_size = 3, stride = 2, padding = 1, bias = False)
        
        self.conv5y5_0 = nn.Conv1d(input_channel, 2, kernel_size = 5, stride = 1, padding = 2, bias = False)
        self.conv5y5_1 = nn.Conv1d(2,             4, kernel_size = 5, stride = 2, padding = 2, bias = False)
        self.conv5y5_2 = nn.Conv1d(4,             8, kernel_size = 5, stride = 2, padding = 2, bias = False)
        
        self.conv7y7_0 = nn.Conv1d(input_channel, 2, kernel_size = 7, stride = 1, padding = 3, bias = False)
        self.conv7y7_1 = nn.Conv1d(2,             4, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.conv7y7_2 = nn.Conv1d(4,             8, kernel_size = 7, stride = 2, padding = 3, bias = False)

        self.conv10 = nn.Conv1d(48, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)

        self.bn0 = nn.BatchNorm1d(1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 1, padding = 1)
        
        
        self.layer0 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer1 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer2 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer3 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        
        self.layer4 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer5 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer6 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer7 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        
        self.layer8 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2) 
        self.layer9 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer10 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer11 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        
        # length == 50000
        
        self.fc0 = nn.Linear(512, 400)
        self.fc1 = nn.Linear(400, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)
        self.fc4 = nn.Linear(10, 1)
        
        '''
        self.fc0 = nn.Linear(256, 200)
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 1)
        self.drop_out = nn.Dropout(0.2)
        '''
    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x, y):
   
        x2 = self.conv3x3_0(x)
        x2 = self.conv3x3_1(x2)
        x2 = self.conv3x3_2(x2)
        
        
        x3 = self.conv5x5_0(x)
        x3 = self.conv5x5_1(x3)
        x3 = self.conv5x5_2(x3)
        
        x4 = self.conv7x7_0(x)
        x4 = self.conv7x7_1(x4)
        x4 = self.conv7x7_2(x4)
        
        x = torch.cat([x2, x3, x4], 1)
       
        
        y2 = self.conv3y3_0(y)
        y2 = self.conv3y3_1(y2)
        y2 = self.conv3y3_2(y2)
        
        y3 = self.conv5y5_0(y)
        y3 = self.conv5y5_1(y3)
        y3 = self.conv5y5_2(y3)
        
        y4 = self.conv7y7_0(y)
        y4 = self.conv7y7_1(y4)
        y4 = self.conv7y7_2(y4)
        
        y = torch.cat([y2, y3, y4], 1)
        
        
        z = torch.cat([x, y], 1)
        z = self.conv10(z)
        z = self.bn3(z)
        z = self.relu(z)
        z = self.maxpool(z)
        z = self.layer0(z)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        
        z = self.layer4(z)
        z = self.layer5(z)
        z = self.layer6(z)
        z = self.layer7(z)
        
        z = self.layer8(z)
        z = self.layer9(z)
        z = self.layer10(z)
        z = self.layer11(z)
        z = self.maxpool(z)
        z = z.view(z.size(0), -1)
        
        z = self.fc0(z)
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        z = self.fc4(z)
        #z = self.fc5(z)

        
        return z        


class AEVB_OnlyResNet5000(nn.Module):
    # for 5000 input
    def __init__(self, input_channel, layers, regression_out = 1):
        self.inplanes = 64
        super(AEVB_OnlyResNet5000, self).__init__()
        
        self.conv1 = nn.Conv1d(2, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer0 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer1 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer2 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer3 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)


        self.layer4 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer5 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2) 
        self.layer6 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer7 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)


        self.layer8 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer9 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer10 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer11 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)

        
        # length == 50000
        self.fc0 = nn.Linear(256, 200)
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 10)        
        self.fc3 = nn.Linear(10, 1)
        
    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
        
    def forward(self, x, y):
        
        z = torch.cat([x, y], 1)
        z = self.conv1(z)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.maxpool(z)
        
        z = self.layer0(z)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        
        z = self.layer4(z)
        z = self.layer5(z)
        z = self.layer6(z)
        z = self.layer7(z)
        
        z = self.layer8(z)
        z = self.layer9(z)
        z = self.layer10(z)
        z = self.layer11(z)
        
        z = self.maxpool(z)
        z = z.view(z.size(0), -1)
        
        z = self.fc0(z)
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
         
        return z
        


class AEVB_OnlyResNet50000(nn.Module):
    # for 5000 input
    def __init__(self, input_channel, layers, regression_out = 1):
        self.inplanes = 64
        super(AEVB_OnlyResNet50000, self).__init__()
        
        self.conv1 = nn.Conv1d(2, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer0 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer1 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer2 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)
        self.layer3 = self._make_layer(BasicResBlock, 64, layers[0], stride = 2)


        self.layer4 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer5 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2) 
        self.layer6 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)
        self.layer7 = self._make_layer(BasicResBlock, 128, layers[1], stride = 2)


        self.layer8 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer9 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer10 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)
        self.layer11 = self._make_layer(BasicResBlock, 256, layers[2], stride = 2)

        
        # length == 50000
        self.fc0 = nn.Linear(512, 200)
        self.fc1 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 10)        
        self.fc3 = nn.Linear(10, 1)
        
    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
        
    def forward(self, x, y):
        
        z = torch.cat([x, y], 1)
        z = self.conv1(z)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.maxpool(z)
        
        z = self.layer0(z)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        
        z = self.layer4(z)
        z = self.layer5(z)
        z = self.layer6(z)
        z = self.layer7(z)
        
        z = self.layer8(z)
        z = self.layer9(z)
        z = self.layer10(z)
        z = self.layer11(z)
        
        z = self.maxpool(z)
        z = z.view(z.size(0), -1)
        
        z = self.fc0(z)
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
         
        return z
        