'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-03-25 19:12:24
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-04-29 15:49:18
FilePath: /Ternary-Spike/models/vgg_q.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''
import torch
import torch.nn as nn
import math
from functools import partial


def conv3x3(in_planes, x_planes, stride=1, fea_all_positive=False):
    " 3x3 convolution with padding"
    return nn.Conv2d(in_planes, x_planes, kernel_size=3, stride=stride, padding=1, bias=False)



class tdBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features)
    def forward(self, x: torch.tensor):
        t, b, c, h, w = x.shape
        return super().forward(x.flatten(0,1)).reshape(t, b, c, h, w)
    

class tdAvgPool2d(nn.AvgPool2d):
    def __init__(self, kernel_size, stride):
        super().__init__(kernel_size, stride)
    def forward(self, x: torch.tensor):
        t, b, c, _, __ = x.shape
        x = super().forward(x.flatten(0,1))
        _, __, h, w = x.shape
        
        return x.reshape(t, b, c, h, w)



class VGGSNN(nn.Module):
    def __init__(self, cfg=None,num_classes=10):
        super().__init__()

        if cfg is None:
            cfg = [64, 128, 'A', 256, 256, 'A', 512, 512,  'A', 512, 512]

        
        self.conv1 = nn.Conv2d(2,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = tdBatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 128)
        self.bn2 = tdBatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
    
        self.pool1 = tdAvgPool2d(2, 2)
        
        self.conv3 = conv3x3(128, 256)
        self.bn3 = tdBatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = conv3x3(256, 256)
        self.bn4 = tdBatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.pool2 = tdAvgPool2d(2, 2)
        
        self.conv5 = conv3x3(256, 512)
        self.bn5 = tdBatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = conv3x3(512, 512)
        self.bn6 = tdBatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)
        
        self.pool3 = tdAvgPool2d(2, 2)
        
        self.conv7 = conv3x3(512, 512)
        self.bn7 = tdBatchNorm2d(512)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = conv3x3(512, 512)
        self.bn8 = tdBatchNorm2d(512)
        self.relu8 = nn.ReLU(inplace=True)
        
        self.pool4 = tdAvgPool2d(2, 2)
        
        W = int(48 / 2 / 2 / 2 / 2)
        
        self.classifier = nn.Linear(512 * W * W, num_classes)
        # self.boost = nn.AvgPool1d(10, 10)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, use_tet=False):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        T, B, C, H, W = x.shape
        
        self.q_info = []
        x, q_info = self.conv1(x, self.q_info)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x, q_info = self.conv2(x, q_info)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.pool1(x)
        
        x, q_info = self.conv3(x, q_info)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x, q_info = self.conv4(x, q_info)
        x = self.bn4(x)
        x = self.relu4(x)
        
        x = self.pool2(x)
        
        x, q_info = self.conv5(x, q_info)
        x = self.bn5(x)
        x = self.relu5(x)
        
        x, q_info = self.conv6(x, q_info)
        x = self.bn6(x)
        x = self.relu6(x)
        
        
        x = self.pool3(x)
        
        x, q_info = self.conv7(x, q_info)
        x = self.bn7(x)
        x = self.relu7(x)
        
        x, q_info = self.conv8(x, q_info)
        x = self.bn8(x)
        x = self.relu8(x)
        
        
        x = self.pool4(x)
        
        
        x = x.reshape(x.shape[0], x.shape[1], 1, -1) # t,b,1,c
        x, q_info = self.classifier(x, q_info)
        x = x.reshape(T, B, -1)
        # x = self.boost(x.flatten(0,1)).reshape(T, B, -1)
        return x if use_tet else x.mean(0)



def vggsnn_dvs_q(num_classes=10):
    model = VGGSNN(num_classes=num_classes)
    return model



if __name__ == '__main__':
    net = vggsnn_dvs_q()
    net.eval()
    x = torch.randn(2, 2, 48, 48)
    y1 = net(x)
    print(y1.shape)