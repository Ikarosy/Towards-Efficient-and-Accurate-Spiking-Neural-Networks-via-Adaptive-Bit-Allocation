'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-03-25 19:12:24
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-08-14 20:18:14
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
from spikingjelly.activation_based import neuron, layer


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



class d128snn(nn.Module):
    def __init__(self, in_channels=700, cfg=None, num_classes=20):
        super().__init__()
        self.in_channels=in_channels
        
        self.linear1 = nn.Linear(in_features=in_channels, out_features=256, bias=False)
        self.fc1_bn = layer.BatchNorm1d(256, step_mode='m')
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = layer.Dropout(0., step_mode='m')

        
        self.linear2 = nn.Linear(in_features=256, out_features=256, bias=False)
        self.fc2_bn = layer.BatchNorm1d(256, step_mode='m')
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = layer.Dropout(0., step_mode='m')

        self.head = nn.Linear(in_features=256, out_features=20, bias=False)
        self.softmax_fn = nn.Softmax(dim=2)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, use_tet=False):
        x = x.permute(1, 0, 2)  #[B,T,C] -> [T, B, C]
        T, B, C= x.shape
        
        self.q_info = []
        x, q_info = self.linear1(x, self.q_info)
        x = self.fc1_bn(x.unsqueeze(3)) # [T, B, C] -> [T, B, C, L]
        x = self.relu1(x)
        x = self.dropout1(x)

        x, q_info = self.linear2(x.squeeze(3), self.q_info)    #  [T, B, C, L] -> [T, B, C]
        x = self.fc2_bn(x.unsqueeze(3))  # [T, B, C] -> [T, B, C, L]
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x, q_info = self.head(x.squeeze(3), q_info) #  [T, B, C, L] -> [T, B, C]

        return x if use_tet else  torch.sum(self.softmax_fn(x), 0)



def d128snn_shd_q(in_channels=700, num_classes=20, args=None):
    model = d128snn(in_channels=in_channels, num_classes=num_classes)
    return model



if __name__ == '__main__':
    net = d128snn_shd_q()
    net.eval()
    x = torch.randn(2, 2, 48, 48)
    y1 = net(x)
    print(y1.shape)