'''
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''
import torch
import torch.nn as nn
from .quantization import *
from .quantization.quantization_modules import layer_num
import math


def conv3x3(in_planes, out_planes, stride=1, fea_all_positive=False, **kwargs):
    " 3x3 convolution with padding"
    return qConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, fea_all_positive=fea_all_positive, **kwargs)

class basic_downsample(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, bias, fea_all_positive, **kwargs) -> None:
        super().__init__()
        self.downsample = qConv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, bias=bias, fea_all_positive=fea_all_positive, **kwargs)
        self.bn = BN(out_planes)
        
    def forward(self, s):
        x, q_info = s
        x, q_info = self.downsample(x, q_info)
        T, B, C, H, W = x.shape
        x = self.bn(x.flatten(0,1)).reshape(T, B, C, H, W)
        return x, q_info


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, fea_all_positive=False, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, fea_all_positive=fea_all_positive, **kwargs)
        self.bn1 = BN(planes)
        self.relu1 = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, fea_all_positive=fea_all_positive, **kwargs)
        self.bn2 = BN(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu2 = ReLU(inplace=True)

    def forward(self, s):
        x, q_info = s
        residual = x
        out, q_info = self.conv1(x, q_info)
        T, B, C, H, W = out.shape
        out = self.bn1(out.flatten(0,1))
      
        out = self.relu1(out).reshape(T, B, C, H, W)

        out, q_info = self.conv2(out, q_info)
        T, B, C, H, W = out.shape
        out = self.bn2(out.flatten(0,1)).reshape(T, B, C, H, W)
        #print(out.min())
        #print(out.max())  

        if self.downsample is not None:
            residual, q_info = self.downsample((residual, q_info))
            
        out += residual
        
        out1 = self.relu2(out.clone())
        #print(out.min())
        #print(out1.min())

        return out1, q_info


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, input_c=2, num_classes=10, rp = False, **kwargs):
        super().__init__()

        global BN
        global layer_num
        BN = nn.BatchNorm2d
        global ReLU
        ReLU = nn.ReLU
        
        # self.scale = nn.Parameter(torch.ones(1, 1, 2048), requires_grad=True)
        self.rp = False 
        inplanes = 128
        self.inplanes = 128
        self.conv1 = qConv2d(input_c, inplanes, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn1 = BN(self.inplanes)
        self.relu = ReLU(inplace=True)
        self.layer1 = self._make_layer(block, inplanes, layers[0], fea_all_positive=False, **kwargs)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2, fea_all_positive=False, **kwargs)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2, fea_all_positive=False, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = qLinear(inplanes * 4 * block.expansion, num_classes, fea_all_positive=False, **kwargs)
        setattr(self.fc, "special_layer", "last linear")
        self.q_info = []
        for m in self.modules():
            if isinstance(m, qConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, qLinear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, fea_all_positive=False, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = basic_downsample(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, fea_all_positive=fea_all_positive, **kwargs)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, fea_all_positive=fea_all_positive, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, fea_all_positive=fea_all_positive, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x, is_adain=False, is_drop=False, feat=None):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        T, B, C, H, W = x.shape
        
        self.q_info = []
        if not feat is None:
            x = self.fc(feat)
            return x
            
        else:
            
            x, q_info = self.conv1(x, self.q_info)
            x = self.bn1(x.flatten(0,1))
            x = self.relu(x)

            x, q_info = self.layer1((x.reshape(T,B,self.conv1.out_channels,H,W), q_info))
            x, q_info = self.layer2((x, q_info))
            x, q_info = self.layer3((x, q_info))
            
            T, B, C, H, W = x.shape
            if is_drop:
                x = self.avgpool(x.flatten(0,1)).reshape(T, B, C)
            else:
                x = self.avgpool(x.flatten(0,1)).reshape(T, B, C)
            
            fea = x.mean(0)
            x, q_info = self.fc(fea, q_info)
            
            # layer_num = 0
            # for idx, q_inf in enumerate(q_info):
            #     print('layer {}'.format(idx))
            #     print('fea_bit:', q_inf['fea_bit'])
            #     print('weight_bit:', q_inf['weight_bit'])
            
            if is_adain:
                return fea,x
            else:
                return x
            

def resnet19_dvs(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 2], **kwargs)
    return model

if __name__ == '__main__':
    net = resnet19_dvs()
    net.eval()
    x = torch.randn(1, 3, 32, 32)
    print(net)
    y1 = net(x)