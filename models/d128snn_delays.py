'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-03-25 19:12:24
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-08-19 03:29:59
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
from DCLS.construct.modules import Dcls1d
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



class d128snn_delays(nn.Module):
    def __init__(self, in_channels=700, cfg=None, num_classes=20, args=None):
        super().__init__()
        ################################################
        #                    Delays                    #
        ################################################
        assert args is not None
        DCLSversion = 'gauss' 
        decrease_sig_method = 'exp'
        kernel_count = 1
        time_step = args.init_spike_step
        self.epochs = args.epochs

        max_delay = 250//time_step
        max_delay = max_delay if max_delay%2==1 else max_delay+1 # to make kernel_size an odd number
        self.max_delay = max_delay
        
        # For constant sigma without the decreasing policy, set model_type == 'snn_delays' and sigInit = 0.23 and final_epoch = 0
        sigInit = max_delay // 2        
        final_epoch = (1*args.epochs)//4     


        left_padding = max_delay-1
        right_padding = (max_delay-1) // 2

        init_pos_method = 'uniform'
        init_pos_a = -max_delay//2
        init_pos_b = max_delay//2



        self.in_channels=in_channels

        self.dcls1d1 = Dcls1d(in_channels=in_channels, out_channels=256, kernel_count=1, groups = 1, 
                                dilated_kernel_size = max_delay, bias=False, version='gauss')

        
        setattr(self.dcls1d1, 'special_layer', 'first dcls1d')
        self.fc1_bn = layer.BatchNorm1d(256, step_mode='m')
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = layer.Dropout(0.4, step_mode='m')

        
        self.dcls1d2 = Dcls1d(in_channels=256, out_channels=256, kernel_count=1, groups = 1, 
                                dilated_kernel_size = max_delay, bias=False, version='gauss')
        self.fc2_bn = layer.BatchNorm1d(256, step_mode='m')
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = layer.Dropout(0.4, step_mode='m')

        self.head = Dcls1d(in_channels=256, out_channels=20, kernel_count=1, groups = 1, 
                                dilated_kernel_size = max_delay, bias=False, version='gauss')
        self.softmax_fn = nn.Softmax(dim=2)

        self.init_weight()
        self.dcls_clamp()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (Dcls1d)):
                torch.nn.init.uniform_(m.P, a = -self.max_delay//2, b = self.max_delay//2)
                torch.nn.init.constant_(m.SIG, self.max_delay // 2 )
                m.SIG.requires_grad = False

    def dcls_clamp(self):
        for m in self.modules():
            if isinstance(m, (Dcls1d)):
                m.clamp_parameters()
    
    
    def decrease_sig(self, epoch):
        # Decreasing to 0.23 instead of 0.5
        alpha = 0
        sig = self.head.SIG[0,0,0,0].detach().cpu().item()
        if epoch < (self.epochs//4) and sig > 0.23:
            alpha = (0.23/(self.max_delay // 2))**(1/(self.epochs//4))
            for m in self.modules():
                if isinstance(m, (Dcls1d)):
                    m.SIG *= alpha
                        

    def forward(self, x, use_tet=False):
        x = x.permute(1, 0, 2)  #[B,T,C] -> [T, B, C]
        T, B, C= x.shape
        
        self.q_info = []
        x, q_info = self.dcls1d1(x, self.max_delay, self.q_info)
        x = self.fc1_bn(x.unsqueeze(3)) # [T, B, C] -> [T, B, C, L]
        x = self.relu1(x)
        x = self.dropout1(x)

        x, q_info = self.dcls1d2(x.squeeze(3), self.max_delay, q_info)    #  [T, B, C, L] -> [T, B, C]
        x = self.fc2_bn(x.unsqueeze(3))  # [T, B, C] -> [T, B, C, L]
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x, q_info = self.head(x.squeeze(3), self.max_delay, q_info) #  [T, B, C, L] -> [T, B, C]

        return x if use_tet else  torch.sum(self.softmax_fn(x), 0)




def d128snn_shd_delays_q(in_channels=700, num_classes=20, args=None):
    model = d128snn_delays(in_channels=in_channels, num_classes=num_classes, args=args)
    return model



if __name__ == '__main__':
    net = d128snn_shd_delays_q()
    net.eval()
    x = torch.randn(2, 2, 48, 48)
    y1 = net(x)
    print(y1.shape)