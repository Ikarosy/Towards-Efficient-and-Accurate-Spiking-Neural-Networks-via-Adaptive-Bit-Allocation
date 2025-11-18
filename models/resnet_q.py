'''
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''
import torch
import torch.nn as nn
import math
from .quantization import *
from .quantization.quantization_modules import layer_num
from functools import partial

# qConv2d = partial(qConv2d, bit=4)
# qLinear = partial(qLinear, bit=4)

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
        x = self.bn(x)
        return x, q_info



class SEWbasic_downsample(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, bias, fea_all_positive, **kwargs) -> None:
        super().__init__()
        self.downsample = qConv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride, bias=bias, fea_all_positive=fea_all_positive, **kwargs)
        self.bn = BN(out_planes)
        self.relu = ReLU(inplace=False)
        
    def forward(self, s):
        x, q_info = s
        x, q_info = self.downsample(x, q_info)
        x = self.bn(x)
        x = self.relu(x)
        return x, q_info
        



class SEWBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, fea_all_positive=False, **kwargs):
        super(SEWBasicBlock, self).__init__()
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
        out = self.bn1(out)
        out = self.relu1(out)

        out, q_info = self.conv2(out, q_info)
        out = self.bn2(out)
        out = self.relu2(out)
        
        if self.downsample is not None:
            residual, q_info = self.downsample((residual, q_info))

        out1 = out + residual
        

        return out1, q_info

    

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
        out = self.bn1(out)
      
        out = self.relu1(out)

        out, q_info = self.conv2(out, q_info)

        out = self.bn2(out)
        #print(out.min())
        #print(out.max())  

        if self.downsample is not None:
            residual, q_info = self.downsample((residual, q_info))
            
        out += residual
        
        out1 = self.relu2(out.clone())
        #print(out.min())
        #print(out1.min())

        return out1, q_info


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = qConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes)
        self.relu1 = ReLU(inplace=True)
        self.conv2 = qConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BN(planes)
        self.relu2 = ReLU(inplace=True)
        self.conv3 = qConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BN(planes * self.expansion)
        self.relu3 = ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual, q_info = x

        out, q_info = self.conv1(x, q_info)
        out = self.bn1(out)
        out = self.relu1(out)

        out, q_info = self.conv2(out, q_info)
        out = self.bn2(out)
        out = self.relu2(out)

        out, q_info = self.conv3(out, q_info)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return (out, q_info)


class ResNet_Cifar_q(nn.Module):

    def __init__(self, block, layers, input_c=3, num_classes=10, rp = False, **kwargs):
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
        self.q_info = []
        if not feat is None:
            x = self.fc(feat)
            return x
            
        else:
            x, q_info = self.conv1(x, self.q_info)
            x = self.bn1(x)
            x = self.relu(x)

            x, q_info = self.layer1((x, q_info))
            x, q_info = self.layer2((x, q_info))
            x, q_info = self.layer3((x, q_info))
            
            if is_drop:
                x = self.avgpool(x)
            else:
                x = self.avgpool(x)
            
            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
                fea = x
            elif len(x.shape) == 5:
                x = x.view(x.size(0), x.size(1), -1)
                fea = x.mean([0])
            x, q_info = self.fc(x, q_info)
            
            # layer_num = 0
            # for idx, q_inf in enumerate(q_info):
            #     print('layer {}'.format(idx))
            #     print('fea_bit:', q_inf['fea_bit'])
            #     print('weight_bit:', q_inf['weight_bit'])
            
            if is_adain:
                return fea,x
            else:
                return x
            

    # def bit_loss(self):
    @torch.no_grad()
    def get_bits_scales(self):
        # if self.fc.out_features == 100 or self.fc.out_features == 10:
        #     x = torch.randn(1,3,32,32).to(self.conv1.weight.device)
        #     self.forward(x, is_adain=False, is_drop=False, feat=None)
        #     return self.q_info
        spike_scales = []
        feat_scales = []
        feat_bits = []
        weight_scales = []
        weight_bits = []
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if 'fire_ratio' in fullname:
                    spike_scales.append({fullname:param.clone().detach().cpu().squeeze().numpy()})
                elif 'fea_quant'in fullname and 'gama' in fullname:
                    feat_scales.append({fullname:param.clone().detach().cpu().squeeze().numpy()})
                elif 'fea_quant'in fullname and 'bit' in fullname:
                    feat_bits.append({fullname:param.clone().detach().cpu().squeeze().numpy()})
                elif 'weight_quant'in fullname and'alh' in fullname:
                    weight_scales.append({fullname:param.clone().detach().cpu().squeeze().numpy()})
                elif 'weight_quant'in fullname and'bit' in fullname:
                    weight_bits.append({fullname:param.clone().detach().cpu().squeeze().numpy()})
        return {'spike_scales':spike_scales,
                'feat_scales':feat_scales,
                'feat_bits':feat_bits,
                'weight_scales':weight_scales,
                'weight_bits':weight_bits,
                }
        
    
    def get_memory_loss(self, target_weight_bit=4, target_fea_bit=4, weight_bit_penalty=1e-2, fea_bit_penalty=1e-2):           
        # q_info.append({
                        # 'fea_bit': self.fea_quant.bit.squeeze(),
                        # 'fea_scale': self.fea_quant.gama.squeeze(),
                        # 'weight_bit': self.weight_quant.bit.squeeze(),
                        # 'weight_scale': self.weight_quant.alh.squeeze(),
                        # 'layer_type':'linear',
                        # 'fea_numel': fea_q.numel(),
                        # 'weight_numel':weight_q.numel(),
            #             })
        weight_bit_penalty = weight_bit_penalty if weight_bit_penalty is not None else 1.0
        fea_bit_penalty = fea_bit_penalty if fea_bit_penalty is not None else 1.0
        fea_bit_sum = []
        weight_bit_sum = []
        tar_fea_bit_sum = []
        tar_weight_bit_sum = []
        for q_inf in self.q_info:
            fea_bit_sum.append(q_inf['fea_bit'].mean() * q_inf['fea_numel'])
            weight_bit_sum.append(q_inf['weight_bit'].mean() * q_inf['weight_numel'])
            tar_fea_bit_sum.append(torch.tensor([target_fea_bit * q_inf['fea_numel']]))
            tar_weight_bit_sum.append(torch.tensor([target_weight_bit * q_inf['weight_numel']]))
        fea_bit_sum = torch.stack(fea_bit_sum).sum() /8/1024               # in 1 KB = 1024*8 bits
        weight_bit_sum = torch.stack(weight_bit_sum).sum() /8/1024        # in 1 KB = 1024*8 bits
        tar_fea_bit_sum = (torch.stack(tar_fea_bit_sum).sum()/8/1024).to(fea_bit_sum.device)    # in 1 KB = 1024*8 bits
        tar_weight_bit_sum = (torch.stack(tar_weight_bit_sum).sum()/8/1024).to(weight_bit_sum.device)   # in 1 KB = 1024*8 bits
        loss_fea_bit = fea_bit_penalty * (fea_bit_sum - tar_fea_bit_sum) ** 2
        loss_weight_bit = weight_bit_penalty  * (weight_bit_sum - tar_weight_bit_sum) ** 2
        # print(loss_fea_bit)
        return loss_fea_bit, loss_weight_bit
    
    
    def get_bit_loss(self, target_weight_bit=4, target_fea_bit=4, weight_bit_penalty=1e-2, fea_bit_penalty=1e-2):           
        # q_info.append({
                        # 'fea_bit': self.fea_quant.bit.squeeze(),
                        # 'fea_scale': self.fea_quant.gama.squeeze(),
                        # 'weight_bit': self.weight_quant.bit.squeeze(),
                        # 'weight_scale': self.weight_quant.alh.squeeze(),
                        # 'layer_type':'linear',
                        # 'fea_numel': fea_q.numel(),
                        # 'weight_numel':weight_q.numel(),
            #             })
        weight_bit_penalty = weight_bit_penalty if weight_bit_penalty is not None else 1.0
        fea_bit_penalty = fea_bit_penalty if fea_bit_penalty is not None else 1.0
        fea_bit_sum = []
        weight_bit_sum = []
        fea_numel_sum = []
        weight_numel_sum = []
        for q_inf in self.q_info:
            fea_bit_sum.append(q_inf['fea_bit'].mean() * q_inf['fea_numel'])
            weight_bit_sum.append(q_inf['weight_bit'].mean() * q_inf['weight_numel'])
            fea_numel_sum.append(torch.tensor([q_inf['fea_numel']]))
            weight_numel_sum.append(torch.tensor([q_inf['weight_numel']]))
        fea_bit_sum = torch.stack(fea_bit_sum).sum()               # in 1 KB = 1024*8 bits
        weight_bit_sum = torch.stack(weight_bit_sum).sum()         # in 1 KB = 1024*8 bits
        fea_numel_sum = (torch.stack(fea_numel_sum).sum()).to(fea_bit_sum.device)    # in 1 KB = 1024*8 bits
        weight_numel_sum = (torch.stack(weight_numel_sum).sum()).to(weight_bit_sum.device)   # in 1 KB = 1024*8 bits
        
        loss_fea_bit = fea_bit_penalty * torch.norm(fea_bit_sum/fea_numel_sum - target_fea_bit) 
        loss_weight_bit = weight_bit_penalty  * torch.norm(weight_bit_sum/weight_numel_sum - target_weight_bit) 
        # print(loss_fea_bit)
        return loss_fea_bit, loss_weight_bit


    def print_t(self):
        for n, p in self.named_parameters():
            if 'threshold' in n:
                print('Param {}, Value {}'.format(n, p.data.item()))




class ResNet_Cifar_real_q(nn.Module):

    def __init__(self, block, layers, input_c=3, num_classes=10, rp = False, **kwargs):
        super().__init__()

        global BN
        global layer_num
        BN = nn.BatchNorm2d
        global ReLU
        ReLU = nn.ReLU
        
        # self.scale = nn.Parameter(torch.ones(1, 1, 2048), requires_grad=True)
        self.rp = False 
        inplanes = 16
        self.inplanes = 16
        self.conv1 = qConv2d(input_c, inplanes, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn1 = BN(self.inplanes)
        self.relu = ReLU(inplace=True)
        self.layer1 = self._make_layer(block, inplanes, layers[0], fea_all_positive=False, **kwargs)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2, fea_all_positive=False, **kwargs)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2, fea_all_positive=False, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = qLinear(inplanes * 4 * block.expansion, num_classes, fea_all_positive=False, **kwargs)
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
        self.q_info = []
        if not feat is None:
            x = self.fc(feat)
            return x
            
        else:
            x, q_info = self.conv1(x, self.q_info)
            x = self.bn1(x)
            x = self.relu(x)

            x, q_info = self.layer1((x, q_info))
            x, q_info = self.layer2((x, q_info))
            x, q_info = self.layer3((x, q_info))
            
            if is_drop:
                x = self.avgpool(x)
            else:
                x = self.avgpool(x)
            
            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
                fea = x
            elif len(x.shape) == 5:
                x = x.view(x.size(0), x.size(1), -1)
                fea = x.mean([0])
            x, q_info = self.fc(x, q_info)
            
            # layer_num = 0
            # for idx, q_inf in enumerate(q_info):
            #     print('layer {}'.format(idx))
            #     print('fea_bit:', q_inf['fea_bit'])
            #     print('weight_bit:', q_inf['weight_bit'])
            
            if is_adain:
                return fea,x
            else:
                return x
            

    # def bit_loss(self):
    @torch.no_grad()
    def get_bits_scales(self):
        # if self.fc.out_features == 100 or self.fc.out_features == 10:
        #     x = torch.randn(1,3,32,32).to(self.conv1.weight.device)
        #     self.forward(x, is_adain=False, is_drop=False, feat=None)
        #     return self.q_info
        spike_scales = []
        feat_scales = []
        feat_bits = []
        weight_scales = []
        weight_bits = []
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if 'fire_ratio' in fullname:
                    spike_scales.append({fullname:param.clone().detach().cpu().squeeze().numpy()})
                elif 'fea_quant'in fullname and 'gama' in fullname:
                    feat_scales.append({fullname:param.clone().detach().cpu().squeeze().numpy()})
                elif 'fea_quant'in fullname and 'bit' in fullname:
                    feat_bits.append({fullname:param.clone().detach().cpu().squeeze().numpy()})
                elif 'weight_quant'in fullname and'alh' in fullname:
                    weight_scales.append({fullname:param.clone().detach().cpu().squeeze().numpy()})
                elif 'weight_quant'in fullname and'bit' in fullname:
                    weight_bits.append({fullname:param.clone().detach().cpu().squeeze().numpy()})
        return {'spike_scales':spike_scales,
                'feat_scales':feat_scales,
                'feat_bits':feat_bits,
                'weight_scales':weight_scales,
                'weight_bits':weight_bits,
                }
        
    
    def get_memory_loss(self, target_weight_bit=4, target_fea_bit=4, weight_bit_penalty=1e-2, fea_bit_penalty=1e-2):           
        # q_info.append({
                        # 'fea_bit': self.fea_quant.bit.squeeze(),
                        # 'fea_scale': self.fea_quant.gama.squeeze(),
                        # 'weight_bit': self.weight_quant.bit.squeeze(),
                        # 'weight_scale': self.weight_quant.alh.squeeze(),
                        # 'layer_type':'linear',
                        # 'fea_numel': fea_q.numel(),
                        # 'weight_numel':weight_q.numel(),
            #             })
        weight_bit_penalty = weight_bit_penalty if weight_bit_penalty is not None else 1.0
        fea_bit_penalty = fea_bit_penalty if fea_bit_penalty is not None else 1.0
        fea_bit_sum = []
        weight_bit_sum = []
        tar_fea_bit_sum = []
        tar_weight_bit_sum = []
        for q_inf in self.q_info:
            fea_bit_sum.append(q_inf['fea_bit'].mean() * q_inf['fea_numel'])
            weight_bit_sum.append(q_inf['weight_bit'].mean() * q_inf['weight_numel'])
            tar_fea_bit_sum.append(torch.tensor([target_fea_bit * q_inf['fea_numel']]))
            tar_weight_bit_sum.append(torch.tensor([target_weight_bit * q_inf['weight_numel']]))
        fea_bit_sum = torch.stack(fea_bit_sum).sum() /8/1024               # in 1 KB = 1024*8 bits
        weight_bit_sum = torch.stack(weight_bit_sum).sum() /8/1024        # in 1 KB = 1024*8 bits
        tar_fea_bit_sum = (torch.stack(tar_fea_bit_sum).sum()/8/1024).to(fea_bit_sum.device)    # in 1 KB = 1024*8 bits
        tar_weight_bit_sum = (torch.stack(tar_weight_bit_sum).sum()/8/1024).to(weight_bit_sum.device)   # in 1 KB = 1024*8 bits
        loss_fea_bit = fea_bit_penalty * (fea_bit_sum - tar_fea_bit_sum) ** 2
        loss_weight_bit = weight_bit_penalty  * (weight_bit_sum - tar_weight_bit_sum) ** 2
        # print(loss_fea_bit)
        return loss_fea_bit, loss_weight_bit
    
    
    def get_bit_loss(self, target_weight_bit=4, target_fea_bit=4, weight_bit_penalty=1e-2, fea_bit_penalty=1e-2):           
        # q_info.append({
                        # 'fea_bit': self.fea_quant.bit.squeeze(),
                        # 'fea_scale': self.fea_quant.gama.squeeze(),
                        # 'weight_bit': self.weight_quant.bit.squeeze(),
                        # 'weight_scale': self.weight_quant.alh.squeeze(),
                        # 'layer_type':'linear',
                        # 'fea_numel': fea_q.numel(),
                        # 'weight_numel':weight_q.numel(),
            #             })
        weight_bit_penalty = weight_bit_penalty if weight_bit_penalty is not None else 1.0
        fea_bit_penalty = fea_bit_penalty if fea_bit_penalty is not None else 1.0
        fea_bit_sum = []
        weight_bit_sum = []
        fea_numel_sum = []
        weight_numel_sum = []
        for q_inf in self.q_info:
            fea_bit_sum.append(q_inf['fea_bit'].mean() * q_inf['fea_numel'])
            weight_bit_sum.append(q_inf['weight_bit'].mean() * q_inf['weight_numel'])
            fea_numel_sum.append(torch.tensor([q_inf['fea_numel']]))
            weight_numel_sum.append(torch.tensor([q_inf['weight_numel']]))
        fea_bit_sum = torch.stack(fea_bit_sum).sum()               # in 1 KB = 1024*8 bits
        weight_bit_sum = torch.stack(weight_bit_sum).sum()         # in 1 KB = 1024*8 bits
        fea_numel_sum = (torch.stack(fea_numel_sum).sum()).to(fea_bit_sum.device)    # in 1 KB = 1024*8 bits
        weight_numel_sum = (torch.stack(weight_numel_sum).sum()).to(weight_bit_sum.device)   # in 1 KB = 1024*8 bits
        
        loss_fea_bit = fea_bit_penalty * torch.norm(fea_bit_sum/fea_numel_sum - target_fea_bit) 
        loss_weight_bit = weight_bit_penalty  * torch.norm(weight_bit_sum/weight_numel_sum - target_weight_bit) 
        # print(loss_fea_bit)
        return loss_fea_bit, loss_weight_bit


    def print_t(self):
        for n, p in self.named_parameters():
            if 'threshold' in n:
                print('Param {}, Value {}'.format(n, p.data.item()))



class ResNet(nn.Module):

    def __init__(self, block, layers, input_c=3, num_classes=1000):
        super(ResNet, self).__init__()

        global BN
        BN = nn.BatchNorm2d
        global ReLU
        ReLU = nn.ReLU
        inplanes = 64
        self.inplanes = 64
        self.conv1 = qConv2d(input_c, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BN(self.inplanes)
        self.relu = ReLU(inplace=True)
        self.avgpool1 = nn.AvgPool2d(3,2,1)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = qLinear(inplanes * 8 * block.expansion, num_classes)

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

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                qConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BN(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, is_adain=False, is_drop=False, feat=None):
        if not feat is None:
            x = self.fc(feat)
            return x         
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.avgpool1(x)

            temp, x = self.layer1((x,x))
            temp, x = self.layer2((temp,x))
            temp, x = self.layer3((temp,x))
            temp, x = self.layer4((temp,x))
            
            if is_drop:
                x = self.avgpool2(temp)
            else:
                x = self.avgpool2(x)
                
            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
                fea = x
            elif len(x.shape) == 5:
                x = x.view(x.size(0), x.size(1), -1)
                fea = x.mean([0])
            x = self.fc(x)
            
            if is_adain:
                return fea,x
            else:
                return x


    def print_t(self):
        for n, p in self.named_parameters():
            if 'threshold' in n:
                print('Param {}, Value {}'.format(n, p.data.item()))


class ResNet_q(nn.Module):
    def __init__(self, block, layers, input_c=3, num_classes=1000, **kwargs):
        super().__init__()

        global BN
        BN = nn.BatchNorm2d
        global ReLU
        ReLU = nn.ReLU
        inplanes = 64
        self.inplanes = 64
        self.conv1 = qConv2d(input_c, inplanes, kernel_size=7, stride=2, padding=3, bias=False, **kwargs)
        self.bn1 = BN(self.inplanes)
        self.relu = ReLU(inplace=True)
        self.avgpool1 = nn.AvgPool2d(3,2,1)
        self.layer1 = self._make_layer(block, inplanes, layers[0], fea_all_positive=False, **kwargs)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2, fea_all_positive=False, **kwargs)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2, fea_all_positive=False, **kwargs)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2, fea_all_positive=False, **kwargs)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = qLinear(inplanes * 8 * block.expansion, num_classes, fea_all_positive=False, **kwargs)
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
        self.q_info = []
        if not feat is None:
            x = self.fc(feat)
            return x         
        else:
            x, q_info = self.conv1(x, self.q_info)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.avgpool1(x)

            x, q_info = self.layer1((x, q_info))
            x, q_info = self.layer2((x, q_info))
            x, q_info = self.layer3((x, q_info))
            x, q_info = self.layer4((x, q_info))
            
            if is_drop:
                x = self.avgpool2(x)
            else:
                x = self.avgpool2(x)
                
            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
                fea = x
            elif len(x.shape) == 5:
                x = x.view(x.size(0), x.size(1), -1)
                fea = x.mean([0])
            x, q_info = self.fc(x, q_info)
            
            if is_adain:
                return fea,x
            else:
                return x


    def print_t(self):
        for n, p in self.named_parameters():
            if 'threshold' in n:
                print('Param {}, Value {}'.format(n, p.data.item()))



class ResNet_cifar_q(nn.Module):
    def __init__(self, block, layers, input_c=3, num_classes=1000, **kwargs):
        super().__init__()

        global BN
        BN = nn.BatchNorm2d
        global ReLU
        ReLU = nn.ReLU
        inplanes = 64
        self.inplanes = 64
        self.conv1 = qConv2d(input_c, inplanes, kernel_size=3, stride=1, padding=1, bias=False, **kwargs)
        self.bn1 = BN(self.inplanes)
        self.relu = ReLU(inplace=True)
        # self.avgpool1 = nn.AvgPool2d(3,2,1)
        self.layer1 = self._make_layer(block, inplanes, layers[0], fea_all_positive=False, **kwargs)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2, fea_all_positive=False, **kwargs)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2, fea_all_positive=False, **kwargs)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2, fea_all_positive=False, **kwargs)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = qLinear(inplanes * 8 * block.expansion, num_classes, fea_all_positive=False, **kwargs)
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
        self.q_info = []
        if not feat is None:
            x = self.fc(feat)
            return x         
        else:
            x, q_info = self.conv1(x, self.q_info)
            x = self.bn1(x)
            x = self.relu(x)
            # x = self.avgpool1(x)

            x, q_info = self.layer1((x, q_info))
            x, q_info = self.layer2((x, q_info))
            x, q_info = self.layer3((x, q_info))
            x, q_info = self.layer4((x, q_info))
            
            if is_drop:
                x = self.avgpool2(x)
            else:
                x = self.avgpool2(x)
                
            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
                fea = x
            elif len(x.shape) == 5:
                x = x.view(x.size(0), x.size(1), -1)
                fea = x.mean([0])
            x, q_info = self.fc(x, q_info)
            
            if is_adain:
                return fea,x
            else:
                return x


    def print_t(self):
        for n, p in self.named_parameters():
            if 'threshold' in n:
                print('Param {}, Value {}'.format(n, p.data.item()))
                

class SEWResNet_q(nn.Module):
    def __init__(self, block, layers, input_c=3, num_classes=1000, **kwargs):
        super().__init__()

        global BN
        BN = nn.BatchNorm2d
        global ReLU
        ReLU = nn.ReLU
        inplanes = 64
        self.inplanes = 64
        self.conv1 = qConv2d(input_c, inplanes, kernel_size=7, stride=2, padding=3, bias=False, **kwargs)
        self.bn1 = BN(self.inplanes)
        self.relu = ReLU(inplace=True)
        self.avgpool1 = nn.MaxPool2d(3,2,1)
        self.layer1 = self._make_layer(block, inplanes, layers[0], fea_all_positive=False, **kwargs)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2, fea_all_positive=False, **kwargs)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2, fea_all_positive=False, **kwargs)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2, fea_all_positive=False, **kwargs)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = qLinear(inplanes * 8 * block.expansion, num_classes, fea_all_positive=False, **kwargs)
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
            downsample = SEWbasic_downsample(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, fea_all_positive=fea_all_positive, **kwargs)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, fea_all_positive=fea_all_positive, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, fea_all_positive=fea_all_positive, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x, is_adain=False, is_drop=False, feat=None):
        self.q_info = []
        if not feat is None:
            x = self.fc(feat)
            return x         
        else:
            x, q_info = self.conv1(x, self.q_info)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.avgpool1(x)

            x, q_info = self.layer1((x, q_info))
            x, q_info = self.layer2((x, q_info))
            x, q_info = self.layer3((x, q_info))
            x, q_info = self.layer4((x, q_info))
            
            if is_drop:
                x = self.avgpool2(x)
            else:
                x = self.avgpool2(x)
                
            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
                fea = x
            elif len(x.shape) == 5:
                x = x.view(x.size(0), x.size(1), -1)
                fea = x.mean([0])
            x, q_info = self.fc(x, q_info)
            
            if is_adain:
                return fea,x
            else:
                return x


    def print_t(self):
        for n, p in self.named_parameters():
            if 'threshold' in n:
                print('Param {}, Value {}'.format(n, p.data.item()))



class ResNet_Cifar_Modified_q(nn.Module):

    def __init__(self, block, layers, num_classes=10, input_c=3, rp = False):
        super(ResNet_Cifar_Modified_q, self).__init__()

        global BN
        BN = nn.BatchNorm2d
        global ReLU
        ReLU = nn.ReLU
        
        self.scale = nn.Parameter(torch.ones(1, 1, 512), requires_grad=True)
        self.rp = rp

        self.inplanes = 64
        self.conv1 = nn.Sequential(
            qConv2d(input_c, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BN(64),
            ReLU(inplace=True),
            qConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BN(64),
            ReLU(inplace=True),
            qConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BN(64),
            ReLU(inplace=True),
        )
        self.avgpool = nn.AvgPool2d(2)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = qLinear(512, num_classes)

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

        # zero_init_residual:
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # AvgDown Layer
            downsample = nn.Sequential(
                nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                qConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                BN(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, is_adain=False, is_drop=False, feat=None):
        
        if not feat is None:
            x = self.fc(feat)
            return x    
        else:    
            x = self.conv1(x)
            x = self.avgpool(x)
            temp, x = self.layer1((x,x))
            temp, x = self.layer2((temp,x))
            temp, x = self.layer3((temp,x))
            temp, x = self.layer4((temp,x))
            #print(temp.sum())
            #print(x.sum())
            if is_drop:
                x = self.avgpool2(temp)
            else:
                x = self.avgpool2(x)
            
            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
                fea = x
            elif len(x.shape) == 5:
                x = x.view(x.size(0), x.size(1), -1)
                
                if self.rp:
                    x = x*self.scale
                fea = x.mean([0])
            x = self.fc(x)
            if is_adain:
                return fea,x
            else:
                return x


def ResNet18_q(**kwargs):
    return ResNet_q(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet18_cifar_q(**kwargs):
    return ResNet_cifar_q(BasicBlock, [2, 2, 2, 2], **kwargs)


def SEWResNet18_q(**kwargs):
    return SEWResNet_q(SEWBasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34_q(**kwargs):
    return ResNet_q(BasicBlock, [3, 4, 6, 3], **kwargs)


def SEWResNet34_q(**kwargs):
    return SEWResNet_q(SEWBasicBlock, [3, 4, 6, 3], **kwargs)


def resnet20_cifar_q(**kwargs):
    model = ResNet_Cifar_q(BasicBlock, [3, 3, 3], **kwargs)
    return model



def resnet20_cifar_real_q(**kwargs):
    model = ResNet_Cifar_real_q(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet20_cifar_modified_q(**kwargs):
    model = ResNet_Cifar_Modified_q(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet19_cifar_q(**kwargs):
    model = ResNet_Cifar_q(BasicBlock, [3, 3, 2], **kwargs)
    return model


def resnet32_cifar_q(**kwargs):
    model = ResNet_Cifar_q(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar_q(**kwargs):
    model = ResNet_Cifar_q(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar_q(**kwargs):
    model = ResNet_Cifar_q(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar_q(**kwargs):
    model = ResNet_Cifar_q(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar_q(**kwargs):
    model = ResNet_Cifar_q(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar_q(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar_q(**kwargs):
    model = ResNet_Cifar_q(Bottleneck, [111, 111, 111], **kwargs)
    return model


if __name__ == '__main__':
    net = resnet20_cifar()
    net.eval()
    x = torch.randn(1, 3, 32, 32)
    print(net)
    y1 = net(x)