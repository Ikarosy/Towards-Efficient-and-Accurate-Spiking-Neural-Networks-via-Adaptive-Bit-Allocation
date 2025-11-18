import os

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.autograd import Variable
from DCLS.construct.modules import Dcls1d, _DclsNd, ConstructKernel1d
from IPython import embed
def init_weights(net):
    """the weights of conv layer and fully connected layers 
    are both initilized with Xavier algorithm, In particular,
    we set the parameters to random values uniformly drawn from [-a, a]
    where a = sqrt(6 * (din + dout)), for batch normalization 
    layers, y=1, b=0, all bias initialized to 0.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0.5)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return net

def split_weights(net):
    decay = []
    no_decay = []
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for module_name, module in net.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname or 'fire_ratio' in fullname or 'bit' in fullname or 'gama' in fullname or 'alh' in fullname:  # bias (no decay)
                no_decay.append(param)
            elif isinstance(module, bn):  # weight (no decay)
                no_decay.append(param)
            else:
                decay.append(param)
                
    
    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]

def classify_params(model):
    all_params = model.parameters()
    spike_scales = []
    
    
    feat_scales = []
    feat_bits = []
    feat_shifts = []
    
    weigth_scales = []
    weight_bits = []
    weight_shifts = []
    
    time_steps = []
    
    weights = []
    other_no_decay = []
     
    positions = [] #snn_delays P
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if 'fire_ratio' in fullname:
                spike_scales.append(param)
            elif 'T' in fullname and (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)):
                time_steps.append(param)
            
            
            elif ('fea_quant'in fullname or 'spikes_quant' in fullname)  and 'gama' in fullname:
                feat_scales.append(param)
            elif ('fea_quant'in fullname or 'spikes_quant' in fullname) and 'bit' in fullname:
                feat_bits.append(param)
            elif ('fea_quant'in fullname or 'spikes_quant' in fullname) and 'bin_shift' in fullname:
                feat_shifts.append(param)
            
            
            elif 'weight_quant'in fullname and'alh' in fullname:
                weigth_scales.append(param)
            elif 'weight_quant'in fullname and'bit' in fullname:
                weight_bits.append(param)
            
            
            elif isinstance(module, bn):  # weight (no decay)
                other_no_decay.append(param)
            elif "bias" in fullname:
                other_no_decay.append(param)
            elif 'weight' in fullname:
                weights.append(param)
            elif isinstance(module, _DclsNd) and 'P' in fullname:
                positions.append(param)
                
    params_id = list(map(id,spike_scales))+list(map(id,feat_scales))+list(map(id,feat_bits))+list(map(id,weigth_scales))+list(map(id,weight_bits))\
    +list(map(id,weights))+list(map(id,other_no_decay))+list(map(id,time_steps))+list(map(id,feat_shifts))+list(map(id,positions))
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    return spike_scales, feat_scales, feat_bits, weigth_scales, weight_bits, weights, other_no_decay, other_params, time_steps, feat_shifts, positions


    
def init_bias(net):
    """the weights of conv layer and fully connected layers 
    are both initilized with Xavier algorithm, In particular,
    we set the parameters to random values uniformly drawn from [-a, a]
    where a = sqrt(6 * (din + dout)), for batch normalization 
    layers, y=1, b=0, all bias initialized to 0.
    """
    for m in net.modules():
            
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            #nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0.5)


    return net