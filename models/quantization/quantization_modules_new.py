import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantization_funcs import *
weight_bit = 4
fea_bit = 1
# bit_learnable = True


layer_num = 0
def dect_errTensor(x, name='conv_feature', layer_count=True):
    global layer_num
    is_inf_or_nan = torch.isinf(x) | torch.isnan(x)
    inf_or_nan_indices = torch.nonzero(is_inf_or_nan).squeeze()
    if inf_or_nan_indices.numel() != 0:
        print('=='*20)
        print('layer {}'.format(layer_num))
        print('wrong tensor name:', name)
        # print('wrong elements:', x[is_inf_or_nan])
        # print('located in ', inf_or_nan_indices)
    if layer_count:
        layer_num += 1

#################################################################################
#
#                   quantization functions for FM and weights of nn.modules
#
#################################################################################


# class u_quant_weight_linear(nn.Module):
#     '''
#         weight uniform quantization for linears.
#     '''
#     def __init__(self, out_channels, bit, alh_init=0.1, alh_std=0.1, learn_weight_bit=True, weight_per_layer=False):
#         super(u_quant_weight_linear, self).__init__()
#         self.bit_bound = bit + 2
#         self.bit = bit
#         # quantizer
#         self.quant_weight_func = u_quant_w_func_alpha_linear_channelwise_div
#         # parameters initialize
#         _init_alh = alh_init
#         # initialize the step size by normal distribution
#         _alh = torch.Tensor(out_channels,1) if not weight_per_layer else torch.Tensor(1,1)
#         self.alh = torch.nn.Parameter(_alh)
#         torch.nn.init.normal_(self.alh, mean=alh_init, std=alh_std)
#         self.alh = torch.nn.Parameter(self.alh.abs())
        
#         # initialize the bit bit=4
#         _init_bit = bit
#         self.bit = torch.nn.Parameter(torch.Tensor(out_channels,1)) if not weight_per_layer else torch.nn.Parameter(torch.Tensor(1,1))
#         torch.nn.init.constant_(self.bit,_init_bit)
#         self.learn_weight_bit = learn_weight_bit
#         if not learn_weight_bit:
#             self.bit.requires_grad = False
#             print('not learn linear weight bit')
        
#     def forward(self, weight, init_weight_bit):
#         # quantization
#         # weight_q = self.quant_weight_func.apply(weight, self.alh, self.bit, init_weight_bit)
#         # bit is tooooo easy to fly so we do clamp
#         bit = torch.clamp(self.bit.abs(), 1, self.bit_bound).detach() + self.bit - self.bit.detach()
#         alh = self.alh.abs()
#         weight_q = self.quant_weight_func.apply(weight, alh, bit, self.learn_weight_bit)
#         return weight_q




class u_quant_fea(nn.Module):
    '''
        feature uniform quantization.
    '''
    def __init__(self, in_channels, bit, gama_init=0.001,gama_std=0.001, quant_fea=True, learn_fea_bit=True):
        super(u_quant_fea, self).__init__()
        self.bit_bound = 2 + bit
        self.bit   = bit
        self.cal_mode = 1
        self.quant_fea = quant_fea
        self.deg_inv_sqrt = 1.0
        # quantizer
        self.quant_fea_func = u_quant_fea_func_gama_layerwise_div if in_channels is None else u_quant_fea_func_gama_channelwise_div
        
        # parameters initialization
        # initialize step size
        self.gama = torch.nn.Parameter(torch.Tensor(1)) if in_channels is None else torch.nn.Parameter(torch.Tensor(1,in_channels,1,1))
        torch.nn.init.normal_(self.gama, mean=gama_init, std=gama_std)
        self.gama = torch.nn.Parameter(self.gama.abs())
        # initialize the bit, bit=4
        self.bit = torch.nn.Parameter(torch.Tensor(1)) if in_channels is None else torch.nn.Parameter(torch.Tensor(1,in_channels,1,1))
        _init_bit = bit
        torch.nn.init.constant_(self.bit,_init_bit)
        self.learn_fea_bit = learn_fea_bit
        if not self.learn_fea_bit:
            self.bit.requires_grad = False
            
    def forward(self, fea, init_fea_bit):
        if(not self.quant_fea):
            fea_q = fea
        else:
            # fea_q = self.quant_fea_func.apply(fea, self.gama, self.bit, init_fea_bit)
            # bit is tooooo easy to fly so we do clamp
            bit = torch.clamp(self.bit.abs(), 1, self.bit_bound).detach() + self.bit - self.bit.detach()
            gama = self.gama.abs()
            fea_q = self.quant_fea_func.apply(fea, gama, bit, self.learn_fea_bit)
        return fea_q
    
    
    
    
    
#################################################################################
#
#                   new modules
#
#################################################################################
# from quantization_funcs import LsqQuan


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class spike_quant_fea(nn.Module):
    '''
        spike feature quantization.
    '''
    def __init__(self, in_channels, bit, gama_init=0.001,gama_std=0.001, quant_fea=True, learn_spike_bit=True, all_positive=True):
        super().__init__()
        #bit settings
        self.bit_bound = bit + 2
        self.bit = bit
        self.all_positive = all_positive
        print("all_positive:", all_positive)
        # quantizer
        # self.quant_fea_func = spike_quant_fea_func_gama_layerwise_div if in_channels is None else spike_quant_fea_func_gama_channelwise_div
        
        # parameters initialization
        # initialize step size
        self.gama = torch.nn.Parameter(torch.Tensor(1)) if in_channels is None else torch.nn.Parameter(torch.Tensor(1,in_channels,1,1))
        torch.nn.init.normal_(self.gama, mean=gama_init, std=gama_std)
        self.gama = torch.nn.Parameter(self.gama.abs())
        # initialize the bit, bit=4
        self.bit = torch.nn.Parameter(torch.Tensor(1)) if in_channels is None else torch.nn.Parameter(torch.Tensor(1,in_channels,1,1))
        _init_bit = bit
        torch.nn.init.constant_(self.bit,_init_bit)
        self.learn_spike_bit = learn_spike_bit
        if not learn_spike_bit:
            self.bit.requires_grad = False
            print('not learn spike bit')

    def forward(self, fea):
        # bit is tooooo easy to fly so we do clamp
        # print(self.bit.grad)
        bit = torch.clamp(self.bit.abs(), 1, self.bit_bound).detach() + self.bit - self.bit.detach()
        if self.all_positive:
            thd_neg = torch.zeros(1).to(fea.device)
            thd_pos = 2 ** bit - 1
        else:
            thd_neg = - 2 ** (bit - 1) + 1
            thd_pos = 2 ** (bit - 1) - 1
    
        s_grad_scale = 1.0 / ((thd_pos.detach() * fea.numel()) ** 0.5)
        gama = self.gama
        gama_gs = grad_scale(gama, s_grad_scale)    
        
        #expand
        # gama_gs.expand(fea)
        fea = fea / gama_gs
        fea = torch.clamp(fea, thd_neg, thd_pos)
        fea = round_pass(fea)
        fea_q = fea * gama_gs            
        return fea_q


class u_quant_weight_conv(nn.Module):
    '''
        weight uniform quantization for linears.
    '''
    def __init__(self, out_channels, bit, alh_init=0.1,alh_std=0.1, learn_weight_bit=True, weight_per_layer=False):
        super(u_quant_weight_conv, self).__init__()
        #bit settings
        self.bit_bound = bit + 2
        self.bit = bit
        # quantizer
        # self.quant_weight_func = u_quant_w_func_alpha_conv_channelwise_div

        # parameters initialize
        # _init_alh = alh_init
        # initialize the step size by normal distribution
        _alh = torch.Tensor(out_channels,1,1,1) if not weight_per_layer else torch.Tensor(1,1,1,1)
        self.alh = torch.nn.Parameter(_alh)
        torch.nn.init.normal_(self.alh, mean=alh_init, std=alh_std)
        self.alh = torch.nn.Parameter(self.alh.abs())
       
        # initialize the bit bit=4
        _init_bit = bit
        self.bit = torch.nn.Parameter(torch.Tensor(out_channels,1,1,1)) if not weight_per_layer else torch.nn.Parameter(torch.Tensor(1,1,1,1))
        torch.nn.init.constant_(self.bit,_init_bit)
        self.learn_weight_bit = learn_weight_bit
        if not learn_weight_bit:
            self.bit.requires_grad = False
            print('not learn conv weight bit')
        if weight_per_layer:
            print('using per_layer quant for weights')
        else:
            print('using per_channel quant for weights')
            
    def forward(self, weight, init_weight_bit):
        # quantization
        # bit is tooooo easy to fly so we do clamp
        # print(self.bit.grad)
        bit = torch.clamp(self.bit.abs(), 1, self.bit_bound).detach() + self.bit - self.bit.detach()
        thd_neg = - 2 ** (bit - 1) + 1
        thd_pos = 2 ** (bit - 1) - 1
    
        s_grad_scale = 1.0 / ((thd_pos.detach() * weight.numel()) ** 0.5)
        alh = self.alh
        alh_gs = grad_scale(alh, s_grad_scale)    

        #expand
        # alh_gs.expand(weight)
        weight = weight / alh_gs
        weight = torch.clamp(weight, thd_neg.detach(), thd_pos)
        weight = round_pass(weight)
        weight_q = weight * alh_gs
        return weight_q
    

class u_quant_weight_linear(nn.Module):
    '''
        weight uniform quantization for linears.
    '''
    def __init__(self, out_channels, bit, alh_init=0.1,alh_std=0.1, learn_weight_bit=True, weight_per_layer=False):
        super(u_quant_weight_linear, self).__init__()
        #bit settings
        self.bit_bound = bit + 2
        self.bit = bit

        # quantizer
        # self.quant_weight_func = u_quant_w_func_alpha_linear_channelwise_div
        # parameters initialize
        # _init_alh = alh_init
        # initialize the step size by normal distribution
        _alh = torch.Tensor(out_channels,1) if not weight_per_layer else torch.Tensor(1,1)
        self.alh = torch.nn.Parameter(_alh)
        torch.nn.init.normal_(self.alh, mean=alh_init, std=alh_std)
        self.alh = torch.nn.Parameter(self.alh.abs())
        
        # initialize the bit bit=4
        _init_bit = bit
        self.bit = torch.nn.Parameter(torch.Tensor(out_channels,1)) if not weight_per_layer else torch.nn.Parameter(torch.Tensor(1,1))
        torch.nn.init.constant_(self.bit,_init_bit)
        self.learn_weight_bit = learn_weight_bit
        if not learn_weight_bit:
            self.bit.requires_grad = False
            print('not learn linear weight bit')
            
        
    def forward(self, weight, init_weight_bit):
        # quantization
        # bit is tooooo easy to fly so we do clamp
        # print(self.bit.grad)
        bit = torch.clamp(self.bit.abs(), 1, self.bit_bound).detach() + self.bit - self.bit.detach()
        thd_neg = - 2 ** (bit - 1) + 1
        thd_pos = 2 ** (bit - 1) - 1
    
        s_grad_scale = 1.0 / ((thd_pos.detach() * weight.numel()) ** 0.5)
        alh = self.alh
        alh_gs = grad_scale(alh, s_grad_scale)    

        #expand
        # alh_gs.expand(weight)
        weight = weight / alh_gs
        weight = torch.clamp(weight, thd_neg.detach(), thd_pos)
        weight = round_pass(weight)
        weight_q = weight * alh_gs
        return weight_q