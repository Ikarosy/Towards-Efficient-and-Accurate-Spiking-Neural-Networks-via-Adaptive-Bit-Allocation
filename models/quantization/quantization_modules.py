import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantization_funcs import *
from .q_search import MSEObserver

weight_bit = 3
fea_bit = 2

bit_detect = False
# bit_learnable = True

renew_switch = True
imagenet = True


def _transform_to_ch_axis(x, ch_axis):
    new_axis_list = [i for i in range(x.dim())]
    new_axis_list[ch_axis] = 0
    new_axis_list[0] = ch_axis
    x_channel = x.permute(new_axis_list)
    y = torch.flatten(x_channel, start_dim=1)
    return y

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


class u_quant_weight_linear(nn.Module):
    '''
        weight uniform quantization for linears.
    '''
    def __init__(self, out_channels, bit, alh_init=0.1, alh_std=0.1, learn_weight_bit=True, weight_per_layer=False):
        super(u_quant_weight_linear, self).__init__()
        self.bit_bound = bit + 2
        self.bit = bit
        # quantizer
        self.quant_weight_func = u_quant_w_func_alpha_linear_channelwise_div
        # parameters initialize
        _init_alh = alh_init
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
            
        #bit renew 
        self.per_tensor = weight_per_layer
        self.register_buffer('old_bit', self.bit.clone().detach())
        # self.val_obs = MSEObserver(bit=bit, symmetric=~all_positive, ch_axis=-1 if in_channels is None else 1)
        self.register_buffer("val_min", torch.tensor(float("inf")).expand_as(self.bit))
        self.register_buffer("val_max", torch.tensor(float("-inf")).expand_as(self.bit))
        self.register_buffer("quant_min", torch.min(-(2 ** (self.bit.clone().detach()-1) -1), -torch.ones_like(self.bit)))
        self.register_buffer("quant_max", torch.max(2 ** (self.bit.clone().detach()-1) -1, torch.ones_like(self.bit)))
        # self.search_grid_num = 100
        self.search_grid_num = 50 if imagenet else 100
        self.eps = 0.00001
        self.weight_renew_switch = renew_switch
        self.bit_detect= bit_detect
    
    def lp_loss(self, pred, tgt, p=2.4):
        x = (pred - tgt).abs().pow(p)
        if self.per_tensor:
            return x.mean().reshape(1,1)
        else:
            return x.mean(1).reshape(-1,1)
    
    def loss_fx(self, x, new_min, new_max, bit_clamp):
        scale = self.scale_shift_cal(new_min, new_max)
        fea_q = self.quant_weight_func.apply(x.detach(), scale.detach(), bit_clamp.detach(), self.learn_weight_bit)
        score = self.lp_loss(fea_q, x)
        return score
    
    def search_minmax(self, x, bit_clamp):
        if self.per_tensor:
            x_min, x_max = torch.aminmax(x)
            x_min, x_max = x_min.reshape(1,1), x_max.reshape(1,1)
        else:
            y = _transform_to_ch_axis(x, 1)
            x_min, x_max = torch.aminmax(y, dim=1)
            x_min, x_max = x_min.reshape(-1,1), x_max.reshape(-1,1)
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        for i in range(1, self.search_grid_num+1):
            thres = xrange  * (i / self.search_grid_num)
            new_min = -thres
            new_max = thres
            score = self.loss_fx(x, new_min, new_max, bit_clamp)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max
    
    def qmax_renew(self, bit_round):
        self.quant_min = torch.min(-(2 ** (bit_round-1) -1), -torch.ones_like(bit_round))
        self.quant_max = torch.max(2 ** (bit_round-1) -1, torch.ones_like(bit_round))
    
    @torch.jit.export
    def scale_shift_cal(self, val_min, val_max):
        val_min_neg = torch.min(val_min, torch.zeros_like(val_min))
        val_max_pos = torch.max(val_max, torch.zeros_like(val_max))
        val_max_pos = torch.max(-val_min_neg, val_max_pos)
        scale = val_max_pos / ( (self.quant_max - self.quant_min).float() / 2 )
        scale = torch.where(scale > self.eps, scale, self.eps)
        return scale
    
    #scale renew 
    @torch.jit.export
    def scale_renew(self, bit_detach, fea_detach):
        bit_round = bit_detach.round()
        bit_mark = (self.old_bit!=bit_round)
        if any(bit_mark):   #start renew
            # print('renew act. scale and shift')
            with torch.no_grad():   
                #update scale and shifts
                self.qmax_renew(bit_round)
                best_min, best_max = self.search_minmax(fea_detach, bit_round)
                self.val_min = torch.min(self.val_min, best_min)
                self.val_max = torch.max(self.val_max, best_max)
                _scale = self.scale_shift_cal(self.val_min, self.val_max)
                if self.per_tensor:
                    self.alh.copy_(_scale)
                else:
                    self.alh[bit_mark] = _scale[bit_mark]
                self.old_bit.copy_(bit_round)  
    
    
    #scale renew 
    @torch.jit.export
    def scale_naive_renew(self, bit_detach, fea_detach):
        bit_round = bit_detach.round()
        bit_mark = (self.old_bit!=bit_round)
        if any(bit_mark):   #start renew
            # print('renew act. scale and shift')
            with torch.no_grad():   
                #update scale and shifts
                old_qmin, old_qmax = self.quant_min.clone(), self.quant_max.clone()
                self.qmax_renew(bit_round)
                new_qmin, new_qmax = self.quant_min, self.quant_max
                alh_scale =  (new_qmax-new_qmin) / (old_qmax-old_qmin)
                self.alh.copy_(self.alh * alh_scale)
                self.old_bit.copy_(bit_round)  
                
    def forward(self, weight, init_weight_bit):
        # quantization
        # weight_q = self.quant_weight_func.apply(weight, self.alh, self.bit, init_weight_bit)
        # bit is tooooo easy to fly so we do clamp
        bit = torch.clamp(self.bit.abs(), 1, self.bit_bound).detach() + self.bit - self.bit.detach()
        #scale renew
        if self.training and self.weight_renew_switch:
            self.scale_renew(bit.clone().detach(), weight.detach())
        # bit detect
        if self.training and not self.weight_renew_switch and self.bit_detect:
            bit_detach = bit.clone().detach()
            bit_round = bit_detach.round()
            bit_mark = (self.old_bit!=bit_round)
            if any(bit_mark):   #start renew
                print('weight bit change found in this layer and batch')
            self.old_bit.copy_(bit_round)  
            
        alh = self.alh
        weight_q = self.quant_weight_func.apply(weight, alh, bit, self.learn_weight_bit)
        return weight_q


class u_quant_weight_conv(nn.Module):
    '''
        weight uniform quantization for linears.
    '''
    def __init__(self, out_channels, bit, alh_init=0.1,alh_std=0.1, learn_weight_bit=True, weight_per_layer=False):
        super(u_quant_weight_conv, self).__init__()
        self.bit_bound = bit + 2
        self.bit = bit
        self.weight_per_layer = weight_per_layer
        # quantizer
        self.quant_weight_func = u_quant_w_func_alpha_conv_channelwise_div
        
        # parameters initialize
        _init_alh = alh_init
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
        # if weight_per_layer:
        #     print('using per_layer quant for weights')
        # else:
        #     print('using per_channel quant for weights')
            
        #bit renew 
        self.per_tensor = weight_per_layer
        self.register_buffer('old_bit', self.bit.clone().detach())
        # self.val_obs = MSEObserver(bit=bit, symmetric=~all_positive, ch_axis=-1 if in_channels is None else 1)
        self.register_buffer("val_min", torch.tensor(float("inf")).expand_as(self.bit))
        self.register_buffer("val_max", torch.tensor(float("-inf")).expand_as(self.bit))
        self.register_buffer("quant_min", torch.min(-(2 ** (self.bit.clone().detach()-1) -1), -torch.ones_like(self.bit)))
        self.register_buffer("quant_max", torch.max(2 ** (self.bit.clone().detach()-1) -1, torch.ones_like(self.bit)))
        # self.search_grid_num = 100
        
        self.search_grid_num = 50 if imagenet else 100
        self.eps = 0.00001
        self.weight_renew_switch = renew_switch
        self.bit_detect= bit_detect
    
    def lp_loss(self, pred, tgt, p=2.4):
        x = (pred - tgt).abs().pow(p)
        if self.per_tensor:
            return x.mean().reshape(1,1,1,1)
        else:
            return x.mean(1).reshape(-1,1,1,1)
    
    def loss_fx(self, x, new_min, new_max, bit_clamp):
        scale = self.scale_shift_cal(new_min, new_max)
        fea_q = self.quant_weight_func.apply(x.detach(), scale.detach(), bit_clamp.detach(), self.learn_weight_bit)
        score = self.lp_loss(fea_q, x)
        return score
    
    def search_minmax(self, x, bit_clamp):
        if self.per_tensor:
            x_min, x_max = torch.aminmax(x)
            x_min, x_max = x_min.reshape(1,1,1,1), x_max.reshape(1,1,1,1)
        else:
            y = _transform_to_ch_axis(x, 1)
            x_min, x_max = torch.aminmax(y, dim=1)
            x_min, x_max = x_min.reshape(-1,1,1,1), x_max.reshape(-1,1,1,1)
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        for i in range(1, self.search_grid_num+1):
            thres = xrange  * (i / self.search_grid_num)
            new_min = -thres
            new_max = thres
            score = self.loss_fx(x, new_min, new_max, bit_clamp)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max
    
    def qmax_renew(self, bit_round):
        self.quant_min = torch.min(-(2 ** (bit_round-1) -1), -torch.ones_like(bit_round))
        self.quant_max = torch.max(2 ** (bit_round-1) -1, torch.ones_like(bit_round))

    @torch.jit.export
    def scale_shift_cal(self, val_min, val_max):
        val_min_neg = torch.min(val_min, torch.zeros_like(val_min))
        val_max_pos = torch.max(val_max, torch.zeros_like(val_max))
        val_max_pos = torch.max(-val_min_neg, val_max_pos)
        scale = val_max_pos / ( (self.quant_max - self.quant_min).float() / 2 )
        scale = torch.where(scale > self.eps, scale, self.eps)
        return scale
    
    #scale renew 
    @torch.jit.export
    def scale_renew(self, bit_detach, fea_detach):
        bit_round = bit_detach.round()
        bit_mark = (self.old_bit!=bit_round)
        if any(bit_mark):   #start renew
            # print('renew act. scale and shift')
            with torch.no_grad():   
                #update scale and shifts
                self.qmax_renew(bit_round)
                best_min, best_max = self.search_minmax(fea_detach, bit_round)
                self.val_min = torch.min(self.val_min, best_min)
                self.val_max = torch.max(self.val_max, best_max)
                _scale = self.scale_shift_cal(self.val_min, self.val_max)
                if self.per_tensor:
                    self.alh.copy_(_scale)
                else:
                    self.alh[bit_mark] = _scale[bit_mark]
                self.old_bit.copy_(bit_round)  
        
        
    #scale renew 
    @torch.jit.export
    def scale_naive_renew(self, bit_detach, fea_detach):
        bit_round = bit_detach.round()
        bit_mark = (self.old_bit!=bit_round)
        if any(bit_mark):   #start renew
            # print('renew act. scale and shift')
            with torch.no_grad():   
                #update scale and shifts
                old_qmin, old_qmax = self.quant_min.clone(), self.quant_max.clone()
                self.qmax_renew(bit_round)
                new_qmin, new_qmax = self.quant_min, self.quant_max
                alh_scale =  (new_qmax-new_qmin) / (old_qmax-old_qmin)
                self.alh.copy_(self.alh * alh_scale)
                self.old_bit.copy_(bit_round)  
        
    def forward(self, weight, init_weight_bit):
        # quantization
        # weight_q = self.quant_weight_func.apply(weight, self.alh, self.bit, init_weight_bit)
        # bit is tooooo easy to fly so we do clamp
        bit = torch.clamp(self.bit.abs(), 1, self.bit_bound).detach() + self.bit - self.bit.detach()
        #scale renew
        if self.training and self.weight_renew_switch:
            self.scale_naive_renew(bit.clone().detach(), weight.detach())
        # bit detect
        if self.training and not self.weight_renew_switch and self.bit_detect:
            bit_detach = bit.clone().detach()
            bit_round = bit_detach.round()
            bit_mark = (self.old_bit!=bit_round)
            if any(bit_mark):   #start renew
                print('weight bit change found in this layer and batch')
            self.old_bit.copy_(bit_round)  
            
        alh = self.alh
        weight_q = self.quant_weight_func.apply(weight, alh, bit, self.learn_weight_bit)
        return weight_q




class u_quant_weight_conv1d(nn.Module):
    '''
        weight uniform quantization for conv1d.
    '''
    def __init__(self, out_channels, bit, alh_init=0.1,alh_std=0.1, learn_weight_bit=True, weight_per_layer=False):
        super().__init__()
        self.bit_bound = bit + 2
        self.bit = bit
        self.weight_per_layer = weight_per_layer
        # quantizer
        self.quant_weight_func = u_quant_w_func_alpha_conv1d_channelwise_div
        
        # parameters initialize
        _init_alh = alh_init
        # initialize the step size by normal distribution
        _alh = torch.Tensor(out_channels,1,1) if not weight_per_layer else torch.Tensor(1,1,1)
        self.alh = torch.nn.Parameter(_alh)
        torch.nn.init.normal_(self.alh, mean=alh_init, std=alh_std)
        self.alh = torch.nn.Parameter(self.alh.abs())
        # initialize the bit bit=4
        _init_bit = bit
        self.bit = torch.nn.Parameter(torch.Tensor(out_channels,1,1)) if not weight_per_layer else torch.nn.Parameter(torch.Tensor(1,1,1))
        torch.nn.init.constant_(self.bit,_init_bit)
        self.learn_weight_bit = learn_weight_bit
        if not learn_weight_bit:
            self.bit.requires_grad = False
            print('not learn conv weight bit')
        # if weight_per_layer:
        #     print('using per_layer quant for weights')
        # else:
        #     print('using per_channel quant for weights')
            
        #bit renew 
        self.per_tensor = weight_per_layer
        self.register_buffer('old_bit', self.bit.clone().detach())
        # self.val_obs = MSEObserver(bit=bit, symmetric=~all_positive, ch_axis=-1 if in_channels is None else 1)
        self.register_buffer("val_min", torch.tensor(float("inf")).expand_as(self.bit))
        self.register_buffer("val_max", torch.tensor(float("-inf")).expand_as(self.bit))
        self.register_buffer("quant_min", torch.min(-(2 ** (self.bit.clone().detach()-1) -1), -torch.ones_like(self.bit)))
        self.register_buffer("quant_max", torch.max(2 ** (self.bit.clone().detach()-1) -1, torch.ones_like(self.bit)))
        # self.search_grid_num = 100
        self.search_grid_num = 50 #imagenet
        self.eps = 0.00001
        self.weight_renew_switch = renew_switch
        self.bit_detect= bit_detect
    
    def lp_loss(self, pred, tgt, p=2.4):
        x = (pred - tgt).abs().pow(p)
        if self.per_tensor:
            return x.mean().reshape(1,1,1)
        else:
            return x.mean(1).reshape(-1,1,1)
    
    def loss_fx(self, x, new_min, new_max, bit_clamp):
        scale = self.scale_shift_cal(new_min, new_max)
        fea_q = self.quant_weight_func.apply(x.detach(), scale.detach(), bit_clamp.detach(), self.learn_weight_bit)
        score = self.lp_loss(fea_q, x)
        return score
    
    def search_minmax(self, x, bit_clamp):
        if self.per_tensor:
            x_min, x_max = torch.aminmax(x)
            x_min, x_max = x_min.reshape(1,1,1), x_max.reshape(1,1,1)
        else:
            y = _transform_to_ch_axis(x, 1)
            x_min, x_max = torch.aminmax(y, dim=1)
            x_min, x_max = x_min.reshape(-1,1,1), x_max.reshape(-1,1,1)
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        for i in range(1, self.search_grid_num+1):
            thres = xrange  * (i / self.search_grid_num)
            new_min = -thres
            new_max = thres
            score = self.loss_fx(x, new_min, new_max, bit_clamp)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max
    
    def qmax_renew(self, bit_round):
        self.quant_min = torch.min(-(2 ** (bit_round-1) -1), -torch.ones_like(bit_round))
        self.quant_max = torch.max(2 ** (bit_round-1) -1, torch.ones_like(bit_round))

    @torch.jit.export
    def scale_shift_cal(self, val_min, val_max):
        val_min_neg = torch.min(val_min, torch.zeros_like(val_min))
        val_max_pos = torch.max(val_max, torch.zeros_like(val_max))
        val_max_pos = torch.max(-val_min_neg, val_max_pos)
        scale = val_max_pos / ( (self.quant_max - self.quant_min).float() / 2 )
        scale = torch.where(scale > self.eps, scale, self.eps)
        return scale
    
    #scale renew 
    @torch.jit.export
    def scale_renew(self, bit_detach, fea_detach):
        bit_round = bit_detach.round()
        bit_mark = (self.old_bit!=bit_round)
        if any(bit_mark):   #start renew
            # print('renew act. scale and shift')
            with torch.no_grad():   
                #update scale and shifts
                self.qmax_renew(bit_round)
                best_min, best_max = self.search_minmax(fea_detach, bit_round)
                self.val_min = torch.min(self.val_min, best_min)
                self.val_max = torch.max(self.val_max, best_max)
                _scale = self.scale_shift_cal(self.val_min, self.val_max)
                if self.per_tensor:
                    self.alh.copy_(_scale)
                else:
                    self.alh[bit_mark] = _scale[bit_mark]
                self.old_bit.copy_(bit_round)  
        
        
    #scale renew 
    @torch.jit.export
    def scale_naive_renew(self, bit_detach, fea_detach):
        bit_round = bit_detach.round()
        bit_mark = (self.old_bit!=bit_round)
        if any(bit_mark):   #start renew
            # print('renew act. scale and shift')
            with torch.no_grad():   
                #update scale and shifts
                old_qmin, old_qmax = self.quant_min.clone(), self.quant_max.clone()
                self.qmax_renew(bit_round)
                new_qmin, new_qmax = self.quant_min, self.quant_max
                alh_scale =  (new_qmax-new_qmin) / (old_qmax-old_qmin)
                self.alh.copy_(self.alh * alh_scale)
                self.old_bit.copy_(bit_round)  
        
    def forward(self, weight, init_weight_bit):
        # quantization
        # weight_q = self.quant_weight_func.apply(weight, self.alh, self.bit, init_weight_bit)
        # bit is tooooo easy to fly so we do clamp
        bit = torch.clamp(self.bit.abs(), 1, self.bit_bound).detach() + self.bit - self.bit.detach()
        #scale renew
        if self.training and self.weight_renew_switch:
            self.scale_naive_renew(bit.clone().detach(), weight.detach())
        # bit detect
        if self.training and not self.weight_renew_switch and self.bit_detect:
            bit_detach = bit.clone().detach()
            bit_round = bit_detach.round()
            bit_mark = (self.old_bit!=bit_round)
            if any(bit_mark):   #start renew
                print('weight bit change found in this layer and batch')
            self.old_bit.copy_(bit_round)  
            
        alh = self.alh
        weight_q = self.quant_weight_func.apply(weight, alh, bit, self.learn_weight_bit)
        return weight_q





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
            gama = self.gama
            fea_q = self.quant_fea_func.apply(fea, gama, bit, self.learn_fea_bit)
        return fea_q


#################################################################################
#
#                   modules
#
#################################################################################
class qLinear(nn.Linear):
    ''' 
    Quantized linear layers.
    '''
    def __init__(self, in_features, out_features, bit=weight_bit, bias=True, fea_all_positive=False,
                para_dict={'alh_init':0.01,'alh_std':0.02,'gama_init':0.1,'gama_std':0.2},
                quant_fea=True,
                **kwargs):
        super(qLinear, self).__init__(in_features, out_features, bias)
        self.bit = bit
        # if the value is all positive then we use the unsign quantization
        # if(fea_all_positive):
        #     bit_fea = fea_bit + 1
        # else:
        bit_fea = fea_bit
        self.fea_all_positive = fea_all_positive
        self.fea_bit = bit_fea
        self.weight_bit = bit
        alh_init = para_dict['alh_init']
        gama_init = para_dict['gama_init']
        alh_std = para_dict['alh_std']
        gama_std = para_dict['gama_std']
        # weight quantization module
        self.weight_quant = u_quant_weight_linear(out_features, self.weight_bit, alh_init=alh_init, alh_std=alh_std)
        # self.weight_quant = nn.Identity()
        
        # features quantization module
        self.fea_quant = u_quant_fea(None, bit_fea, gama_init=gama_init,gama_std=gama_std)
        if(quant_fea==False):
            # Do not quantize the feature when the value is 0 or 1
            self.fea_quant = nn.Identity()
        # glorot(self.weight)
        self.register_buffer('init_state', torch.zeros(1))
    
    def forward(self, x, q_info=None):
        #lsq initial
        if self.training and self.init_state == 0:
            torch.nn.init.constant_(self.fea_quant.gama, 2 * x.abs().mean() / math.sqrt(2**(self.fea_bit-1)))
            torch.nn.init.constant_(self.weight_quant.alh, 2 * self.weight.abs().mean() / math.sqrt(2**(self.weight_bit-1)))
            self.init_state.fill_(1)
            
        # weight quantization
        weight_q = self.weight_quant(self.weight, self.weight_bit)
        # weight_q  = self.weight
        fea_q = self.fea_quant(x, self.fea_bit)
        # fea_q = x
        
        if q_info is not None:
            q_info.append({
                        'fea_bit': self.fea_quant.bit.clone().squeeze(),
                        'fea_scale': self.fea_quant.gama.clone().squeeze(),
                        'weight_bit': self.weight_quant.bit.clone().squeeze(),
                        'weight_scale': self.weight_quant.alh.clone().squeeze(),
                        'layer_type':'linear',
                        'fea_numel': fea_q.numel(),
                        'weight_numel':weight_q.numel(),
                        })
            
        # dect_errTensor(fea_q,name='linear_feature')
        # dect_errTensor(weight_q,name='linear_weight', layer_count=False)
        return F.linear(fea_q, weight_q, self.bias), q_info

    
    
    

class qConv2d(nn.Conv2d):
    def __init__(
                self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros', device=None, dtype=None,
                
                bit=weight_bit, fea_all_positive=False,
                para_dict={'alh_init':0.01,'alh_std':0.02,'gama_init':0.1,'gama_std':0.2},
                quant_fea=True,
                **kwargs
                ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.bit = bit
        # if the value is all positive then we use the unsign quantization
        if(fea_all_positive):
            bit_fea = fea_bit + 1
        else:
            bit_fea = fea_bit
        self.fea_all_positive = fea_all_positive
        self.fea_bit = bit_fea
        self.weight_bit = bit
        alh_init = para_dict['alh_init']
        gama_init = para_dict['gama_init']
        alh_std = para_dict['alh_std']
        gama_std = para_dict['gama_std']
        # weight quantization module
        self.weight_quant = u_quant_weight_conv(out_channels, bit, alh_init=alh_init,alh_std=alh_std)
        # self.weight_quant = nn.Identity()
        # features quantization module
        self.fea_quant = u_quant_fea(None, bit_fea, gama_init=gama_init,gama_std=gama_std)
        if(quant_fea==False):
            # Do not quantize the feature when the value is 0 or 1
            self.fea_quant = nn.Identity()
        # glorot(self.weight)
        self.register_buffer('init_state', torch.zeros(1))
    
    def forward(self, x, q_info=None):
        #lsq initial
        if self.training and self.init_state == 0:
            torch.nn.init.constant_(self.fea_quant.gama, 2 * x.abs().mean() / math.sqrt(2**(self.fea_bit-1)))
            torch.nn.init.constant_(self.weight_quant.alh, 2 * self.weight.abs().mean() / math.sqrt(2**(self.weight_bit-1)))
            self.init_state.fill_(1)
        # weight quantization
        weight_q = self.weight_quant(self.weight, self.weight_bit)
        # weight_q  = self.weight
        fea_q = self.fea_quant(x, self.fea_bit)
        # fea_q = x
        if q_info is not None:
            q_info.append({
                        'fea_bit': self.fea_quant.bit.clone().squeeze(),
                        'fea_scale': self.fea_quant.gama.clone().squeeze(),
                        'weight_bit': self.weight_quant.bit.clone().squeeze(),
                        'weight_scale': self.weight_quant.alh.clone().squeeze(),
                        'layer_type':'conv',
                        'fea_numel': fea_q.numel(),
                        'weight_numel':weight_q.numel(),
                        })
        
        # dect_errTensor(fea_q,name='conv_feature')
        # dect_errTensor(weight_q,name='conv_weight', layer_count=False)
        return  self._conv_forward(fea_q, weight_q, self.bias), q_info