import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import copy
import numpy as np
from IPython import embed
from models.quantization.quantization_modules import u_quant_weight_conv, u_quant_weight_linear
from models.quantization.quantization_funcs import spike_quant_fea_func_gama_layerwise_div, spike_quant_fea_func_gama_channelwise_div
from models.quantization.q_search import MSEObserver

strict_spike_model = False

bit_detect = False

ace_cal = True

using_shift = False
renew_switch = True


imagenet = True
T_max = None
# T_max = 2 if imagenet else 3





def _transform_to_ch_axis(x, ch_axis):
    new_axis_list = [i for i in range(x.dim())]
    new_axis_list[ch_axis] = 0
    new_axis_list[0] = ch_axis
    x_channel = x.permute(new_axis_list)
    y = torch.flatten(x_channel, start_dim=1)
    return y

class SpikeModule(nn.Module):

    def __init__(self):
        super().__init__()
        self._spiking = False

    def set_spike_state(self, use_spike=True):
        self._spiking = use_spike

    def forward(self, x):
        # shape correction
        if self._spiking is not True and len(x.shape) == 5:
            x = x.mean([0])
        return x



def spike_activation(x, binary=False, temp=1.0):
    

    if binary:
        out_s = torch.gt(x, 0.5)
        out_bp = torch.clamp(x, 0, 1)
        #out_bp = (torch.tanh(temp * (out_bp-0.5)) + np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
        return (out_s.float() - out_bp).detach() + out_bp
    else:
        out_s = torch.sign(x)
        out_s[torch.abs(x)<0.5] = torch.tensor(0.)
        out_bp = torch.clamp(x, -1, 1)
        #out_bp[out_bp>0.] = (torch.tanh(temp * (out_bp[out_bp>0.]-0.5)) + np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
        #out_bp[out_bp<=0.] = (torch.tanh(temp * (out_bp[out_bp<=0.]+0.5)) - np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
        return (out_s.float() - out_bp).detach() + out_bp


def MPR(s,thresh):

    s[s>1.] = s[s>1.]**(1.0/3)
    s[s<0.] = -(-(s[s<0.]-1.))**(1.0/3)+1.
    s[(0.<s)&(s<1.)] = 0.5*torch.tanh(3.*(s[(0.<s)&(s<1.)]-thresh))/np.tanh(3.*(thresh))+0.5
    
    return s


def gradient_scale(x, scale):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    return y


def mem_update(x_in, mem, V_th, decay, fire_ratio,grad_scale=1., temp=1.0):
    mem = mem * decay + x_in
    #if mem.shape[1]==256:
    #    embed()
    #V_th = gradient_scale(V_th, grad_scale)
    #mem2 = MPR(mem, 0.5)
    spike = spike_activation(mem / V_th, temp=temp)
    mem = mem * (1 - torch.abs(spike))
    #mem = mem - spike
    spike = spike * fire_ratio
    return mem, spike


class LIFAct(SpikeModule):
    """ Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self, step):
        super(LIFAct, self).__init__()
        self.step = step
        #self.V_th = nn.Parameter(torch.tensor(1.))
        self.V_th = 1.0
        # self.tau = nn.Parameter(torch.tensor(-1.1))
        self.temp = 3.0
        #self.temp = nn.Parameter(torch.tensor(1.))
        self.grad_scale = 0.1
        
        #self.fire_ratio = nn.Parameter(torch.ones(1,512,1,1), requires_grad=True)
        self.fire_ratio = nn.Parameter(torch.tensor(1.))
        # self.fire_ratio = 1


    def forward(self, x):
        if self._spiking is not True:
            return F.relu(x)
        if self.grad_scale is None:
            self.grad_scale = 1 / math.sqrt(x[0].numel()*self.step)
        u = torch.zeros_like(x[0])
        out = []
        T, B, C, H, W = x.shape
        for i in range(self.step):
        
            u, out_i = mem_update(x_in=x[i], mem=u, V_th=self.V_th,fire_ratio=self.fire_ratio,
                                  grad_scale=self.grad_scale, decay=0.25, temp=self.temp)
            out += [out_i]
        out = torch.stack(out)
        return out


class SpikeConv(SpikeModule):


    def __init__(self, conv, step=2):
        super(SpikeConv, self).__init__()
        self.conv = conv
        self.step = step

    def forward(self, x):
        if self._spiking is not True:
            return self.conv(x)
        out = []
        for i in range(self.step):
            out += [self.conv(x[i])]
        out = torch.stack(out)
        return out


class SpikePool(SpikeModule):

    def __init__(self, pool, step=2):
        super().__init__()
        self.pool = pool
        self.step = step

    def forward(self, x):
        if self._spiking is not True:
            return self.pool(x)
        T, B, C, H, W = x.shape
        out = x.reshape(-1, C, H, W)
        out = self.pool(out)
        B_o, C_o, H_o, W_o = out.shape
        out = out.view(T, B, C_o, H_o, W_o).contiguous()
        return out

class myBatchNorm3d(SpikeModule):
    def __init__(self, BN: nn.BatchNorm2d, step=2):
        super().__init__()
        self.bn = nn.BatchNorm3d(BN.num_features)
        self.step = step
    def forward(self, x):
        if self._spiking is not True:
            return BN(x)
        out = x.permute(1, 2, 0, 3, 4)
        out = self.bn(out)
        out = out.permute(2, 0, 1, 3, 4).contiguous()
        return out


class tdBatchNorm2d(nn.BatchNorm2d, SpikeModule):
    
    def __init__(self, bn: nn.BatchNorm2d, alpha: float):
        super(tdBatchNorm2d, self).__init__(bn.num_features, bn.eps, bn.momentum, bn.affine, bn.track_running_stats)
        self.alpha = alpha
        self.V_th = 0.5
        # self.weight.data = bn.weight.data
        # self.bias.data = bn.bias.data
        # self.running_mean.data = bn.running_mean.data
        # self.running_var.data = bn.running_var.data

    def forward(self, input):
        if self._spiking is not True:
            # compulsory eval mode for normal bn
            self.training = False
            return super().forward(input)

        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 1, 3, 4])
            # use biased var in train
            var = input.var([0, 1, 3, 4], unbiased=False)
            n = input.numel() / input.size(2)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        channel_dim = input.shape[2]
        input = self.alpha * self.V_th * (input - mean.reshape(1, 1, channel_dim, 1, 1)) / \
                (torch.sqrt(var.reshape(1, 1, channel_dim, 1, 1) + self.eps))
        if self.affine:
            input = input * self.weight.reshape(1, 1, channel_dim, 1, 1) + self.bias.reshape(1, 1, channel_dim, 1, 1)

        return input




weight_bit = 4
def ste_round(x):
    return torch.round(x) - x.detach() + x


def ste_clamp(x, min, max):
    return torch.clamp(x, min, max).detach() - x.detach() + x



class AlphaInit(nn.Parameter):
    def __init__(self, tensor,  requires_grad=True):
        super(AlphaInit, self).__new__(nn.Parameter, data=tensor, requires_grad=requires_grad)
        self.initialized = False

    def _initialize(self, init_tensor):
        assert not self.initialized, 'already initialized.'
        self.data.copy_(init_tensor)
        self.initialized = True

    def initialize_wrapper(self, tensor, num_bits, symmetric, init_method='default'):
        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** (num_bits) - 1
        if Qp == 0:
            Qp = 1.0
        if init_method == 'default':
            init_val = 2 * tensor.abs().mean() / math.sqrt(Qp) if symmetric \
                else 4 * tensor.abs().mean() / math.sqrt(Qp)
        elif init_method == 'uniform':
            init_val = 1./(2*Qp+1) if symmetric else 1./Qp

        self._initialize(init_val)


# old version
class spike_quant_fea(nn.Module):
    '''
        spike feature quantization.
    '''
    def __init__(self, in_channels, bit, gama_init=0.001,gama_std=0.001, quant_fea=True, learn_spike_bit=True, all_positive=True):
        super().__init__()
        self.bit_bound = bit + 2
        self.bit = bit
        self.quant_fea = quant_fea
        self.deg_inv_sqrt = 1.0
        self.all_positive = all_positive
        self.in_channels = in_channels
        print("all_positive:", all_positive)
       
      # quantizer
        self.quant_fea_func = spike_quant_fea_func_gama_layerwise_div if in_channels is None else spike_quant_fea_func_gama_channelwise_div
        # parameters initialization
        # initialize step size
        self.gama = torch.nn.Parameter(torch.Tensor(1).reshape(-1)) if in_channels is None else torch.nn.Parameter(torch.Tensor(1,in_channels,1,1))
        torch.nn.init.normal_(self.gama, mean=gama_init, std=gama_std)
        self.gama = torch.nn.Parameter(self.gama.abs())
        # initialize the bit, bit=4
        self.bit = torch.nn.Parameter(torch.Tensor(1).reshape(-1)) if in_channels is None else torch.nn.Parameter(torch.Tensor(1,in_channels,1,1))
        _init_bit = bit
        torch.nn.init.constant_(self.bit,_init_bit)
        self.learn_spike_bit = learn_spike_bit
        if not learn_spike_bit:
            self.bit.requires_grad = False
            print('not learn spike bit')
        
        #special shift para only designed for the binary case 
        self.bin_shift = torch.nn.Parameter(torch.Tensor(1).reshape(-1)) if in_channels is None else torch.nn.Parameter(torch.Tensor(1,in_channels,1,1))
        torch.nn.init.constant_(self.bin_shift, 0.)
        if not using_shift:
            print('not learn spike shift')
            self.bin_shift.requires_grad = False
        self.using_shift = using_shift
            
            
        
        #bit-triggered renew mechanism
        self.per_tensor = in_channels is None
        self.register_buffer('old_bit', self.bit.clone().detach())
        # self.val_obs = MSEObserver(bit=bit, symmetric=~all_positive, ch_axis=-1 if in_channels is None else 1)
        self.register_buffer("val_min", torch.tensor(float("inf")).expand_as(self.bit))
        self.register_buffer("val_max", torch.tensor(float("-inf")).expand_as(self.bit))
        self.register_buffer("quant_min", self.bit.clone().detach() * 0. if self.all_positive else torch.min(-(2 ** (self.bit.clone().detach()-1) -1), -torch.ones_like(self.bit)))
        self.register_buffer("quant_max", (2 ** self.bit.clone().detach() - 1) if self.all_positive else torch.max(2 ** (self.bit.clone().detach()-1) -1, torch.ones_like(self.bit)))
        # self.search_grid_num = 100
        self.search_grid_num = 50 if imagenet else 100
        self.eps = 0.00001
        self.act_renew_switch = renew_switch
        self.bit_detect=bit_detect
        
    def lp_loss(self, pred, tgt, p=2.4):
        x = (pred - tgt).abs().pow(p)
        if self.per_tensor:
            return x.mean().reshape(-1)
        else:
            return x.mean((0,2,3)).reshape(1,-1,1,1)
    
    def loss_fx(self, x, new_min, new_max, bit_clamp):
        scale, shift = self.scale_shift_cal(new_min, new_max)
        fea_q = self.quant_fea_func.apply(x.detach(), scale.detach(), bit_clamp.detach(), shift.detach(), self.learn_spike_bit, self.all_positive)
        score = self.lp_loss(fea_q, x)
        return score
    
    def search_minmax(self, x, bit_clamp):
        if self.per_tensor:
            x_min, x_max = torch.aminmax(x)
            x_min, x_max = x_min.reshape(-1), x_max.reshape(-1)
        else:
            y = _transform_to_ch_axis(x, 1)
            x_min, x_max = torch.aminmax(y, dim=1)
            x_min, x_max = x_min.reshape(1,-1,1,1), x_max.reshape(1,-1,1,1)
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        for i in range(1, self.search_grid_num+1):
            thres = xrange  * (i / self.search_grid_num)
            new_min = torch.zeros_like(x_min) if self.all_positive else -thres
            new_max = thres
            score = self.loss_fx(x, new_min, new_max, bit_clamp)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max
    
    def qmax_renew(self, bit_round):
        if self.all_positive:
            self.quant_min = bit_round * 0.
            self.quant_max = 2 ** bit_round - 1
        else:
            self.quant_min = torch.min(-(2 ** (bit_round-1) -1), -torch.ones_like(bit_round))
            self.quant_max = torch.max(2 ** (bit_round-1) -1, torch.ones_like(bit_round))
    
    @torch.jit.export
    def scale_shift_cal(self, val_min, val_max):
        val_min_neg = torch.min(val_min, torch.zeros_like(val_min))
        val_max_pos = torch.max(val_max, torch.zeros_like(val_max))
        if self.all_positive:
            scale = (val_max_pos - val_min_neg) / (self.quant_max - self.quant_min).float()
            scale = torch.where(scale > self.eps, scale, self.eps)
            shift = torch.round(val_min_neg / scale) - self.quant_min
            # shift = torch.clamp(shift, self.quant_min, self.quant_max)
        else:
            val_max_pos = torch.max(-val_min_neg, val_max_pos)
            scale = val_max_pos / ( (self.quant_max - self.quant_min).float() / 2 )
            scale = torch.where(scale > self.eps, scale, self.eps)
            shift = torch.zeros_like(val_min)
        return scale, shift
    
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
                _scale, _shift = self.scale_shift_cal(self.val_min, self.val_max)
                if self.per_tensor:
                    self.gama.copy_(_scale)
                    if self.using_shift:
                        self.bin_shift.copy_(_shift)
                else:
                    self.gama[bit_mark] = _scale[bit_mark]
                    if self.using_shift:
                        self.bin_shift[bit_mark] = _shift[bit_mark]
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
                gama_scale =  (new_qmax-new_qmin) / (old_qmax-old_qmin)
                self.gama.copy_(self.gama * gama_scale)
                self.old_bit.copy_(bit_round)  
        
    def forward(self, fea):
        if(not self.quant_fea):
            fea_q = fea
        else:
            # bit is tooooo easy to fly so we do clamp
            # print(self.bit.grad)
            bit = torch.clamp(self.bit.abs(), 1, self.bit_bound).detach() + self.bit - self.bit.detach()
            #scale renew due to bit change
            if self.training and self.act_renew_switch:
                self.scale_renew(bit.clone().detach(), fea.detach())
                
            # bit detect
            if self.training and not self.act_renew_switch and self.bit_detect:
                bit_detach = bit.clone().detach()
                bit_round = bit_detach.round()
                bit_mark = (self.old_bit!=bit_round)
                if any(bit_mark):   #start renew
                    print('act bit change found in this layer and batch')
                self.old_bit.copy_(bit_round)  
                
            gama = self.gama
            ## bit shift
            # bin_shift = torch.sigmoid(self.bin_shift)
            fea_q = self.quant_fea_func.apply(fea, gama, bit, self.bin_shift, self.learn_spike_bit, self.all_positive) ## bin shift added 
            # fea_q = self.quant_fea_func.apply(fea, gama, bit, self.learn_spike_bit, self.all_positive) ## no bin shift added 
        return fea_q

    


class SpikeConv2d_q(nn.Conv2d):
    def __init__(
                self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros', device=None, dtype=None,
                
                init_weight_bit=weight_bit, step = 2, init_spike_bit=2, max_step=4, fea_all_positive=False,
                para_dict={'alh_init':0.01,'alh_std':0.02,'gama_init':0.1,'gama_std':0.2},
                **kwargs
                ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        if fea_all_positive:
            self.spike_bit = init_spike_bit
            print('fea_all_positive')
        else:
            self.spike_bit = init_spike_bit
        self.fea_all_positive = fea_all_positive
        self.weight_bit = init_weight_bit
        
        if 'learn_spike_bit' in kwargs and kwargs['learn_spike_bit']:
            self.learn_spike_bit = True
        else:
            self.learn_spike_bit = False
            print('not learn spike bit')
        
        if 'learn_weight_bit' in kwargs and kwargs['learn_weight_bit']:
            self.learn_weight_bit = True
            print('learn weight bit')
        else:
            self.learn_weight_bit = False
            print('not learn weight bit')
        if 'learn_time_step' in kwargs and kwargs['learn_time_step']:
            self.learn_time_step = True
        else:
            self.learn_time_step = False
            print('not learn spike step')
            
        alh_init = para_dict['alh_init']
        gama_init = para_dict['gama_init']
        alh_std = para_dict['alh_std']
        gama_std = para_dict['gama_std']
        # weight quantization module
        self.weight_per_layer = kwargs['weight_per_layer']
        self.weight_quant = u_quant_weight_conv(out_channels, self.weight_bit, alh_init=alh_init,alh_std=alh_std, learn_weight_bit = self.learn_weight_bit, weight_per_layer=kwargs['weight_per_layer'])
        # self.weight_quant = nn.Identity()
        
        
        # spikes quantization module
            #layer-wise
        self.T = torch.nn.Parameter(torch.Tensor(1))
        _init_step = step
        torch.nn.init.constant_(self.T, _init_step)
        if not self.learn_time_step:
            self.T.requires_grad = False
            self.T_max = step
        else:
            self.T_max = step+1 if T_max is None else T_max
        # self.act_clip_val = nn.ParameterList([AlphaInit(torch.tensor(1.0), requires_grad =False) for i in range(self.T)])
        
        if 'spike_per_layer' in kwargs and kwargs['spike_per_layer']:
            self.spike_quant_channel = None
            print('spike per layer')
        else:
            self.spike_quant_channel = in_channels
            print('spike per channel')
        self.spikes_quant = nn.ModuleList([
            spike_quant_fea(self.spike_quant_channel, self.spike_bit, gama_init=gama_init,gama_std=gama_std, learn_spike_bit=self.learn_spike_bit, all_positive=self.fea_all_positive) for i in range(self.T_max)
        ])
        
        
        self.register_buffer('init_training_state', torch.zeros(1))
        self.decay = kwargs['decay'] if 'decay' in kwargs.keys() else 1.
        if 'decay' in kwargs.keys():
            print('decay:', self.decay)
        if 'learn_decay' in kwargs and kwargs['learn_decay']:
            print('learn decay')
            self.decay = torch.nn.Parameter(torch.tensor(self.decay))
    
        
        
        
    def forward(self, x, q_info=None):
        #lsq initial
        if self.training and self.init_training_state == 0:
            print('initial scales for the first forward in training')
            weight_init_bit = self.weight_bit if self.weight_bit > 1 else 2
            spike_init_bit = self.spike_bit if self.spike_bit > 1 else 2
            
            # init wegith alpha
            if self.weight_per_layer:
                torch.nn.init.constant_(self.weight_quant.alh, 2 * self.weight.detach().abs().mean() / math.sqrt(2**(weight_init_bit-1)-1))
            else:
                self.weight_quant.alh.data.copy_(2 * self.weight.detach().abs().mean(dim=list(range(1, self.weight.dim())), keepdim=True) / math.sqrt(2**(weight_init_bit-1)-1))
            
            # init act gamma
            if self.spike_quant_channel is None: # per layer
                if self.fea_all_positive:
                    for i in range(len(self.spikes_quant)):
                        torch.nn.init.constant_(self.spikes_quant[i].gama, 2 * x.detach().abs().mean() / math.sqrt(2**(self.spike_bit)-1))
                else:
                    for i in range(len(self.spikes_quant)):
                        torch.nn.init.constant_(self.spikes_quant[i].gama, 2 * x.detach().abs().mean() / math.sqrt(2**(spike_init_bit-1)-1))
            else:
                if self.fea_all_positive:
                    for i in range(len(self.spikes_quant)):
                        self.spikes_quant[i].gama.data.copy_(2 * x.detach().abs().mean(dim=(0,2,3), keepdim=True) / math.sqrt(2**(self.spike_bit)-1))
                else:
                    for i in range(len(self.spikes_quant)):
                        self.spikes_quant[i].gama.data.copy_(2 * x.detach().abs().mean(dim=(0,2,3), keepdim=True) / math.sqrt(2**(spike_init_bit-1)-1))
            self.init_training_state.fill_(1)
            
        # weight quantization
        weight_q = self.weight_quant(self.weight, self.weight_bit)

        # spike quantization fea
        mem = x.clone()
        output = []
        output.append(self.spikes_quant[0](mem))
        valid_T = torch.clamp(self.T.detach().round().int(), 1, self.T_max)
        if valid_T>1:
            mem_res = mem.detach() - output[0].detach()
            for i in range(1, valid_T):
                mem = mem_res * self.decay + x 
                output.append(self.spikes_quant[i](mem))
                mem_res = mem.detach() - output[i].detach()
                
        T_grad = valid_T + self.T - self.T.detach()
        
        
        spike_out = torch.stack(output, dim=0).sum(0) / T_grad
        # spike_out = self.spikes_quant[0](x)
        
        #straght through the learnable params up
        if q_info is not None:
            q_info.append({
                        'spike_bit': torch.stack([ste_clamp(self.spikes_quant[i].bit.clone().abs(), min=1., max=self.spikes_quant[i].bit_bound) for i in range(valid_T)]).squeeze().reshape(-1),
                        'spike_scale': torch.stack([self.spikes_quant[i].gama.clone() for i in range(valid_T)]).squeeze().reshape(-1),
                        'weight_bit': ste_clamp(self.weight_quant.bit.clone().abs().squeeze().reshape(-1), 1.,self.weight_quant.bit_bound),
                        'weight_scale': self.weight_quant.alh.clone().abs().squeeze().reshape(-1),
                        'T': T_grad.clone().reshape(-1),
                        'T_max':self.T_max,
                        'layer_type':'conv',
                        'fea_numel': spike_out.numel(),
                        'weight_numel':weight_q.numel(),
                        'learn_spike_bit':self.learn_spike_bit,
                        'learn_weight_bit':self.learn_weight_bit,
                        })
        #add nice and ns-nice cal
        if not self.training and ace_cal:    
            B,C,H,W = x.shape
            spike_bit = q_info[-1]['spike_bit'].detach().round()
            weight_bit = q_info[-1]['weight_bit'].detach().round()
            bit_budget = spike_bit * weight_bit
            bit_budget_mean = spike_bit.max() * weight_bit.max()
            # print(spike_bit, weight_bit)
            kernel_size = self.kernel_size if not isinstance(self.kernel_size, tuple) else self.kernel_size[0]
            stride = self.stride if not isinstance(self.stride, tuple) else self.stride[0]
            padding = self.padding if not isinstance(self.padding, tuple) else self.padding[0]
            ops = math.ceil((H + padding * 2 - (kernel_size-1)) / stride) ** 2 * kernel_size ** 2 * C * self.out_channels
            sops = ops * torch.min((valid_T*spike_bit.mean()), weight_bit.mean())
            s_ace = (bit_budget * ops).sum()
            s_ace_mean = (bit_budget_mean * ops)
           
            
            f_num = 0
            f_act_state_num = 0
            for i,out in enumerate(output):
                f_num += torch.count_nonzero(out)
                f_act_state_num += torch.count_nonzero(out) * ste_clamp(self.spikes_quant[i].bit.clone().abs(), min=1., max=self.spikes_quant[i].bit_bound).squeeze()
            fr = f_num/(x.numel() * valid_T)
            
        
            
            
            ns_ace = s_ace*fr
            ace = {
                's_ace':s_ace,
                'sops':sops,
                'ns_ace':ns_ace,
                'fr':fr,
                'ops':ops,
                's_ace_mean':s_ace_mean,
                'f_num':f_num,
                'tot_num':x.numel() * valid_T,
                'f_act_state_num':f_act_state_num,
                'neuron_num': spike_out.numel(),
            }
            q_info[-1].update(ace)   
            # print('T:', valid_T)
                     
        return self._conv_forward(spike_out, weight_q, self.bias), q_info

    
    @staticmethod
    def inherit_from(ori_module, step=2, **kwargs):
        assert isinstance(ori_module, torch.nn.Conv2d)
        new_module = SpikeConv2d_q(
            in_channels = ori_module.in_channels,
            out_channels = ori_module.out_channels,
            kernel_size = ori_module.kernel_size,
            stride = ori_module.stride,
            padding = ori_module.padding,
            dilation = ori_module.dilation,
            groups = ori_module.groups,
            bias = ori_module.bias is not None,
            # bit = getattr(ori_module, 'weight_bit', weight_bit),
            step = step,
            fea_all_positive =  False if 'spike_all_positive' not in kwargs else kwargs['spike_all_positive'],
            **kwargs
        )
        new_module.weight.data.copy_(ori_module.weight.detach())
        if ori_module.bias is not None:
            new_module.bias.data.copy_(ori_module.bias.detach())
            
        return new_module
    

class SpikeLinear_q(nn.Linear):
    def __init__(
                self, in_features, out_features, bias=True, fea_all_positive=False,
                
                init_weight_bit=weight_bit, step = 2, init_spike_bit=2, max_step=4, 
                para_dict={'alh_init':0.01,'alh_std':0.02,'gama_init':0.1,'gama_std':0.2},
                **kwargs
                ) -> None:
        super().__init__(in_features, out_features, bias)
        if fea_all_positive:
            self.spike_bit = init_spike_bit
            print('fea_all_positive')
        else:
            self.spike_bit = init_spike_bit
            
        self.fea_all_positive = fea_all_positive
        self.weight_bit = init_weight_bit
        
        if 'learn_spike_bit' in kwargs and kwargs['learn_spike_bit']:
            self.learn_spike_bit = True
        else:
            self.learn_spike_bit = False
            print('not learn spike bit')
        
        if 'learn_weight_bit' in kwargs and kwargs['learn_weight_bit']:
            self.learn_weight_bit = True
        else:
            self.learn_weight_bit = False
            print('not learn weight bit')
        if 'learn_time_step' in kwargs and kwargs['learn_time_step']:
            self.learn_time_step = True
        else:
            self.learn_time_step = False
            print('not learn spike step')
            
        alh_init = para_dict['alh_init']
        gama_init = para_dict['gama_init']
        alh_std = para_dict['alh_std']
        gama_std = para_dict['gama_std']    
        # weight quantization module             
        self.weight_quant = u_quant_weight_linear(out_features, self.weight_bit, alh_init=alh_init, alh_std=alh_std, learn_weight_bit=self.learn_weight_bit, weight_per_layer=kwargs['weight_per_layer'])
        self.weight_per_layer=kwargs['weight_per_layer']
        # self.weight_quant = nn.Identity()
        
        
        # spikes quantization module
            #layer-wise
        # self.fea_quant = spike_quant_fea(bit_fea, gama_init=gama_init,gama_std=gama_std) 
        self.T = torch.nn.Parameter(torch.Tensor(1))
        _init_step = step
        torch.nn.init.constant_(self.T, _init_step)
        if not self.learn_time_step:
            self.T.requires_grad = False
            self.T_max = step
        else:
            self.T_max = step+1 if T_max is None else T_max
            
        # self.act_clip_val = nn.ParameterList([AlphaInit(torch.tensor(1.0), requires_grad =False) for i in range(self.T)])
        
            
        self.spikes_quant = nn.ModuleList([
            spike_quant_fea(None, self.spike_bit, gama_init=gama_init,gama_std=gama_std, learn_spike_bit=self.learn_spike_bit, all_positive=self.fea_all_positive) for i in range(self.T_max)
        ])
        
        
        
        self.register_buffer('init_training_state', torch.zeros(1))
        self.decay = kwargs['decay'] if 'decay' in kwargs.keys() else 1.
        if 'decay' in kwargs.keys():
            print('decay:', self.decay)
        if 'learn_decay' in kwargs and kwargs['learn_decay']:
            print('learn decay')
            self.decay = torch.nn.Parameter(torch.tensor(self.decay))
            
        
    def forward(self, x, q_info=None):
        #lsq initial
        if self.training and self.init_training_state == 0:
            print('initial scales for the first forward in training')
            
            weight_init_bit = self.weight_bit if self.weight_bit > 1 else 2
            spike_init_bit = self.spike_bit if self.spike_bit > 1 else 2
            # init wegith alpha
            if self.weight_per_layer:
                torch.nn.init.constant_(self.weight_quant.alh, 2 * self.weight.detach().abs().mean() / math.sqrt(2**(weight_init_bit-1)-1))
            else:
                self.weight_quant.alh.data.copy_(2 * self.weight.detach().abs().mean(dim=list(range(1, self.weight.dim())), keepdim=True) / math.sqrt(2**(weight_init_bit-1)-1))
            
            # init wegith alpha
            if self.fea_all_positive:
                for i in range(len(self.spikes_quant)):
                    torch.nn.init.constant_(self.spikes_quant[i].gama, 2 * x.detach().abs().mean() / math.sqrt(2**(self.spike_bit)-1))
            else:
                for i in range(len(self.spikes_quant)):
                    torch.nn.init.constant_(self.spikes_quant[i].gama, 2 * x.detach().abs().mean() / math.sqrt(2**(spike_init_bit-1)-1))
            self.init_training_state.fill_(1)
            
 
        # weight quantization
        weight_q = self.weight_quant(self.weight, self.weight_bit)

        # spike quantization fea
        mem = x.clone()
        output = []
        output.append(self.spikes_quant[0](mem))
        
        valid_T = torch.clamp(self.T.detach().round().int(), 1, self.T_max)
        if valid_T>1:
            mem_res = mem.detach() - output[0].detach()
            for i in range(1, valid_T):
                mem = mem_res * self.decay + x 
                output.append(self.spikes_quant[i](mem))
                mem_res = mem.detach() - output[i].detach()
        
        T_grad = valid_T + self.T - self.T.detach()
        spike_out = torch.stack(output, dim=0).sum(0) / T_grad
        # spike_out = self.spikes_quant[0](x)
        # fea_q = x
        #straght through the learnable params up
        if q_info is not None:
            q_info.append({
                        'spike_bit': torch.stack([ste_clamp(self.spikes_quant[i].bit.clone().abs(), min=1., max=self.spikes_quant[i].bit_bound) for i in range(valid_T)]).squeeze().reshape(-1),
                        'spike_scale': torch.stack([self.spikes_quant[i].gama.clone() for i in range(valid_T)]).squeeze().reshape(-1),
                        'weight_bit': ste_clamp(self.weight_quant.bit.clone().abs().squeeze().reshape(-1), 1.,self.weight_quant.bit_bound),
                        'weight_scale': self.weight_quant.alh.clone().abs().squeeze().reshape(-1),
                        'T': T_grad.clone().reshape(-1),
                        'layer_type':'linear',
                        'fea_numel': spike_out.numel(),
                        'weight_numel':weight_q.numel(),
                        'learn_spike_bit':self.learn_spike_bit,
                        'learn_weight_bit':self.learn_weight_bit,
                        })
            #not straght through the learnable params up
            # q_info.append({
            #             'spike_bit': torch.stack([self.spikes_quant[i].bit.clone().abs() for i in range(valid_T)]).squeeze().reshape(-1),
            #             'spike_scale': torch.stack([self.spikes_quant[i].gama.clone().abs() for i in range(valid_T)]).squeeze().reshape(-1),
            #             'weight_bit': self.weight_quant.bit.clone().abs().squeeze().reshape(-1),
            #             'weight_scale': self.weight_quant.alh.clone().abs().squeeze().reshape(-1),
            #             'T': self.T.clone().reshape(-1),
            #             'layer_type':'linear',
            #             'fea_numel': spike_out.numel(),
            #             'weight_numel':weight_q.numel(),
            #             })
        #add nice and ns-nice cal
        if not self.training and ace_cal:    
            B,C = x.shape
            spike_bit = q_info[-1]['spike_bit'].detach().round()
            weight_bit = q_info[-1]['weight_bit'].detach().round()
            bit_budget = spike_bit * weight_bit
            bit_budget_mean = spike_bit.max() * weight_bit.max()
            
            ops = self.in_features * self.out_features
            sops = ops* torch.min((valid_T*spike_bit.mean()), weight_bit.mean())
            s_ace = (bit_budget * ops).sum()
            s_ace_mean = bit_budget_mean *ops
            
            f_num = 0
            f_act_state_num = 0
            for i,out in enumerate(output):
                f_num += torch.count_nonzero(out)
                f_act_state_num += torch.count_nonzero(out) * ste_clamp(self.spikes_quant[i].bit.clone().abs(), min=1., max=self.spikes_quant[i].bit_bound).squeeze()
            fr = f_num/(x.numel() * valid_T)
            ns_ace = s_ace*fr
            ace = {
                's_ace':s_ace,
                'sops':sops,
                'ns_ace':ns_ace,
                'fr':fr,
                'ops':ops,
                'bit_budget_mean':bit_budget_mean,
                's_ace_mean':s_ace_mean,
                'f_num':f_num,
                'tot_num':x.numel() * valid_T,
                'f_act_state_num':f_act_state_num,
                'neuron_num': spike_out.numel(),
            }
            q_info[-1].update(ace)   
            # print('T:', valid_T)
            
        return F.linear(spike_out, weight_q, self.bias), q_info
    
    @staticmethod
    def inherit_from(ori_module, step=2, **kwargs):
        assert isinstance(ori_module, torch.nn.Linear)
        new_module = SpikeLinear_q(
            in_features = ori_module.in_features,
            out_features = ori_module.out_features,
            bias = ori_module.bias is not None,
            # bit = getattr(ori_module, 'weight_bit', weight_bit),
            step = step,
            fea_all_positive =  False if 'spike_all_positive' not in kwargs else kwargs['spike_all_positive'],
            # max_step = 2 * step,
            **kwargs
        )
        new_module.weight.data.copy_(ori_module.weight.detach())
        if ori_module.bias is not None:
            new_module.bias.data.copy_(ori_module.bias.detach())
        
        return new_module