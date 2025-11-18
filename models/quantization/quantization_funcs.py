import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function



#################################################################################
#
#                   quantization functions for each forwards
#
#################################################################################


##############################weight quant##############################

class u_quant_w_func_alpha_linear_channelwise_div(Function):
    @staticmethod
    def forward(ctx, weight, alpha, bit, learn_weight_bit=True):
        """
        weight:[out_features, in_features]
        alpha:[out_features,1] or [1,1]
        bit:[out_features,1] or [1,1]
        init_weight_bit: float len=1
        """
        # Convert the quantization parameters b to integers by round function
        # bit = torch.round(bit.abs())
        # Convert the quantization parameters b to integers by round function
        bit = torch.round(bit)
        
        #when bit = 1, binary should be specially discussed for using lsq
        non_binary = bit>1
        qmax = (2**(bit-1)-1) * non_binary.float() + (1.- non_binary.float()) * (2**(bit)-1)
        qmax_ = qmax.expand_as(weight)
        non_binary = non_binary.expand_as(weight)
        
        # alpha = alpha.abs()
        eps = torch.tensor(0.00001).float().to(alpha.device)
        alpha = torch.where(alpha > eps, alpha, eps)
        
        w_sign = torch.sign(weight)
        weight_div = weight.div(alpha)
        
        weight_q = torch.min(torch.max(weight_div, -(qmax_)), qmax_)
        weight_q[(weight_div>0.)*(~non_binary)] = qmax_[(weight_div>0.)*(~non_binary)]
        weight_q[(weight_div<=0.)*(~non_binary)] = -qmax_[(weight_div<=0.)*(~non_binary)]
        
        weight_q[non_binary] = torch.floor(weight_q[non_binary].abs() + 0.5) * w_sign[non_binary] # after round, int fea
        
        # init_weight_max = 2**(init_weight_bit-1)-1
        ctx.save_for_backward(weight_div, weight_q, qmax_, alpha) 
        ctx.learn_weight_bit = learn_weight_bit
        ctx.non_binary = non_binary
        # print('==='*20)
        # q_err_f_norm = torch.norm(weight-weight_q.mul(alpha))
        # print('q_err_f_norm:',q_err_f_norm)
        # print('bit:', bit)
        return weight_q.mul(alpha)

    @staticmethod
    def backward(ctx, grad_output):
        grad_weight = grad_output  # grad for weights will not be clipped
        # grad alpha
        weight_div, weight_q, qmax_, alpha = ctx.saved_variables
        # Gradient for step size
        i = (weight_div.abs() <= qmax_).float()
        # w_q_sign = torch.sign(weight_q.mul(alpha)-w0)
        grad_alpha_0 = (weight_q - weight_div)* i
        grad_alpha_1 = (1-i)  * qmax_ * torch.sign(weight_div)
        g_scale = 1.0 / torch.sqrt(weight_div.numel() * qmax_)
        # g_scale = 1.0  #no lsq
        grad_alpha = (grad_output*(grad_alpha_0 + grad_alpha_1)*g_scale).sum(1).reshape(-1,1) 
        # torch.ten()
        # grad for bit
        if ctx.learn_weight_bit:
            grad_b_0 = (1-i) * torch.sign(weight_div) * math.log(2) * alpha * (qmax_ + ctx.non_binary.float())
            grad_b = (grad_output * grad_b_0).sum(1).reshape(-1,1)
        else:
            grad_b = None
        return grad_weight * i, grad_alpha, grad_b, None #torch.zeros_like(grad_b).to(grad_b.device)

# class u_quant_w_func_alpha_linear_channelwise_div(Function):
#     """
#     direct from zzy
#     """
  
#     @staticmethod
#     def forward(ctx, weight, alpha, bit):
#         """
#         weight:[out_features, in_features]
#         alpha:[out_features,1]
#         w_max:[out_features,1]
#         """
#         # Convert the quantization parameters b to integers by round function
#         bit = torch.round(bit.abs())
#         qmax = 2**(bit-1)-1
#         qmax_ = qmax.expand_as(weight)
#         alpha = alpha.abs()
#         w_max = qmax_*alpha
#         alpha_sign = torch.sign(alpha)
#         w_sign = torch.sign(weight)
#         weight_div = weight.div(alpha)
#         weight_div[weight_div>qmax_] = qmax_[weight_div>qmax_]
#         weight_div[weight_div<-qmax_] = -qmax_[weight_div<-qmax_]
#         weight_q = torch.floor(weight_div.abs()+0.5)*w_sign
#         ctx.save_for_backward(weight, weight_div, weight_q, w_max, alpha, alpha_sign)
#         ctx.qmax = qmax_
#         return weight_q.mul(alpha)

#     @staticmethod
#     def backward(ctx, grad_output):
#         weight0, weight, weight_q, w_max, alpha, alpha_sign = ctx.saved_variables
#         # Gradient for weight
#         grad_weight = grad_output 
#         grad_weight[weight>ctx.qmax] = 0
#         grad_weight[weight<-ctx.qmax] = 0
#         # Gradient for step size
#         i = (weight0.abs() <= w_max).float()
#         grad_alpha_0 = (weight_q - weight) * i
#         grad_alpha_1 = (1-i) * torch.sign(weight) * ctx.qmax
#         grad_alpha = 1*(grad_output*(grad_alpha_0 + grad_alpha_1)).sum(1).reshape(-1,1)
#         # Gradient for bit
#         grad_b_0 = (1-i) * torch.sign(weight)*(ctx.qmax+1)*math.log(2)*alpha
#         grad_b = (grad_output * grad_b_0).sum(1).reshape(-1,1)
#         return grad_weight, grad_alpha, grad_b
 
    

class u_quant_w_func_alpha_conv_channelwise_div(u_quant_w_func_alpha_linear_channelwise_div):
    @staticmethod
    def backward(ctx, grad_output):
        
        """
        weight:[out_features, in_features, kernel, kernel]
        alpha:[out_features,1,1,1] or [1,1,1,1]
        bit:[out_features,1,1,1] or [1,1,1,1]
        """
        grad_weight = grad_output  # grad for weights will not be clipped
        # grad alpha
        weight_div, weight_q, qmax_, alpha = ctx.saved_variables
        # Gradient for step size
        i = (weight_div.abs() <= qmax_).float()
        # w_q_sign = torch.sign(weight_q.mul(alpha)-w0)
        grad_alpha_0 = (weight_q - weight_div)* i
        grad_alpha_1 = (1-i)  * qmax_ * torch.sign(weight_div)
        g_scale = 1.0 / torch.sqrt(weight_div.numel() * qmax_)
        # g_scale = 1.0  #no lsq
        grad_alpha = (grad_output*(grad_alpha_0 + grad_alpha_1)*g_scale).sum(tuple(range(1,4))).reshape(-1,1,1,1) * g_scale 
        # torch.ten()
        # grad for bit
        if ctx.learn_weight_bit:
            grad_b_0 = (1-i) * torch.sign(weight_div) * math.log(2) * alpha * (qmax_ + ctx.non_binary.float())
            grad_b = (grad_output * grad_b_0).sum(tuple(range(1,4))).reshape(-1,1,1,1)
        else:
            grad_b = None
        return grad_weight * i, grad_alpha, grad_b, None #torch.zeros_like(grad_b).to(grad_b.device)
    

class u_quant_w_func_alpha_conv1d_channelwise_div(u_quant_w_func_alpha_linear_channelwise_div):
    @staticmethod
    def backward(ctx, grad_output):
        
        """
        weight:[out_features, in_features, kernel, kernel]
        alpha:[out_features,1,1] or [1,1,1]
        bit:[out_features,1,1] or [1,1,1]
        """
        grad_weight = grad_output  # grad for weights will not be clipped
        # grad alpha
        weight_div, weight_q, qmax_, alpha = ctx.saved_variables
        # Gradient for step size
        i = (weight_div.abs() <= qmax_).float()
        # w_q_sign = torch.sign(weight_q.mul(alpha)-w0)
        grad_alpha_0 = (weight_q - weight_div)* i
        grad_alpha_1 = (1-i)  * qmax_ * torch.sign(weight_div)
        g_scale = 1.0 / torch.sqrt(weight_div.numel() * qmax_)
        # g_scale = 1.0  #不用 lsq
        grad_alpha = (grad_output*(grad_alpha_0 + grad_alpha_1)*g_scale).sum(tuple(range(1,3))).reshape(-1,1,1) * g_scale 
        # torch.ten()
        # grad for bit
        if ctx.learn_weight_bit:
            grad_b_0 = (1-i) * torch.sign(weight_div) * math.log(2) * alpha * (qmax_ + ctx.non_binary.float())
            grad_b = (grad_output * grad_b_0).sum(tuple(range(1,3))).reshape(-1,1,1)
        else:
            grad_b = None
        return grad_weight * i, grad_alpha, grad_b, None #torch.zeros_like(grad_b).to(grad_b.device)


##############################feature quant##############################


class u_quant_fea_func_gama_layerwise_div(Function):
    @staticmethod
    def forward(ctx, feature, gama, bit, learn_fea_bit=True):
        """
            args:
                feature: [Batch, c, w, h] or [Batch, c] or [Batch, token, c]
                gama: [1]
                bit: [1]
                init_fea_bit: float len=1
            output:
                fea_a : [N, F]
        """
        # Convert the quantization parameters b to integers by round function
        bit = torch.round(bit)
        
        #when bit = 1, binary should be specially discussed for using lsq
        non_binary = bit>1
        qmax = (2**(bit-1)-1) * non_binary.float() + (1.- non_binary.float()) * (2**(bit)-1)
        
        non_binary = non_binary.expand_as(feature)
        qmax_ = qmax.expand_as(feature)
        # q_bord_line =  torch.zeros_like(qmax_).to(qmax.device) if binary else qmax_
        
        # gama = gama.abs()
        eps = torch.tensor(0.00001).float().to(gama.device)
        gama = torch.where(gama > eps, gama, eps)
        fea_max = qmax_*gama
        
        # gama_sign = torch.sign(gama)
        fea_sign = torch.sign(feature)
        fea_div = feature.div(gama)
        # fea_div[fea_div>qmax_] = qmax_[fea_div>qmax_]
        # fea_div[fea_div<-qmax_] = -qmax_[fea_div<-qmax_] # before round, float fea
        
        fea_q = torch.min(torch.max(fea_div, -(qmax_)), qmax_)
        fea_q[(fea_div>0.)*(~non_binary)] = qmax_[(fea_div>0.)*(~non_binary)]
        fea_q[(fea_div<=0.)*(~non_binary)] = -qmax_[(fea_div<=0.)*(~non_binary)]
        
        fea_q[non_binary] = torch.floor(fea_q[non_binary].abs() + 0.5) * fea_sign[non_binary] # after round, int fea
        
        # init_fea_max = 2**(init_fea_bit-1)-1
        ctx.save_for_backward(feature, fea_div, fea_q, fea_max, gama)
        ctx.qmax = qmax_
        ctx.qshape = gama.shape
        ctx.learn_fea_bit = learn_fea_bit
        ctx.non_binary = non_binary
        return fea_q.mul(gama)
   
    @staticmethod
    def backward(ctx, grad_output):
        grad_feature = grad_output  # grad for grad_b will be clipped
        # grad gama
        fea_0, feature, fea_q, fea_max, gama = ctx.saved_variables
        # Gradient for step size
        i = (fea_0.abs() <= fea_max).float()
        # fea_sign = torch.sign(fea_q.mul(gama)-fea_0)
        grad_gama_0 = (fea_q - feature)* i
        grad_gama_1 = (1-i)  *ctx.qmax*torch.sign(fea_q)
        # if ctx.qmax.mean() != 0:
        #     print()
        g_scale = 1.0 / torch.sqrt(feature.numel() * ctx.qmax)   #
        # g_scale = 1.0  #no lsq
        grad_gama = (grad_output*(grad_gama_0 + grad_gama_1) * g_scale).sum().reshape(ctx.qshape) 
        # torch.ten()
        
        # grad for bit
        if ctx.learn_fea_bit:
            grad_b_0 = (1-i) * torch.sign(fea_q) * math.log(2) * gama * (ctx.qmax + ctx.non_binary.float())
            # if ctx.binary:
            #     grad_b_0 = (1-i) * torch.sign(fea_q)*(ctx.qmax)*math.log(2)*gama
            # else:
            #     grad_b_0 = (1-i) * torch.sign(fea_q)*(ctx.qmax+1)*math.log(2)*gama
            grad_b = (grad_output * grad_b_0).sum().reshape(ctx.qshape)
        else:
            grad_b = None
        # grad_b = torch.clamp(grad_b, )
        # grad_b_f_norm = torch.norm(grad_b)
        # grad_gama_f_norm = torch.norm(grad_gama)
        return grad_feature * i, grad_gama, grad_b, None
    

class u_quant_fea_func_gama_channelwise_div(Function):
    @staticmethod
    def forward(ctx, feature, gama, bit, learn_fea_bit=True):
        """
            args:
                feature: [Batch, c, w, h]
                gama:[1, c, 1, 1] 
                bit: [1, c, 1, 1] 
                init_fea_bit: float len=1
            output:
                fea_a : [N, F]
        """
        # Convert the quantization parameters b to integers by round function
        bit = torch.round(bit)
        
        #when bit = 1, binary should be specially discussed for using lsq
        non_binary = bit>1
        qmax = (2**(bit-1)-1) * non_binary.float() + (1.- non_binary.float()) * (2**(bit)-1)
        
        non_binary = non_binary.expand_as(feature)
        qmax_ = qmax.expand_as(feature)
        # q_bord_line =  torch.zeros_like(qmax_).to(qmax.device) if binary else qmax_
        
        # gama = gama.abs()
        eps = torch.tensor(0.00001).float().to(gama.device)
        gama = torch.where(gama > eps, gama, eps)
        fea_max = qmax_*gama
        
        # gama_sign = torch.sign(gama)
        fea_sign = torch.sign(feature)
        fea_div = feature.div(gama)
        # fea_div[fea_div>qmax_] = qmax_[fea_div>qmax_]
        # fea_div[fea_div<-qmax_] = -qmax_[fea_div<-qmax_] # before round, float fea
        
        fea_q = torch.min(torch.max(fea_div, -(qmax_)), qmax_)
        fea_q[(fea_div>0.)*(~non_binary)] = qmax_[(fea_div>0.)*(~non_binary)]
        fea_q[(fea_div<=0.)*(~non_binary)] = -qmax_[(fea_div<=0.)*(~non_binary)]
        
        fea_q[non_binary] = torch.floor(fea_q[non_binary].abs() + 0.5) * fea_sign[non_binary] # after round, int fea
        
        # init_fea_max = 2**(init_fea_bit-1)-1
        ctx.save_for_backward(feature, fea_div, fea_q, fea_max, gama)
        ctx.qmax = qmax_
        ctx.qshape = gama.shape
        ctx.learn_fea_bit = learn_fea_bit
        ctx.non_binary = non_binary
        return fea_q.mul(gama)
   
    @staticmethod
    def backward(ctx, grad_output):
        grad_feature = grad_output  # grad for grad_b will be clipped
        # grad gama
        fea_0, feature, fea_q, fea_max, gama = ctx.saved_variables
        # Gradient for step size
        i = (fea_0.abs() <= fea_max).float()
        # fea_sign = torch.sign(fea_q.mul(gama)-fea_0)
        grad_gama_0 = (fea_q - feature)* i
        grad_gama_1 = (1-i)  *ctx.qmax*torch.sign(fea_q)
        # if ctx.qmax.mean() != 0:
        #     print()
        g_scale = 1.0 / torch.sqrt(feature.numel() * ctx.qmax)   # 
        # g_scale = 1.0  #no lsq
        grad_gama = (grad_output*(grad_gama_0 + grad_gama_1) * g_scale).sum((0,2,3)).reshape(ctx.qshape) 
        # torch.ten()
        
        # grad for bit
        if ctx.learn_fea_bit:
            grad_b_0 = (1-i) * torch.sign(fea_q) * math.log(2) * gama * (ctx.qmax + ctx.non_binary.float())
            # if ctx.binary:
            #     grad_b_0 = (1-i) * torch.sign(fea_q)*(ctx.qmax)*math.log(2)*gama
            # else:
            #     grad_b_0 = (1-i) * torch.sign(fea_q)*(ctx.qmax+1)*math.log(2)*gama
            grad_b = (grad_output * grad_b_0).sum((0,2,3)).reshape(ctx.qshape)
        else:
            grad_b = None
        # grad_b = torch.clamp(grad_b, )
        # grad_b_f_norm = torch.norm(grad_b)
        # grad_gama_f_norm = torch.norm(grad_gama)
        return grad_feature * i, grad_gama, grad_b, None
    
        





#################################################################################
#
#                   bin shift added 
#
#################################################################################
    
# class spike_quant_fea_func_gama_layerwise_div(Function):
#     @staticmethod
#     def forward(ctx, feature, gama, bit, bin_shift, learn_fea_bit=True, all_positive=True):
#         """
#             args:
#                 feature: [Batch, c, w, h] or [Batch, c] or [Batch, token, c]
#                 gama:[1]
#                 bit: [1]
#                 init_fea_bit: float len=1
#             output:
#                 fea_a : [N, F]
#         """
#         # Convert the quantization parameters b to integers by round function
#         bit = torch.round(bit)
        
#         #when bit = 1, binary should be specially discussed for using lsq
#         non_binary = bit>1
#         if all_positive:
#             qmax = 2**bit-1
#         else:
#             qmax = (2**(bit-1)-1) * non_binary.float() + (1.- non_binary.float()) * (2**(bit)-1)
        
#         bin_shift_ = bin_shift.expand_as(feature)
#         non_binary = non_binary.expand_as(feature)
#         qmax_ = qmax.expand_as(feature)
#         # q_bord_line =  torch.zeros_like(qmax_).to(qmax.device) if binary else qmax_
        
#         # gama = gama.abs()
#         eps = torch.tensor(0.00001).float().to(gama.device)
#         gama = torch.where(gama > eps, gama, eps)
#         fea_max = qmax_*gama
        
#         # gama_sign = torch.sign(gama)
#         fea_sign = torch.sign(feature)
#         fea_div = feature.div(gama)
#         # fea_div[fea_div>qmax_] = qmax_[fea_div>qmax_]
#         # fea_div[fea_div<-qmax_] = -qmax_[fea_div<-qmax_] # before round, float fea
        
#         # binary case. we try shifts 
#         if not all_positive:
#             fea_q = torch.min(torch.max(fea_div, -(qmax_)), qmax_)
#             fea_q[(fea_div>bin_shift_)*(~non_binary)] = 1.
#             fea_q[(fea_div<=bin_shift_)*(~non_binary)] = -1.
#         else:
#             fea_q = torch.min(torch.max(fea_div, qmax_ * 0.), qmax_)
#             fea_q[(fea_div>(bin_shift_+0.5))*(~non_binary)] = 1.
#             fea_q[(fea_div<=(bin_shift_+0.5))*(~non_binary)] = 0.
#         # non-binary case
#         fea_q[non_binary] = torch.floor(fea_q[non_binary].abs() + 0.5) * fea_sign[non_binary] # after round, int fea
        
#         # init_fea_max = 2**(init_fea_bit-1)-1
#         ctx.save_for_backward(feature, fea_div, fea_q, fea_max, gama)
#         ctx.qmax = qmax_
#         ctx.bin_shift_ = bin_shift_
#         ctx.qshape = gama.shape
#         ctx.learn_fea_bit = learn_fea_bit
#         ctx.non_binary = non_binary
#         ctx.all_positive = all_positive
#         return fea_q.mul(gama)
   
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_feature = grad_output  # grad for grad_b will be clipped
#         # grad gama
#         fea_0, feature, fea_q, fea_max, gama = ctx.saved_variables
#         # Gradient for step size
#         qp = (fea_0 > fea_max)
#         qn = (fea_0 < -fea_max) if not ctx.all_positive else (fea_0 < 0.)
#         qm = (~qp) * (~qn)
#         # fea_sign = torch.sign(fea_q.mul(gama)-fea_0)
#         grad_gama_0 = (fea_q - feature) * qm.float()
#         grad_gama_1 = qp.float() * ctx.qmax * torch.sign(fea_q)
#         grad_gama_2 = qn.float() * ctx.qmax * torch.sign(fea_q) if not ctx.all_positive else 0.
#         # if ctx.qmax.mean() != 0:
#         #     print()
#         g_scale = 1.0 / torch.sqrt(feature.numel() * ctx.qmax)   # 
#         # g_scale = 1.0  #no lsq
#         grad_gama = (grad_output*(grad_gama_0 + grad_gama_1 + grad_gama_2) * g_scale).sum().reshape(ctx.qshape) 
        
#         # grad for bit
#         if ctx.learn_fea_bit:
#             grad_b_flag = (~qm).float() if not ctx.all_positive else qp.float()
#             grad_b_0 = grad_b_flag * torch.sign(fea_q) * math.log(2) * gama * (ctx.qmax + ctx.non_binary.float()) 
#             grad_b = (grad_output * grad_b_0).sum().reshape(ctx.qshape)
#         else:
#             grad_b = None
            
#         # grad for bin shift    
#         qbm = (feature >= (ctx.bin_shift_ - 1)) * (feature <= (ctx.bin_shift_ + 1)) if not ctx.all_positive else (feature >= ctx.bin_shift_ + 0.2) * (feature <= (ctx.bin_shift_ + 1))
#         qbm *= ~ctx.non_binary
#         grad_b_shift = (-grad_feature * qbm.float()).sum().reshape(ctx.qshape) * gama
        
#         return grad_feature * qm.float(), grad_gama, grad_b, grad_b_shift, None, None
    

# class spike_quant_fea_func_gama_channelwise_div(spike_quant_fea_func_gama_layerwise_div):               
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_feature = grad_output  # grad for grad_b will be clipped
#         # grad gama
#         fea_0, feature, fea_q, fea_max, gama = ctx.saved_variables
#         # Gradient for step size
#         qp = (fea_0 > fea_max)
#         qn = (fea_0 < -fea_max) if not ctx.all_positive else (fea_0 < 0.)
#         qm = (~qp) * (~qn)
#         # fea_sign = torch.sign(fea_q.mul(gama)-fea_0)
#         grad_gama_0 = (fea_q - feature) * qm.float()
#         grad_gama_1 = qp.float() * ctx.qmax * torch.sign(fea_q)
#         grad_gama_2 = qn.float() * ctx.qmax * torch.sign(fea_q) if not ctx.all_positive else 0.
#         # if ctx.qmax.mean() != 0:
#         #     print()
#         g_scale = 1.0 / torch.sqrt(feature.numel() * ctx.qmax)   # 
#         # g_scale = 1.0  #no lsq
#         grad_gama = (grad_output*(grad_gama_0 + grad_gama_1 + grad_gama_2) * g_scale).sum((0,2,3)).reshape(ctx.qshape) 
#         # torch.ten()
        
#         # grad for bit
#         if ctx.learn_fea_bit:
#             grad_b_flag = (~qm).float() if not ctx.all_positive else qp.float()
#             grad_b_0 = grad_b_flag * torch.sign(fea_q) * math.log(2) * gama * (ctx.qmax + ctx.non_binary.float()) 
            
#             # if ctx.binary:
#             #     grad_b_0 = (1-i) * torch.sign(fea_q)*(ctx.qmax)*math.log(2)*gama
#             # else:
#             #     grad_b_0 = (1-i) * torch.sign(fea_q)*(ctx.qmax+1)*math.log(2)*gama
#             grad_b = (grad_output * grad_b_0).sum((0,2,3)).reshape(ctx.qshape)
#         else:
#             grad_b = None
        
#         # grad for bin shift    
#         qbm = (feature >= (ctx.bin_shift_ - 1)) * (feature <= (ctx.bin_shift_ + 1)) if not ctx.all_positive else (feature >= ctx.bin_shift_ + 0.2) * (feature <= (ctx.bin_shift_ + 1))
#         qbm *= ~ctx.non_binary
#         grad_b_shift = (-grad_feature * qbm.float()).sum((0,2,3)).reshape(ctx.qshape) * gama
        
#         return grad_feature * qm.float(), grad_gama, grad_b, grad_b_shift, None, None
    
    






#################################################################################
#
#                   all fea shift added 
#
#################################################################################
    
class spike_quant_fea_func_gama_layerwise_div(Function):
    @staticmethod
    def forward(ctx, feature, gama, bit, bin_shift, learn_fea_bit=True, all_positive=True):
        """
            args:
                feature: [Batch, c, w, h] or [Batch, c] or [Batch, token, c]
                gama:[1]
                bit: [1]
                init_fea_bit: float len=1
            output:
                fea_a : [N, F]
        """
        # Convert the quantization parameters b to integers by round function
        bit = torch.round(bit)
        
        #when bit = 1, binary should be specially discussed for using lsq
        non_binary = bit>1
        if all_positive:
            qmax = 2**bit-1
        else:
            qmax = (2**(bit-1)-1) * non_binary.float() + (1.- non_binary.float()) * (2**(bit)-1)
        
        bin_shift_ = bin_shift.expand_as(feature)
        non_binary = non_binary.expand_as(feature)
        qmax_ = qmax.expand_as(feature)
        # q_bord_line =  torch.zeros_like(qmax_).to(qmax.device) if binary else qmax_
        
        # gama = gama.abs()
        eps = torch.tensor(0.00001).float().to(gama.device)
        gama = torch.where(gama > eps, gama, eps)
        
        #post-shift
        fea_sign = torch.sign(feature)
        fea_div = feature.div(gama)
        fea_div = fea_div - bin_shift_
        
        #pre-shift
        # fea_sign = torch.sign(feature)
        # fea_div = (feature-bin_shift_).div(gama)
        
        
        if not all_positive:
            fea_q = torch.min(torch.max(fea_div, -(qmax_)), qmax_)  #for all cases, do clamp
            fea_q[(fea_div>0.)*(~non_binary)] = 1.  # binary case
            fea_q[(fea_div<=0.)*(~non_binary)] = -1.    # binary case
        else:
            fea_q = torch.min(torch.max(fea_div, qmax_ * 0.), qmax_)    #for all cases, do clamp
            fea_q[(fea_div>0.5)*(~non_binary)] = 1. # binary case
            fea_q[(fea_div<=0.5)*(~non_binary)] = 0.    # binary case
        # non-binary case
        fea_q[non_binary] = torch.floor(fea_q[non_binary].abs() + 0.5) * fea_sign[non_binary] # after round, int fea
        
        # init_fea_max = 2**(init_fea_bit-1)-1
        ctx.save_for_backward(fea_div, fea_q, gama)
        ctx.qmax = qmax_
        ctx.bin_shift_ = bin_shift_
        ctx.qshape = gama.shape
        ctx.learn_fea_bit = learn_fea_bit
        ctx.non_binary = non_binary
        ctx.all_positive = all_positive
        return (fea_q+bin_shift_).mul(gama)
   
    @staticmethod
    def backward(ctx, grad_output):
        grad_feature = grad_output  # grad for grad_b will be clipped
        # grad gama
        feature, fea_q, gama = ctx.saved_variables
        # Gradient for step size
        qp = (feature > ctx.qmax)
        qn = (feature < -ctx.qmax) if not ctx.all_positive else (feature < 0.)
        qm = (~qp) * (~qn)
        # fea_sign = torch.sign(fea_q.mul(gama)-fea_0)
        grad_gama_0 = (fea_q - feature) * qm.float()
        grad_gama_1 = qp.float() * ctx.qmax * torch.sign(fea_q)
        grad_gama_2 = qn.float() * ctx.qmax * torch.sign(fea_q) if not ctx.all_positive else 0.
        # if ctx.qmax.mean() != 0:
        #     print()
        g_scale = 1.0 / torch.sqrt(feature.numel() * ctx.qmax)   # 
        # g_scale = 1.0  #no lsq
        grad_gama = (grad_output*(grad_gama_0 + grad_gama_1 + grad_gama_2) * g_scale).sum().reshape(ctx.qshape) 
        
        # grad for bit
        if ctx.learn_fea_bit:
            grad_b_flag = (~qm).float() if not ctx.all_positive else qp.float()
            grad_b_0 = grad_b_flag * torch.sign(fea_q) * math.log(2) * gama * (ctx.qmax + ctx.non_binary.float()) 
            grad_b = (grad_output * grad_b_0).sum().reshape(ctx.qshape)
        else:
            grad_b = None
            
        # grad for bin post-shift  
        # grad_b_shift = (-grad_feature * qm.float()).sum().reshape(ctx.qshape) * gama
        grad_b_shift = (grad_feature * g_scale * (~qm).float()).sum().reshape(ctx.qshape) * gama #gradient scaling
        
        
        # grad for bin pre-shift  
        # grad_b_shift = (-grad_feature * qm.float()).sum().reshape(ctx.qshape)
        # grad_b_shift = (-grad_feature * g_scale * qm.float()).sum().reshape(ctx.qshape) #gradient scaling
        
        return grad_feature * qm.float(), grad_gama, grad_b, grad_b_shift, None, None
    

class spike_quant_fea_func_gama_channelwise_div(spike_quant_fea_func_gama_layerwise_div):               
    @staticmethod
    def backward(ctx, grad_output):
        grad_feature = grad_output  # grad for grad_b will be clipped
        # grad gama
        feature, fea_q, gama = ctx.saved_variables
        # Gradient for step size
        qp = (feature > ctx.qmax)
        qn = (feature < -ctx.qmax) if not ctx.all_positive else (feature < 0.)
        qm = (~qp) * (~qn)
        # fea_sign = torch.sign(fea_q.mul(gama)-fea_0)
        grad_gama_0 = (fea_q - feature) * qm.float()
        grad_gama_1 = qp.float() * ctx.qmax * torch.sign(fea_q)
        grad_gama_2 = qn.float() * ctx.qmax * torch.sign(fea_q) if not ctx.all_positive else 0.
        # if ctx.qmax.mean() != 0:
        #     print()
        g_scale = 1.0 / torch.sqrt(feature.numel() * ctx.qmax)   # 
        # g_scale = 1.0  #no lsq
        grad_gama = (grad_output*(grad_gama_0 + grad_gama_1 + grad_gama_2) * g_scale).sum((0,2,3)).reshape(ctx.qshape) 
        # torch.ten()
        
        # grad for bit
        if ctx.learn_fea_bit:
            grad_b_flag = (~qm).float() if not ctx.all_positive else qp.float()
            grad_b_0 = grad_b_flag * torch.sign(fea_q) * math.log(2) * gama * (ctx.qmax + ctx.non_binary.float()) 
            
            # if ctx.binary:
            #     grad_b_0 = (1-i) * torch.sign(fea_q)*(ctx.qmax)*math.log(2)*gama
            # else:
            #     grad_b_0 = (1-i) * torch.sign(fea_q)*(ctx.qmax+1)*math.log(2)*gama
            grad_b = (grad_output * grad_b_0).sum((0,2,3)).reshape(ctx.qshape)
        else:
            grad_b = None
        
        # grad for bin post-shift  
        # grad_b_shift = (-grad_feature * qm.float()).sum((0,2,3)).reshape(ctx.qshape) * gama
        grad_b_shift = (grad_feature * g_scale * (~qm).float()).sum((0,2,3)).reshape(ctx.qshape) * gama
        
        
        # grad for bin pre-shift  
        # grad_b_shift = (-grad_feature * qm.float()).sum((0,2,3)).reshape(ctx.qshape) 
        # grad_b_shift = (-grad_feature * g_scale * qm.float()).sum((0,2,3)).reshape(ctx.qshape)
        
        return grad_feature * qm.float(), grad_gama, grad_b, grad_b_shift, None, None
    
    




#################################################################################
#
#                  no bin shift added 
#
#################################################################################
    
# class spike_quant_fea_func_gama_layerwise_div(Function):
#     @staticmethod
#     def forward(ctx, feature, gama, bit, learn_fea_bit=True, all_positive=True):
#         """
#             args:
#                 feature: [Batch, c, w, h] or [Batch, c] or [Batch, token, c]
#                 gama:[1]
#                 bit: [1]
#                 init_fea_bit: float len=1
#             output:
#                 fea_a : [N, F]
#         """
#         # Convert the quantization parameters b to integers by round function
#         bit = torch.round(bit)
        
#         #when bit = 1, binary should be specially discussed for using lsq
#         non_binary = bit>1
#         if all_positive:
#             qmax = 2**bit-1
#         else:
#             qmax = (2**(bit-1)-1) * non_binary.float() + (1.- non_binary.float()) * (2**(bit)-1)
        
#         non_binary = non_binary.expand_as(feature)
#         qmax_ = qmax.expand_as(feature)
#         # q_bord_line =  torch.zeros_like(qmax_).to(qmax.device) if binary else qmax_
        
#         # gama = gama.abs()
#         eps = torch.tensor(0.00001).float().to(gama.device)
#         gama = torch.where(gama > eps, gama, eps)
#         fea_max = qmax_*gama
        
#         # gama_sign = torch.sign(gama)
#         fea_sign = torch.sign(feature)
#         fea_div = feature.div(gama)
#         # fea_div[fea_div>qmax_] = qmax_[fea_div>qmax_]
#         # fea_div[fea_div<-qmax_] = -qmax_[fea_div<-qmax_] # before round, float fea
        
#         if not all_positive:
#             fea_q = torch.min(torch.max(fea_div, -(qmax_)), qmax_)
#             # binary case
#             fea_q[(fea_div>0.)*(~non_binary)] = 1.
#             fea_q[(fea_div<=0.)*(~non_binary)] = -1.
            
#             # non-binary case
#             fea_q[non_binary] = torch.floor(fea_q[non_binary].abs() + 0.5) * fea_sign[non_binary] # after round, int fea
#         else:
#             fea_q = torch.min(torch.max(fea_div, qmax_ * 0.), qmax_)
#             # binary case. we try shifts 
#             fea_q[(fea_div>0.5)*(~non_binary)] = 1.
#             fea_q[(fea_div<=0.5)*(~non_binary)] = 0.
            
#             # non-binary case
#             fea_q[non_binary] = torch.floor(fea_q[non_binary].abs() + 0.5) * fea_sign[non_binary] # after round, int fea
        
#         # init_fea_max = 2**(init_fea_bit-1)-1
#         ctx.save_for_backward(feature, fea_div, fea_q, fea_max, gama)
#         ctx.qmax = qmax_
#         ctx.qshape = gama.shape
#         ctx.learn_fea_bit = learn_fea_bit
#         ctx.non_binary = non_binary
#         ctx.all_positive = all_positive
#         return fea_q.mul(gama)
   
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_feature = grad_output  # grad for grad_b will be clipped
#         # grad gama
#         fea_0, feature, fea_q, fea_max, gama = ctx.saved_variables
#         # Gradient for step size
#         qp = (fea_0 > fea_max)
#         qn = (fea_0 < -fea_max) if not ctx.all_positive else (fea_0 < 0.)
#         qm = (~qp) * (~qn)
#         # fea_sign = torch.sign(fea_q.mul(gama)-fea_0)
#         grad_gama_0 = (fea_q - feature) * qm.float()
#         grad_gama_1 = qp.float() * ctx.qmax * torch.sign(fea_q)
#         grad_gama_2 = qn.float() * ctx.qmax * torch.sign(fea_q) if not ctx.all_positive else 0.
#         # if ctx.qmax.mean() != 0:
#         #     print()
#         g_scale = 1.0 / torch.sqrt(feature.numel() * ctx.qmax)   # 
#         # g_scale = 1.0  #no lsq
#         grad_gama = (grad_output*(grad_gama_0 + grad_gama_1 + grad_gama_2) * g_scale).sum().reshape(ctx.qshape) 
        
#         # grad for bit
#         if ctx.learn_fea_bit:
#             grad_b_flag = (~qm).float() if not ctx.all_positive else qp.float()
#             grad_b_0 = grad_b_flag * torch.sign(fea_q) * math.log(2) * gama * (ctx.qmax + ctx.non_binary.float()) 
#             grad_b = (grad_output * grad_b_0).sum().reshape(ctx.qshape)
#         else:
#             grad_b = None
            
        
#         return grad_feature * qm.float(), grad_gama, grad_b, None, None
    

# class spike_quant_fea_func_gama_channelwise_div(spike_quant_fea_func_gama_layerwise_div):               
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_feature = grad_output  # grad for grad_b will be clipped
#         # grad gama
#         fea_0, feature, fea_q, fea_max, gama = ctx.saved_variables
#         # Gradient for step size
#         qp = (fea_0 > fea_max)
#         qn = (fea_0 < -fea_max) if not ctx.all_positive else (fea_0 < 0.)
#         qm = (~qp) * (~qn)
#         # fea_sign = torch.sign(fea_q.mul(gama)-fea_0)
#         grad_gama_0 = (fea_q - feature) * qm.float()
#         grad_gama_1 = qp.float() * ctx.qmax * torch.sign(fea_q)
#         grad_gama_2 = qn.float() * ctx.qmax * torch.sign(fea_q) if not ctx.all_positive else 0.
#         # if ctx.qmax.mean() != 0:
#         #     print()
#         g_scale = 1.0 / torch.sqrt(feature.numel() * ctx.qmax)   # 
#         # g_scale = 1.0  #no lsq
#         grad_gama = (grad_output*(grad_gama_0 + grad_gama_1 + grad_gama_2) * g_scale).sum((0,2,3)).reshape(ctx.qshape) 
#         # torch.ten()
        
#         # grad for bit
#         if ctx.learn_fea_bit:
#             grad_b_flag = (~qm).float() if not ctx.all_positive else qp.float()
#             grad_b_0 = grad_b_flag * torch.sign(fea_q) * math.log(2) * gama * (ctx.qmax + ctx.non_binary.float()) 
            
#             # if ctx.binary:
#             #     grad_b_0 = (1-i) * torch.sign(fea_q)*(ctx.qmax)*math.log(2)*gama
#             # else:
#             #     grad_b_0 = (1-i) * torch.sign(fea_q)*(ctx.qmax+1)*math.log(2)*gama
#             grad_b = (grad_output * grad_b_0).sum((0,2,3)).reshape(ctx.qshape)
#         else:
#             grad_b = None
        
        
#         return grad_feature * qm.float(), grad_gama, grad_b, None, None
    
    


class Quantizer(torch.nn.Module):
    def __init__(self, bit):
        super().__init__()

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError
    

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad



class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = torch.nn.Parameter(torch.ones(1))

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = torch.nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = torch.nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x
