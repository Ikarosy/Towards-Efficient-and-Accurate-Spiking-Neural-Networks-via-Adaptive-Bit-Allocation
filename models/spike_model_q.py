import torch
import torch.nn as nn
from models.spike_layer_q import SpikeConv, LIFAct, tdBatchNorm2d, SpikePool, SpikeModule, myBatchNorm3d, SpikeConv2d_q, SpikeLinear_q
from models.spike_block_q import specials_q
from models.quantization.quantization_modules import qConv2d, qLinear
import copy


def ste_round(x):
    return torch.round(x) - x.detach() + x

first_layer_88=True
                        # #kargs
                        # learn_spike_bit=args.learn_spike_bit, learn_weight_bit=args.learn_weight_bit, learn_time_step=args.learn_time_step,
                        # init_spike_bit=args.init_spike_bit, init_weight_bit=args.init_weight_bit,
                        # weight_per_layer=args.weight_quant_per_layer, spike_per_layer=args.spike_quant_per_layer, 
                        # spike_all_positive = args.spike_quant_all_positive

class SpikeModel_q(SpikeModule):

    def __init__(self, model: nn.Module, step=2, **kwargs):
        '''
        #kargs
        learn_spike_bit=args.learn_spike_bit, learn_weight_bit=args.learn_weight_bit, learn_time_step=args.learn_time_step,
        init_spike_bit=args.init_spike_bit, init_weight_bit=args.init_weight_bit
        weight_per_layer=args.weight_quant_per_layer, spike_per_layer=args.spike_quant_per_layer, 
        spike_all_positive = args.spike_quant_all_positive
        '''
        super().__init__()
        self.model = model
        self.step = step
        self.q_info = None
        self.kwargs = kwargs
        if kwargs['spike_all_positive']:
            self.first_conv_not_all_positive_flag = False
        else:
            self.first_conv_not_all_positive_flag = True
        self.spike_module_refactor(self.model, step=step)
        self.weight_numel_sum = None
        self.fea_numel_sum = None
        
        
        
    def spike_module_refactor(self, module: nn.Module, step=2):
        """
        Recursively replace the normal conv2d and Linear layer to SpikeLayer
        """
        for name, child_module in module.named_children():
            # if type(child_module) in specials_q:
            #     setattr(module, name, specials_q[type(child_module)](child_module, step=step))

            if isinstance(child_module, nn.Sequential):
                self.spike_module_refactor(child_module, step=step)

            elif isinstance(child_module, qConv2d):
                # print('found conv2d, converted')
                if not self.first_conv_not_all_positive_flag:
                    self.first_conv_not_all_positive_flag = True
                    kwargs = copy.copy(self.kwargs) 
                    kwargs['spike_all_positive'] = False
                    if first_layer_88:
                        kwargs['init_spike_bit'], kwargs['init_weight_bit']=8, 8
                        kwargs['learn_spike_bit'], kwargs['learn_weight_bit'] = False, False
                        print('layer name: {}, is deemed as the 1st layer, now 88 q static'.format(name))
                    print('layer name: {}, is deemed as the 1st layer, now all pos cancelled'.format(name))
                else:
                    kwargs = self.kwargs
                
                setattr(module, name, SpikeConv2d_q.inherit_from(child_module, step=step, **kwargs))
                
            elif isinstance(child_module, qLinear):
                # print('found qLinear, converted')
                setattr(module, name, SpikeLinear_q.inherit_from(child_module, step=step, **self.kwargs))

            # elif isinstance(child_module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
            #     setattr(module, name, SpikePool(child_module, step=step))

            # elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
            #     setattr(module, name, nn.Identity())
                
            # # elif isinstance(child_module, nn.BatchNorm2d):
            #     setattr(module, name, SpikeConv(child_module, step=step))
            else:
                self.spike_module_refactor(child_module, step=step)

    def forward(self, input, use_tet=False, is_drop=False):
        
        # if len(input.shape) == 4:
        #     input = input.repeat(self.step, 1, 1, 1, 1)
        # else:
        #     input = input.permute([1, 0, 2, 3, 4])
        if use_tet:
            out = self.model(input, use_tet, is_drop)
        else:
            out = self.model(input, is_drop)
        self.q_info = self.model.q_info
        return out

    def set_spike_state(self, use_spike=True):
        self._spiking = use_spike
        for m in self.model.modules():
            if isinstance(m, SpikeModule):
                m.set_spike_state(use_spike)

    def set_spike_before(self, name):
        self.set_spike_state(False)
        for n, m in self.model.named_modules():
            if isinstance(m, SpikeModule):
                m.set_spike_state(True)
            if name == n:
                break


    # @torch.no_grad()
    def get_bits_scales(self):
        if self.model.fc.out_features == 100 or self.model.fc.out_features == 10:
            x = torch.randn(1,3,32,32).to(self.model.conv1.weight.device)
        elif self.model.fc.out_features == 1000:   #imagenet
            x = torch.randn(1,3,224,224).to(self.model.conv1.weight.device)
        else:
            raise NotImplementedError
        self.forward(x, is_drop=False)
        spike_scales = []
        spike_bits = []
        weight_scales = []
        weight_bits = []
        time_steps = []
        with torch.no_grad():
            for layer_info in self.q_info:
                if (not layer_info.get('learn_spike_bit', True)) and (not layer_info.get('learn_weight_bit', True)):
                    # print('skip', layer_info['layer_type'])
                    continue
                
                spike_scales.append(layer_info['spike_scale'].detach().cpu())
                spike_bits.append(layer_info['spike_bit'].detach().round().cpu())
                weight_scales.append(layer_info['weight_scale'].detach().cpu())
                weight_bits.append(layer_info['weight_bit'].detach().round().cpu())
                time_steps.append(layer_info['T'].detach().cpu())
                
            return {'spike_scales':spike_scales,
                    'spike_bits':spike_bits,
                    'weight_scales':weight_scales,
                    'weight_bits':weight_bits,
                    'time_steps': time_steps,
                }
    
    def get_bit_weighted_mean(self):
        weight_bit_sum = []
        spike_bit_sum = []
        spike_step_sum = []
        
        
        weight_numel_sum = 0
        fea_numel_sum = 0
        for layer_info in self.q_info:
            if (not layer_info.get('learn_spike_bit', True)) and (not layer_info.get('learn_weight_bit', True)):
                # print('skip', layer_info['layer_type'])
                continue
            weight_numel_sum += layer_info['weight_numel']
            fea_numel_sum += layer_info['fea_numel']
            
        with torch.no_grad():
            for layer_info in self.q_info:
                if (not layer_info.get('learn_spike_bit', True)) and (not layer_info.get('learn_weight_bit', True)):
                #     # print('skip', layer_info['layer_type'])
                    continue
                weight_bit_sum.append(layer_info['weight_bit'].detach().round().mean() * layer_info['weight_numel'] / weight_numel_sum) #mem
                spike_bit_sum.append(layer_info['spike_bit'].detach().round().mean() * layer_info['fea_numel'] / fea_numel_sum) #mem
                spike_step_sum.append(layer_info['T'].detach().round() * layer_info['fea_numel'] / fea_numel_sum) #nice
            
            # print(weight_bit_sum)
            # print(len(weight_bit_sum))
            weight_bit_sum = torch.stack(weight_bit_sum, dim=0).sum()
            spike_bit_sum = torch.stack(spike_bit_sum, dim=0).sum()
            spike_step_sum = torch.stack(spike_step_sum, dim=0).sum()
        
        return spike_bit_sum, weight_bit_sum, spike_step_sum
    
    def penalty_anneal(self, p, ep):
        pass
    
    def get_bit_loss(self, target_weight_bit=3.5, target_spike_bit=1.5, target_spike_step=1.5, 
                     weight_bit_penalty=1e-2, spike_bit_penalty=1e-2, spike_step_penalty=1e-2):
        
        weight_bit_sum = []
        spike_bit_sum = []
        spike_step_sum = []
        
        
        weight_numel_sum = 0
        fea_numel_sum = 0
        for layer_info in self.q_info:
            if (not layer_info.get('learn_spike_bit', True)) and (not layer_info.get('learn_weight_bit', True)):
                # print('skip', layer_info['layer_type'])
                continue
            weight_numel_sum += layer_info['weight_numel']
            fea_numel_sum += layer_info['fea_numel']
            
        for layer_info in self.q_info:
            if (not layer_info.get('learn_spike_bit', True)) and (not layer_info.get('learn_weight_bit', True)):
                # print('skip', layer_info['layer_type'])
                continue
            weight_bit_sum.append(ste_round(layer_info['weight_bit']).mean() * layer_info['weight_numel'] / weight_numel_sum) #mem
            spike_bit_sum.append(ste_round(layer_info['spike_bit']).mean() * layer_info['fea_numel'] / fea_numel_sum) #mem
            spike_step_sum.append(ste_round(layer_info['T']) * layer_info['fea_numel'] / fea_numel_sum) #nice
        
        weight_bit_sum = torch.stack(weight_bit_sum, dim=0).sum()
        spike_bit_sum = torch.stack(spike_bit_sum, dim=0).sum()
        spike_step_sum = torch.stack(spike_step_sum, dim=0).sum()
        
        loss_weight_bit = torch.norm(weight_bit_sum-target_weight_bit) * weight_bit_penalty
        loss_spike_bit = torch.norm(spike_bit_sum-target_spike_bit) * spike_bit_penalty
        loss_spike_step = torch.norm(spike_step_sum-target_spike_step) * spike_step_penalty
        
        return loss_weight_bit, loss_spike_bit, loss_spike_step
    
    def get_ace(self):
        assert not self.training
        tot_ace = {
            's_ace':0,
            'sops':0,
            'ns_ace':0,
            'ops':0,
            'fr':[],
            's_ace_mean':0,
            'f_num':0.,
            'tot_num':0.,
            'f_act_state_num':0.,
            'neuron_num': 0.,
        }
        for layer_info in self.q_info:
            if (not layer_info.get('learn_spike_bit', True)) and (not layer_info.get('learn_weight_bit', True)):
                # print('skip', layer_info['layer_type'])
                continue
            tot_ace['s_ace']+=layer_info['s_ace']
            tot_ace['sops']+=layer_info['sops']
            tot_ace['ns_ace']+=layer_info['ns_ace']
            tot_ace['ops']+=layer_info['ops']
            tot_ace['s_ace_mean']+=layer_info['s_ace_mean']
            tot_ace['f_num']+=layer_info['f_num']
            tot_ace['tot_num']+=layer_info['tot_num']
            tot_ace['f_act_state_num']+=layer_info['f_act_state_num']
            tot_ace['neuron_num']+=layer_info['neuron_num']
            
            tot_ace['fr'].append(layer_info['fr'])
            
        return tot_ace
    
    
    # def get_ace_loss(self, target_weight_bit=3.5, target_spike_bit=1.5, target_spike_step=1.5, 
    #                  weight_bit_penalty=1e-2, spike_bit_penalty=1e-2, spike_step_penalty=1e-2):
        
    #     weight_bit_sum = []
    #     spike_bit_sum = []
    #     spike_step_sum = []
        
    #     weight_numel_sum = 0
    #     fea_numel_sum = 0
    #     for layer_info in self.q_info:
    #         weight_numel_sum += layer_info['weight_numel']
    #         fea_numel_sum += layer_info['fea_numel']
        
    #     for layer_info in self.q_info:
    #         weight_bit_sum.append(layer_info['weight_bit'].mean() * layer_info['weight_numel'] / weight_numel_sum) #mem
    #         spike_bit_sum.append(layer_info['spike_bit'].mean() * layer_info['fea_numel'] / fea_numel_sum) #mem
    #         spike_step_sum.append(layer_info['T'] * layer_info['fea_numel'] / fea_numel_sum) #nice
        
    #     # print(weight_bit_sum)
    #     # print(len(weight_bit_sum))
    #     weight_bit_sum = torch.stack(weight_bit_sum, dim=0).sum()
    #     spike_bit_sum = torch.stack(spike_bit_sum, dim=0).sum()
    #     spike_step_sum = torch.stack(spike_step_sum, dim=0).sum()
        
    #     # print('weight_bit_sum', weight_bit_sum.item())
    #     # print('spike_bit_sum', spike_bit_sum.item())
    #     # print('spike_step_sum', spike_step_sum.item())
        
    #     loss_weight_bit = torch.norm(weight_bit_sum-target_weight_bit) * weight_bit_penalty
    #     loss_spike_bit = torch.norm(spike_bit_sum-target_spike_bit) * spike_bit_penalty
    #     loss_spike_step = torch.norm(spike_step_sum-target_spike_step) * spike_step_penalty
        
        
    #     # print('weight_bit_sum', weight_bit_sum.item())
    #     # print('spike_bit_sum', spike_bit_sum.item())
    #     # print('spike_step_sum', spike_step_sum.item())
        
    #     return loss_weight_bit, loss_spike_bit, loss_spike_step
        
    