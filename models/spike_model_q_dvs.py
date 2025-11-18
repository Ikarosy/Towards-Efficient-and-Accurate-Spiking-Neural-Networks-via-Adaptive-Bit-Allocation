import torch
import torch.nn as nn
from models.spike_layer_q_dvs import SpikeConv, LIFAct, tdBatchNorm2d, SpikePool, SpikeModule, myBatchNorm3d, SpikeConv2d_q, SpikeLinear_q, SpikeConv2d_full, SpikeLinear_full, Dcls1d_q
# from models.spike_block_q import specials_q
from models.quantization.quantization_modules import qConv2d, qLinear
from DCLS.construct.modules import Dcls1d, _DclsNd, ConstructKernel1d
# from model_dq import fake_ssa
import copy


def ste_round(x):
    return torch.round(x) - x.detach() + x

first_layer_88=False
first_layer_norm= False
ssa_full = False
                        # #kargs
                        # learn_spike_bit=args.learn_spike_bit, learn_weight_bit=args.learn_weight_bit, learn_time_step=args.learn_time_step,
                        # init_spike_bit=args.init_spike_bit, init_weight_bit=args.init_weight_bit,
                        # weight_per_layer=args.weight_quant_per_layer, spike_per_layer=args.spike_quant_per_layer, 
                        # spike_all_positive = args.spike_quant_all_positive

class SpikeModel_q_dvs(SpikeModule):

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
        # if kwargs['spike_all_positive']:
        #     self.first_conv_not_all_positive_flag = False
        # else:
        #     self.first_conv_not_all_positive_flag = True
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

            elif isinstance(child_module, nn.Conv2d):
                # print('found conv2d, converted')
                special_layer = getattr(child_module, 'special_layer', None)
                kwargs = copy.copy(self.kwargs)
                kwargs['special_layer'] = special_layer
                if special_layer == 'first conv':
                    kwargs['spike_all_positive'] = False
                    print('layer name: {}, is deemed as the 1st layer, now all pos cancelled'.format(name))
                    if first_layer_88:
                        kwargs['init_spike_bit'], kwargs['init_weight_bit']=8, 8
                        kwargs['learn_spike_bit'], kwargs['learn_weight_bit'] = False, False
                        print('layer name: {}, is deemed as the 1st layer, now 88 q static'.format(name))
                    elif first_layer_norm:
                        kwargs['init_spike_bit'], kwargs['init_weight_bit']=8, 8
                        kwargs['learn_spike_bit'], kwargs['learn_weight_bit'] = False, False
                        print('layer name: {}, is deemed as the 1st layer, now 88 q full static'.format(name))
                        setattr(module, name, SpikeConv2d_full.inherit_from(child_module, step=1, **kwargs))
                    else:
                        setattr(module, name, SpikeConv2d_q.inherit_from(child_module, step=step, **kwargs))
                        raise NotImplemented
                        
                else:
                    kwargs = self.kwargs
                    setattr(module, name, SpikeConv2d_q.inherit_from(child_module, step=step, **kwargs))
                
            elif isinstance(child_module, nn.Linear):
                # print('found qLinear, converted')
                special_layer = getattr(child_module, 'special_layer', None)
                kwargs = copy.copy(self.kwargs)
                kwargs['special_layer'] = special_layer
                if special_layer=='last linear':
                    print("last linear layer is deemed as this one: {}".format(name))
                    kwargs['learn_time_step'] = False
                    setattr(module, name, SpikeLinear_full.inherit_from(child_module, step=1, **kwargs))
                else:
                    setattr(module, name, SpikeLinear_q.inherit_from(child_module, step=step, **kwargs))

            elif isinstance(child_module, _DclsNd):
                    # print('found qLinear, converted')
                    special_layer = getattr(child_module, 'special_layer', None)
                    kwargs = copy.copy(self.kwargs)
                    kwargs['special_layer'] = special_layer
                    if special_layer=='last dcls1d':
                        print("last dcls1d layer is deemed as this one: {}".format(name))
                        kwargs['learn_time_step'] = False
                        setattr(module, name, Dcls1d_q.inherit_from(child_module, step=1, **kwargs))
                    elif special_layer=='first dcls1d':
                        print("first dcls1d layer is deemed as this one: {}".format(name))
                        kwargs['init_spike_bit'], kwargs['init_weight_bit']=4, 4
                        kwargs['learn_spike_bit'], kwargs['learn_weight_bit'] = False, False
                        setattr(module, name, Dcls1d_q.inherit_from(child_module, step=step, **kwargs))
                    else:
                        setattr(module, name, Dcls1d_q.inherit_from(child_module, step=step, **kwargs))


            else:
                self.spike_module_refactor(child_module, step=step)

    def forward(self, input, is_drop=None):
        # print(input.shape)
        # if len(input.shape) == 4:
        #     input = input.repeat(self.step, 1, 1, 1, 1)
        # else:
        #     input = input.permute([1, 0, 2, 3, 4])
        
        out = self.model(input)
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
        try:
            x = torch.randn(1,1,2,48,48).to(self.model.conv1.weight.device)
        except AttributeError:
            if hasattr(self.model, 'in_channels') and hasattr(self.model, 'linear1'):
                x = torch.randn(1,1,self.model.in_channels).to(self.model.linear1.weight.device)
            elif hasattr(self.model, 'in_channels') and hasattr(self.model, 'dcls1d1'):
                x = torch.randn(1,1,self.model.in_channels).to(self.model.dcls1d1.weight.device)
            else:
                x = torch.randn(1,1,700).to(self.model.linear1.weight.device)
        self.forward(x)
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
                
                elif  layer_info['layer_type'] == 'ssa_func':
                    spike_scales.append(layer_info['spike_scale_q'].detach().cpu())
                    spike_bits.append(layer_info['spike_bit_q'].detach().round().cpu())
                    
                    spike_scales.append(layer_info['spike_scale_k'].detach().cpu())
                    spike_bits.append(layer_info['spike_bit_k'].detach().round().cpu())
                    
                    spike_scales.append(layer_info['spike_scale_v'].detach().cpu())
                    spike_bits.append(layer_info['spike_bit_v'].detach().round().cpu())
                    
                else:
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
            elif  layer_info['layer_type'] == 'ssa_func':
                fea_numel_sum += layer_info['fea_numel_q']
                fea_numel_sum += layer_info['fea_numel_k']
                fea_numel_sum += layer_info['fea_numel_v']
            else:
                weight_numel_sum += layer_info['weight_numel']
                fea_numel_sum += layer_info['fea_numel']
            
        with torch.no_grad():
            for layer_info in self.q_info:
                if (not layer_info.get('learn_spike_bit', True)) and (not layer_info.get('learn_weight_bit', True)):
                #     # print('skip', layer_info['layer_type'])
                    continue
                
                elif  layer_info['layer_type'] == 'ssa_func':
                    spike_bit_sum.append(layer_info['spike_bit_q'].detach().round().mean() * layer_info['fea_numel_q'] / fea_numel_sum) #mem
                    spike_bit_sum.append(layer_info['spike_bit_k'].detach().round().mean() * layer_info['fea_numel_k'] / fea_numel_sum) #mem
                    spike_bit_sum.append(layer_info['spike_bit_v'].detach().round().mean() * layer_info['fea_numel_v'] / fea_numel_sum) #mem
                    
                    spike_step_sum.append(layer_info['T'].detach().round() * (layer_info['fea_numel_q']+layer_info['fea_numel_k']+layer_info['fea_numel_v']) / fea_numel_sum) #nice
                    
                else:
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
            elif  layer_info['layer_type'] == 'ssa_func':
                fea_numel_sum += layer_info['fea_numel_q']
                fea_numel_sum += layer_info['fea_numel_k']
                fea_numel_sum += layer_info['fea_numel_v']
            else:
                weight_numel_sum += layer_info['weight_numel']
                fea_numel_sum += layer_info['fea_numel']
            
        for layer_info in self.q_info:
            if (not layer_info.get('learn_spike_bit', True)) and (not layer_info.get('learn_weight_bit', True)):
                # print('skip', layer_info['layer_type'])
                continue
            elif  layer_info['layer_type'] == 'ssa_func':
                spike_bit_sum.append(ste_round(layer_info['spike_bit_q']).mean() * layer_info['fea_numel_q'] / fea_numel_sum) #mem
                spike_bit_sum.append(ste_round(layer_info['spike_bit_k']).mean() * layer_info['fea_numel_k'] / fea_numel_sum) #mem
                spike_bit_sum.append(ste_round(layer_info['spike_bit_v']).mean() * layer_info['fea_numel_v'] / fea_numel_sum) #mem
                
                spike_step_sum.append(ste_round(layer_info['T']) * (layer_info['fea_numel_q']+layer_info['fea_numel_k']+layer_info['fea_numel_v']) / fea_numel_sum) #nice
                
            else:
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
        }
        for layer_info in self.q_info:
            if (not layer_info.get('learn_spike_bit', True)) and (not layer_info.get('learn_weight_bit', True)):
                # print('skip', layer_info['layer_type'])
                continue
            tot_ace['s_ace']+=layer_info['s_ace']
            tot_ace['sops']+=layer_info['sops']
            tot_ace['ns_ace']+=layer_info['ns_ace']
            tot_ace['ops']+=layer_info['ops']
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
        
    