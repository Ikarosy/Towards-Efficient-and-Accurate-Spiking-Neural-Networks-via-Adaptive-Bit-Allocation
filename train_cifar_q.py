import argparse
import os

#os.system('wandb login xxx')
#import wandb
import time
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from kdutils import seed_all, GradualWarmupScheduler, build_data, build_imagenet, dvs_load_data, dvs128_load_data, binned_SHD_dataloaders, SHD_dataloaders
#from torchvision.models.resnet import resnet18
from torchvision import transforms
# from models import *
from models.init import init_weights, split_weights, init_bias, classify_params
from models.resnet_q import resnet20_cifar_q, ResNet34_q, SEWResNet34_q, ResNet18_q, ResNet18_cifar_q
from models.resnet_dvs import resnet19_dvs
from models.vgg_q import vggsnn_dvs_q
from models.d128snn import d128snn_shd_q
from models.d128snn_delays import d128snn_shd_delays_q
from models.resnet import SEWResNet34, ResNet18, ResNet18_cifar
from models.spike_model_q import SpikeModel_q
from models.spike_model_q_dvs import SpikeModel_q_dvs
from models.quantization.quantization_modules import weight_bit,fea_bit
from utils import accuracy, AvgrageMeter, TET_loss
from functools import partial
# from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
#from models import ImageNet_cnn, ImageNet_snn
#from loss_kd import feature_loss,  logits_loss
from spikingjelly.activation_based import functional



parser = argparse.ArgumentParser(description="Q_SNN_Training")


parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name',
                    choices=['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet', 'DVSG', 'DVSCIFAR', 'dvsgesture', 'SHD', 'binned_SHD'])

parser.add_argument("--datapath", type=str, default='/mnt/lustre/GPU8/home/yaoxingting/CIFAR10')

parser.add_argument('--arch', default='resnet20_cifar_q', type=str, help='arch name',
                    choices=['resnet20_cifar', 'resnet19_cifar', 'resnet20_cifar_modified', 'ResNet18', 'ResNet34', 'ResNet18_q', 'ResNet18_cifar_q', 'ResNet18_cifar',
                             'resnet20_cifar_q', 'ResNet34_q', 'SEWResNet34', 'SEWResNet34_q', 'resnet20_cifar_real_q', 'resnet19_dvs', 'vggsnn_dvs_q', 'vggsnn', 'd128snn_shd_q', 'd128snn_shd_delays_q'])
parser.add_argument('--modeltag', type=str, default='SNN', help='decide the name of the exp.')
parser.add_argument("--batch", type=int, default=40)
parser.add_argument("--epochs", type=int, default=320)
parser.add_argument('--pretrained', type=str, default=None, help='path to the pretrained model ck')
parser.add_argument('--resume', type=str, default=None, help='path to the resume model ck')



parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
# parser.add_argument('--lr_spike_scale', default=1e-3, type=float)
parser.add_argument('--lr_q_weight_bit',type=float,default=1e-1)
parser.add_argument('--lr_q_weight_scale',type=float,default=1e-1)
parser.add_argument('--lr_q_weight_shift',type=float,default=1e-3)

parser.add_argument("--weight_quant_per_layer", action='store_true', default=False)
parser.add_argument("--spike_quant_per_layer", action='store_true', default=False)
parser.add_argument("--spike_quant_all_positive", action='store_true', default=False)

parser.add_argument('--lr_q_feature_bit',type=float,default=1e-1)
parser.add_argument('--lr_q_feature_scale',type=float,default=1e-1)
parser.add_argument('--lr_q_feature_shift',type=float,default=1e-3)

parser.add_argument('--lr_q_time_step',type=float,default=1e-2)

parser.add_argument('--clip_feat_bit',type=float,default=1.)
parser.add_argument('--clip_weight_bit',type=float,default=1.)

parser.add_argument("--learn_weight_bit", action='store_true', default=False)
parser.add_argument("--learn_spike_bit", action='store_true', default=False)
parser.add_argument("--learn_time_step", action='store_true', default=False)


parser.add_argument("--use_bit_loss", action='store_true', default=False)

parser.add_argument('--init_weight_bit',type=int,default=4)
parser.add_argument('--init_spike_bit',type=int,default=2)
parser.add_argument('--init_spike_step',type=int,default=2)

parser.add_argument('--weight_bit_tar',type=float,default=4)
parser.add_argument('--spike_bit_tar',type=float,default=1.5)
parser.add_argument('--spike_step_tar',type=float,default=1.5)

parser.add_argument('--weight_bit_penalty',type=float,default=1e-2)
parser.add_argument('--spike_bit_penalty',type=float,default=1e-2)
parser.add_argument('--spike_step_penalty',type=float,default=1e-2)


parser.add_argument("--beta", type=float, default=10.)


parser.add_argument('--spiking_mode', default=False, action='store_true', help='use spiking network')  
# parser.add_argument('--step', default=2, type=int, help='snn step')
parser.add_argument('--seed', type=int, default=9, metavar='S',
                    help='random seed (default: 9)')


parser.add_argument('--renew_switch_epoch',type=int,default=-1, help='第renew_switch_epoch个 epoch 时 关闭 renewal机制；如果renew_switch_epoch是负的，则进入百分比模式；如果是 0，则不开启 renewal 机制')
parser.add_argument('--renew_switch_least_epoch',type=int,default=-1, help='renewal机制在<renew_switch_least_epoch时绝对不会关，如果renew_switch_least_epoch是负的，关闭此功能。注意此功能仅在renew_switch_epoch<0的时候才会开启')

parser.add_argument("--decay", type=float, default=0.5)
parser.add_argument("--learn_decay", action='store_true', default=False)

#ImageNet args
parser.add_argument("--warm_up", action='store_true', default=False)
parser.add_argument("--amp", action='store_true', default=False)


#dvs args
parser.add_argument('--dsr_da', default=False, action='store_true', help='use spiking network')  
parser.add_argument('--nda_da', default=False, action='store_true', help='use spiking network')  
parser.add_argument('--dsr_da128', default=False, action='store_true', help='use spiking network')  
parser.add_argument('--nda_da128', default=False, action='store_true', help='use spiking network')  
parser.add_argument('--T', default=10, type=int, help='simulation steps')
parser.add_argument('--test_only', default=False, action='store_true', help='use spiking network')  

parser.add_argument('--use_tet', default=False, action='store_true', help='use TET')  

#


args = parser.parse_args()
seed_all(args.seed)

if torch.cuda.device_count() > 1 and args.dataset == 'ImageNet':
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
else:
    local_rank = None
     
if args.use_tet:
    assert 'dvs' in args.dataset.lower() or 'shd' in args.dataset.lower()

best_acc = 0.3
sta = time.time()

# ----------------------------
for k,v in sorted(vars(args).items()):
    print(k,'=',v)
    
######## input model #######
if 'imagenet' in args.dataset.lower():
    model = eval(args.arch)(num_classes= 1000)
else:
    use_cifar10 = args.dataset == 'CIFAR10'
    if 'dvs' in args.arch and args.dataset=='dvsgesture':
        model = eval(args.arch)(num_classes= 11 )
    elif 'dvs' in args.arch:
        model = eval(args.arch)(num_classes= 10 )
    elif 'shd' in args.arch and 'binned' in args.dataset.lower():
        model = eval(args.arch)(in_channels=140, num_classes= 20, args=args)
    elif 'shd' in args.arch:
        model = eval(args.arch)(in_channels=700, num_classes= 20, args=args)
    else:
        model = eval(args.arch)(num_classes= 10 if use_cifar10 else 100)

if args.resume is not None:
    print('loading resumed model')
    checkpoint = torch.load(args.resume, map_location='cpu')
    
elif args.pretrained is not None:
    print('loading pretrained model')
    checkpoint = torch.load(args.pretrained, map_location='cpu')
    missed_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
    print('Some items are missed or mismatched:', missed_keys)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")


######## save model #######
model_save_name = './logs/' + args.modeltag + '.pth'


######## load weight #######
#model.load_state_dict(torch.load('raw/ann-resnet18.pth', map_location='cpu'))

######## change to snn #######
if args.spiking_mode is True and ('dvs' in args.dataset.lower() or 'shd' in args.dataset.lower()):
    model = SpikeModel_q_dvs(model, args.init_spike_step, 
                         #kargs
                         learn_spike_bit=args.learn_spike_bit, learn_weight_bit=args.learn_weight_bit, learn_time_step=args.learn_time_step,
                         init_spike_bit=args.init_spike_bit, init_weight_bit=args.init_weight_bit,
                         weight_per_layer=args.weight_quant_per_layer, spike_per_layer=args.spike_quant_per_layer, 
                         spike_all_positive = args.spike_quant_all_positive,
                         decay=args.decay, learn_decay=args.learn_decay
                         )

elif args.spiking_mode is True:
    model = SpikeModel_q(model, args.init_spike_step, 
                         #kargs
                         learn_spike_bit=args.learn_spike_bit, learn_weight_bit=args.learn_weight_bit, learn_time_step=args.learn_time_step,
                         init_spike_bit=args.init_spike_bit, init_weight_bit=args.init_weight_bit,
                         weight_per_layer=args.weight_quant_per_layer, spike_per_layer=args.spike_quant_per_layer, 
                         spike_all_positive = args.spike_quant_all_positive,
                         decay=args.decay, learn_decay=args.learn_decay
                         )
    model.set_spike_state(True)


if args.resume is not None:
    #resume
    # 'state_dict', 'best_epoch', 'best_acc', "optimizer", 
    # 'scheduler', 'scaler', 'q_info',
    new_ck = {k:v for k,v in checkpoint['state_dict'].items() if 'init_pretrain_state' not in k}
    missed_keys = model.load_state_dict(new_ck, strict=True)
    print('Some items are missed or mismatched, this is not allowed when resuming:', missed_keys)

######## init bias #######    
#model = init_bias(model)    
SNN = model.cuda() if torch.cuda.is_available() else model

######## show parameters #######
# n_parameters = sum(p.numel() for p in SNN.parameters() if p.requires_grad)
# print('number of params:', n_parameters)
print(SNN)

######## parallel #######

if torch.cuda.device_count() > 1 and args.dataset == 'ImageNet':
    device = torch.device(local_rank)
    SNN = torch.nn.SyncBatchNorm.convert_sync_batchnorm(SNN)
    SNN = torch.nn.parallel.DistributedDataParallel(SNN, device_ids=[[local_rank]],output_device=[local_rank], find_unused_parameters=False if 'relu' in args.modeltag.lower() else True)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######## amp #######
loss_fun = torch.nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else torch.nn.CrossEntropyLoss()
if args.amp:
    scaler = torch.GradScaler("cuda")
else:
    scaler = None

######## split BN #######

spike_scales, feat_scales, feat_bits, weigth_scales, weight_bits, weights, other_no_decay, other_params, time_steps, feat_shifts, positions = classify_params(SNN)



#making bits not learnable
# for para in weight_bits:
#     para.requires_grad = False

# for para in feat_bits:
#     para.requires_grad = False

if 'shd' in args.arch:
    optimer = []
    optimer.append(torch.optim.Adam([{'params':weights, 'lr':args.lr, 'weight_decay':1e-5},
                                        {'params':other_params},
                                        {'params':other_no_decay, 'lr':args.lr, 'weight_decay':0},

                                        {'params':spike_scales, 'weight_decay':0},#irrelevant
    
                                        {'params':feat_scales,'lr':args.lr_q_feature_scale,'weight_decay':0},
                                        {'params':feat_bits,'lr':args.lr_q_feature_bit,'weight_decay':0},
                                        {'params':feat_shifts,'lr':args.lr_q_feature_shift,'weight_decay':0},

                                        {'params':weigth_scales,'lr':args.lr_q_weight_scale,'weight_decay':0},
                                        {'params':weight_bits,'lr':args.lr_q_weight_bit,'weight_decay':0},
                                     
                                        {'params':time_steps,'lr':args.lr_q_time_step,'weight_decay':0},
                                    ], lr=args.lr))
    if 'delay' in args.arch:
        optimer.append(torch.optim.Adam(positions, lr = 100*args.lr, weight_decay=0))
    else:
        optimer = optimer[0]

else:
    optimer = torch.optim.SGD([{'params':spike_scales, 'weight_decay':0},#irrelevant
                            
                                {'params':weights}, 
                                {'params':other_params},
                                {'params':other_no_decay, 'weight_decay':0},
                                
                                
                                {'params':feat_scales,'lr':args.lr_q_feature_scale,'weight_decay':0},
                                {'params':feat_bits,'lr':args.lr_q_feature_bit,'weight_decay':0},
                                {'params':feat_shifts,'lr':args.lr_q_feature_shift,'weight_decay':0},
                                
                                
                                {'params':weigth_scales,'lr':args.lr_q_weight_scale,'weight_decay':0},
                                {'params':weight_bits,'lr':args.lr_q_weight_bit,'weight_decay':0},
                                
                                {'params':time_steps,'lr':args.lr_q_time_step,'weight_decay':0},
                                ],
                            lr=args.lr, momentum=0.9, weight_decay=1e-4) 



#optimer = torch.optim.AdamW(params=parameters, lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-3) 
#optimer = torch.optim.AdamW(params=SNN.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-3)


if 'shd' in args.arch:
    if 'delay' in args.arch:
        scheduler = []
        scheduler.append(torch.optim.lr_scheduler.OneCycleLR(optimer[0], max_lr=5 * args.lr, total_steps=args.epochs))
        scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimer[1], T_max=args.epochs))
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimer, max_lr=5 * args.lr, total_steps=args.epochs)
else:
    scheduler = CosineAnnealingLR(optimer, T_max=args.epochs, eta_min=0)
scheduler_warm = None
if args.warm_up:
    scheduler_warm = GradualWarmupScheduler(optimer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
writer = None

# ------------------------------
###data
if 'dvscifar' == args.dataset.lower():
    train_data, test_data = dvs_load_data(dataset_dir=args.datapath, distributed=False, T=args.T, args=args)
elif 'shd' == args.dataset.lower():
    train_data, test_data = SHD_dataloaders(data_path=args.datapath, batch_size=args.batch, args=args)
elif 'binned_shd' == args.dataset.lower():
    train_data, test_data = binned_SHD_dataloaders(data_path=args.datapath, batch_size=args.batch, args=args)
elif 'cifar' in args.dataset.lower():
    train_data, test_data = build_data(cutout=True, use_cifar10=use_cifar10, auto_aug=True, batch_size=args.batch, datapath=args.datapath)
elif 'imagenet' in args.dataset.lower():
    train_data, test_data = build_imagenet(datapath=args.datapath, batch=args.batch, num_gpu=torch.cuda.device_count())
elif 'dvsgesture' == args.dataset.lower():
    train_data, test_data = dvs128_load_data(dataset_dir=args.datapath, distributed=False, T=args.T, args=args)
    


##mics && resume 
best = {'acc': 0., 'epoch': -1}
if args.resume is not None:
    #resume
    # 'state_dict', 'best_epoch', 'best_acc', "optimizer", 
    # 'scheduler', 'scaler', 'q_info',
    checkpoint = torch.load(args.resume, map_location='cpu')
    if 'shd' in args.arch and 'delay' in args.arch:
        optimer[0].load_state_dict(checkpoint['optimizer'])
        optimer[1].load_state_dict(checkpoint['optimizer_pos'])
    else:
        optimer.load_state_dict(checkpoint['optimizer'])
    if args.warm_up:
        scheduler_warm.load_state_dict(checkpoint['scheduler'])
    else:
        if 'shd' in args.arch and 'delay' in args.arch:
            scheduler[0].load_state_dict(checkpoint['scheduler'])
            scheduler[1].load_state_dict(checkpoint['scheduler_pos'])
        else:
            scheduler.load_state_dict(checkpoint['scheduler'])
    if args.amp:
        scaler.load_state_dict(checkpoint['scaler'])
    best = {'acc': checkpoint['best_acc'], 'epoch': checkpoint['best_epoch']}


def switch_renew(model, stop_mode='act', switch=False):
    if torch.cuda.device_count() > 1 and args.dataset == 'ImageNet':
        modelm = model.module
    else:
        modelm = model
    for module_name, module in modelm.named_modules():
        if hasattr(module, 'act_renew_switch') and (stop_mode == 'act' or stop_mode == 'both'):
            setattr(module, 'act_renew_switch', switch)
            print('found '+module_name+ '||act renew switch', 'switching to', switch)
        if hasattr(module, 'weight_renew_switch') and (stop_mode == 'weight' or stop_mode == 'both'):
            setattr(module, 'weight_renew_switch', switch)
            print('found '+module_name+ '||weight renew switch', 'switching to', switch)
                    
            

def save_checkpoint(state, epoch, tag=''):
    if not os.path.exists("./logs_q/models"):
        os.makedirs("./logs_q/models")
    filename = os.path.join(
        "./logs_q_new/models/{}-checkpoint-{:06}.pth".format(tag, epoch))
    torch.save(state, filename)

save_ck_func = partial(save_checkpoint, tag = args.modeltag)

    

def train(args, model, device, train_loader, optimizer, epoch, criterion, feat_bits, weight_bits, scaler=None):
    Top1, Top5 = 0.0, 0.0
    model.train()
    t1 = time.time()
    for batch_idx, packed in enumerate(train_loader):
        if 'shd' in args.dataset.lower():
            data, target, _ = packed[0], packed[1], packed[2]
            data, target = data.to(device, non_blocking=True, dtype=torch.float32), target.to(device, non_blocking=True)
  
        else:
            data, target = packed[0], packed[1]
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        if 'shd' in args.arch and 'delay' in args.arch:
            for opt in optimizer: opt.zero_grad()
        else:
            optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast("cuda"):
                output = model(data, is_drop=False)
                if args.use_bit_loss:
                    target_weight_bit, target_spike_bit, target_spike_step = args.weight_bit_tar, args.spike_bit_tar, args.spike_step_tar
                    if torch.cuda.device_count() > 1 and args.dataset.lower() == 'imagenet':
                        loss_weight_bit, loss_fea_bit, loss_spike_step = model.module.get_bit_loss(target_weight_bit=target_weight_bit, target_spike_bit=target_spike_bit, target_spike_step=target_spike_step,
                                                                        weight_bit_penalty=args.weight_bit_penalty, spike_bit_penalty=args.spike_bit_penalty, spike_step_penalty=args.spike_step_penalty)
                    else:
                        loss_weight_bit, loss_fea_bit, loss_spike_step = model.get_bit_loss(target_weight_bit=target_weight_bit, target_spike_bit=target_spike_bit, target_spike_step=target_spike_step,
                                                                        weight_bit_penalty=args.weight_bit_penalty, spike_bit_penalty=args.spike_bit_penalty, spike_step_penalty=args.spike_step_penalty)
                    q_loss = loss_fea_bit + loss_weight_bit + loss_spike_step
                else:
                    loss_fea_bit, loss_weight_bit, loss_spike_step = torch.tensor([-6.6]), torch.tensor([-6.6]), torch.tensor([-6.6])
                    q_loss = 0.
                if args.use_tet:
                    loss = TET_loss(output, target, mode="TB") + q_loss
                else:
                    loss = criterion(output, target) + q_loss
            
            
            scaler.scale(loss).backward()
            if 'shd' in args.arch and 'delay' in args.arch:
                scaler.step(optimizer[0])
                scaler.step(optimizer[1])
            else:
                scaler.step(optimizer)
            scaler.update()
        
        else:
            output = model(data,is_drop=False)
            loss_fea_bit, loss_weight_bit = torch.tensor([-6.6]), torch.tensor([-6.6])
            if args.use_bit_loss:
                target_weight_bit, target_spike_bit, target_spike_step = args.weight_bit_tar, args.spike_bit_tar, args.spike_step_tar
                if torch.cuda.device_count() > 1 and args.dataset.lower() == 'imagenet':
                    loss_weight_bit, loss_fea_bit, loss_spike_step = model.module.get_bit_loss(target_weight_bit=target_weight_bit, target_spike_bit=target_spike_bit, target_spike_step=target_spike_step,
                                                                    weight_bit_penalty=args.weight_bit_penalty, spike_bit_penalty=args.spike_bit_penalty, spike_step_penalty=args.spike_step_penalty)
                else:
                    loss_weight_bit, loss_fea_bit, loss_spike_step = model.get_bit_loss(target_weight_bit=target_weight_bit, target_spike_bit=target_spike_bit, target_spike_step=target_spike_step,
                                                                    weight_bit_penalty=args.weight_bit_penalty, spike_bit_penalty=args.spike_bit_penalty, spike_step_penalty=args.spike_step_penalty)

                q_loss = loss_fea_bit + loss_weight_bit + loss_spike_step
                # q_loss.backward(retain_graph=True)
            else:
                loss_fea_bit, loss_weight_bit, loss_spike_step = torch.tensor([-6.6]), torch.tensor([-6.6]), torch.tensor([-6.6])
                q_loss = 0.
            loss = criterion(output, target) + q_loss
                
            loss.backward()
            if 'shd' in args.arch and 'delay' in args.arch:
                for opt in optimizer: opt.step()
            else:
                optimizer.step()
        
        functional.reset_net(model)
        if 'shd' in args.arch and 'delay' in args.arch:
            model.model.dcls_clamp()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        Top1 += prec1.item() / 100
        Top5 += prec5.item() / 100
        
        display_interval = 20
        if torch.any(torch.isnan(loss.detach())):
            spike_bit_sum, weight_bit_sum, spike_step_sum = model.module.get_bit_weighted_mean() if torch.cuda.device_count() > 1 and args.dataset == 'ImageNet' else model.get_bit_weighted_mean()
            print('spike_bit_sum:{},\tweight_bit_sum:{},\tspike_step_sum:{}'.format(spike_bit_sum.item(), weight_bit_sum.item(), spike_step_sum.item()))
            print('saving wrong ck to ... ./Wrong_cks')
            wrong_model_ck=model.module.state_dict() if torch.cuda.device_count() > 1 and args.dataset == 'ImageNet' else model.state_dict(),
            bits_scales = model.module.get_bits_scales() if torch.cuda.device_count() > 1 and args.dataset == 'ImageNet' else model.get_bits_scales()
            if args.spiking_mode:
                print('spike_bits')
                spike_bits_np = torch.cat(bits_scales['spike_bits'], dim=0).numpy()
                print(spike_bits_np.max())
                print(spike_bits_np.min())
                print(spike_bits_np.mean())
                print('spike_scales')
                spike_scales_np = torch.cat(bits_scales['spike_scales'], dim=0).numpy()
                print(spike_scales_np.max())
                print(spike_scales_np.min())
                print(spike_scales_np.mean())
                print('weight_bits')
                weight_bits_np = torch.cat(bits_scales['weight_bits'], dim=0).numpy()
                print(weight_bits_np.max())
                print(weight_bits_np.min())
                print(weight_bits_np.mean())
                print('weight_scales')
                weight_scales_np = torch.cat(bits_scales['weight_scales'], dim=0).numpy()
                print(weight_scales_np.max())
                print(weight_scales_np.min())
                print(weight_scales_np.mean())
                print('time steps')
                time_steps_np = torch.cat(bits_scales['time_steps'], dim=0).numpy()
                print(time_steps_np.max())
                print(time_steps_np.min())
                print(time_steps_np.mean())
            
            
            torch.save(wrong_model_ck, './Wrong_cks/{}-epoch{}.pth'.format(args.modeltag, epoch))
            assert not torch.isnan(loss.detach())
            assert not torch.isnan(loss_fea_bit.detach())
            assert not torch.isnan(loss_weight_bit.detach())
            assert not torch.isnan(loss_spike_step.detach())
                
        if batch_idx % display_interval == 0 and args.learn_time_step:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tloss_spike_bit: {:.6f}\tloss_weight_bit: {:.6f}\tloss_spike_step: {:.6f}\tTop-1 = {:.6f}\tTop-5 = {:.6f}\tTime = {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(), loss_fea_bit.item(), loss_weight_bit.item(), loss_spike_step.item(),
                           Top1 / display_interval, Top5 / display_interval, time.time() - t1
                )
            )
        elif batch_idx % display_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tloss_fea_bit: {:.6f}\tloss_weight_bit: {:.6f}\tTop-1 = {:.6f}\tTop-5 = {:.6f}\tTime = {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(), loss_fea_bit.item(), loss_weight_bit.item(),
                           Top1 / display_interval, Top5 / display_interval, time.time() - t1
                )
            )
            Top1, Top5 = 0.0, 0.0
        
    print('time used in this epoch:{}'.format(time.time() - t1))


def test(args, model, device, test_loader, epoch, criterion, best= best, optimizer=optimer, scheduler=scheduler, scaler=scaler):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()# inactivate BN
    # print('start testing')
    t1 = time.time()
    with torch.no_grad():
        for packed in test_loader:
            if 'shd' in args.dataset.lower():
                data, target, _ = packed[0], packed[1], packed[2]
                data, target = data.to(device, non_blocking=True, dtype=torch.float32), target.to(device, non_blocking=True)
            else:
                data, target = packed[0], packed[1]
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data, is_drop=False)
            # else:
            #     output = model(data, teas)
            loss = criterion(output, target)
            functional.reset_net(model)
            if 'shd' in args.arch and 'delay' in args.arch:
                model.model.dcls_clamp()
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        print('TEST Epoch {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
                  'Top-1  = {:.6f},\t'.format(top1.avg / 100) + \
                  'Top-5  = {:.6f},\t'.format(top5.avg / 100) + \
                  'val_time = {:.6f}\n'.format(time.time() - t1))
     

        w_b, s_b, t_b, bb = None, None, None, None
        if best is not None:
            if top1.avg / 100 > best['acc']:
                best['acc'], best['epoch'] = top1.avg / 100, epoch
                print('saving...')
                save_ck_func({
                    'state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 and args.dataset == 'ImageNet' else model.state_dict(),
                    'best_epoch': best['epoch'],
                    'best_acc': best['acc'],
                    "optimizer": optimizer[0].state_dict() if 'shd' in args.arch and 'delay' in args.arch else optimizer.state_dict(),
                    "optimizer_pos": optimizer[1].state_dict() if 'shd' in args.arch and 'delay' in args.arch else None,
                    'scheduler': scheduler[0].state_dict() if 'shd' in args.arch and 'delay' in args.arch else scheduler.state_dict(),  
                    'scheduler_pos': scheduler[1].state_dict() if 'shd' in args.arch and 'delay' in args.arch else None,      
                    'scaler':scaler.state_dict() if scaler is not None else None,
                    'q_info': None if 'relu' in args.modeltag.lower() else model.module.q_info if torch.cuda.device_count() > 1 and args.dataset == 'ImageNet' else model.q_info,
                    'epoch':epoch
                    }, -1)#"./logs/models"

            print('best acc is {} found in epoch {}'.format(best['acc'], best['epoch']))
            if 'relu' not in args.modeltag.lower():
                bits_scales = model.module.get_bits_scales() if torch.cuda.device_count() > 1 and args.dataset == 'ImageNet' else model.get_bits_scales()
                if args.spiking_mode and args.learn_weight_bit and args.learn_spike_bit:
                    print('spike_bits')
                    spike_bits_np = torch.cat(bits_scales['spike_bits'], dim=0).numpy()
                    print(spike_bits_np.max())
                    print(spike_bits_np.min())
                    print(spike_bits_np.mean())
                    print('spike_scales')
                    spike_scales_np = torch.cat(bits_scales['spike_scales'], dim=0).numpy()
                    print(spike_scales_np.max())
                    print(spike_scales_np.min())
                    print(spike_scales_np.mean())
                    print('weight_bits')
                    weight_bits_np = torch.cat(bits_scales['weight_bits'], dim=0).numpy()
                    print(weight_bits_np.max())
                    print(weight_bits_np.min())
                    print(weight_bits_np.mean())
                    print('weight_scales')
                    weight_scales_np = torch.cat(bits_scales['weight_scales'], dim=0).numpy()
                    print(weight_scales_np.max())
                    print(weight_scales_np.min())
                    print(weight_scales_np.mean())
                    print('time steps')
                    time_steps_np = torch.cat(bits_scales['time_steps'], dim=0).numpy()
                    print(time_steps_np.max())
                    print(time_steps_np.min())
                    print(time_steps_np.mean())
                    spike_bit_sum, weight_bit_sum, spike_step_sum = model.module.get_bit_weighted_mean() if torch.cuda.device_count() > 1 and args.dataset == 'ImageNet' else model.get_bit_weighted_mean()
                    
                    print('weight_bit_sum:{},\tspike_bit_sum:{},\tspike_step_sum:{}'.format(weight_bit_sum.item(), spike_bit_sum.item(), spike_step_sum.item()))
                    print('w' + "{}".format(round(weight_bit_sum.item(), 2)) + '  a' + "{}".format(round(spike_bit_sum.item(), 2)) + '  t' + "{}".format(round(spike_step_sum.item(), 2)) + '  bb' + "{}".format(round((weight_bit_sum*spike_bit_sum*spike_step_sum).item(), 2)))
                    w_b, s_b, t_b, bb = round(weight_bit_sum.item(), 3), round(spike_bit_sum.item(), 3), round(spike_step_sum.item(), 3), round((weight_bit_sum*spike_bit_sum*spike_step_sum).item(), 3)
                elif args.learn_weight_bit and args.learn_spike_bit:
                    print('feat_bits')
                    # feat_bits_np = np.array([list(i.values()) for i in bits_scales['feat_bits']]) # per layer
                    feat_bits_np = np.concatenate([np.array(list(i.values())).flatten() for i in bits_scales['feat_bits']], axis=-1) # per channel
                    print(feat_bits_np.max())
                    print(feat_bits_np.min())
                    print(feat_bits_np.mean())
                    print('feat_scales')
                    # feat_scale_np = np.array([list(i.values()) for i in bits_scales['feat_scales']]) # per layer
                    feat_scale_np = np.concatenate([np.array(list(i.values())).flatten() for i in bits_scales['feat_scales']], axis=-1) # per channel
                    print(feat_scale_np.max())
                    print(feat_scale_np.min())
                    print(feat_scale_np.mean())
                    print('--'*20)
                    print('weight_bits')
                    weight_bits_np = np.concatenate([np.array(list(i.values())).flatten() for i in bits_scales['weight_bits']], axis=-1)
                    print(weight_bits_np.max())
                    print(weight_bits_np.min())
                    print(weight_bits_np.mean())
                    weight_scale_np = np.concatenate([np.array(list(i.values())).flatten() for i in bits_scales['weight_scales']], axis=-1)
                    print('weight_scales')
                    print(weight_scale_np.max())
                    print(weight_scale_np.min())
                    print(weight_scale_np.mean())
            
            if epoch % 9 == 0 and args.dataset == 'ImageNet':
                print('saving...')
                save_ck_func({
                    'state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 and args.dataset == 'ImageNet' else model.state_dict(),
                    'best_epoch': best['epoch'],
                    'best_acc': best['acc'],
                    "optimizer": optimizer[0].state_dict() if 'shd' in args.arch and 'delay' in args.arch else optimizer.state_dict(),
                    "optimizer_pos": optimizer[1].state_dict() if 'shd' in args.arch and 'delay' in args.arch else None,
                    'scheduler': scheduler[0].state_dict() if 'shd' in args.arch and 'delay' in args.arch else scheduler.state_dict(),  
                    'scheduler_pos': scheduler[1].state_dict() if 'shd' in args.arch and 'delay' in args.arch else None,      
                    'scaler':scaler.state_dict() if scaler is not None else None,   
                    'q_info': None if 'relu' in args.modeltag.lower() else model.module.q_info if torch.cuda.device_count() > 1 and args.dataset == 'ImageNet' else model.q_info,
                    'epoch':epoch
                    }, epoch)  # "./logs/models"
    return w_b, s_b, t_b, bb
    

if __name__ == '__main__':
    import sys
    if args.resume is not None:
        epoch = checkpoint['epoch'] + 1
    else:
        epoch = 0
    
    # print('local rank', local_rank)
    # if local_rank > 1:
    #     sys.stdout = open(os.devnull, 'w')
    w_stop_renew_flag, s_stop_renew_flag = False, False
    adaptive_w_stop_e, adaptive_s_stop_e = None, None
    
    if args.resume is not None or args.pretrained is not None:
        print('initial test right after loading pretrained or resumed model')
        w_b_, s_b_, t_b_, bb_ = test(args, model=SNN, device=device, test_loader=test_data, epoch=epoch, criterion=loss_fun, best= best, optimizer=optimer, scheduler=scheduler_warm if args.warm_up else scheduler, scaler=scaler)
        if args.test_only:
            exit()
        print('start real training')
    while (epoch < args.epochs):
        train(args, model=SNN, device=device, train_loader=train_data, optimizer=optimer, epoch=epoch, criterion=loss_fun, feat_bits=feat_bits, weight_bits=weight_bits, scaler=scaler)
        # if epochs % 1 == 0:
        # if 'relu' not in args.modeltag or local_rank == 0:
            # print('start testing')
        w_b, s_b, t_b, bb = test(args, model=SNN, device=device, test_loader=test_data, epoch=epoch, criterion=loss_fun, best= best, optimizer=optimer, scheduler=scheduler_warm if args.warm_up else scheduler, scaler=scaler)
        if args.warm_up:
            scheduler_warm.step()
        else:
            if 'shd' in args.arch and 'delay' in args.arch:
                for sche in scheduler: sche.step()
                SNN.model.decrease_sig(epoch)
            else:
                scheduler.step()
            
        if args.renew_switch_epoch>0:
            if epoch == args.renew_switch_epoch:
                print('shutting renew swiching')
                switch_renew(SNN, stop_mode='both', switch=False)
        
        elif args.renew_switch_epoch==0:
            print('shutting renew swiching from beginning')
            switch_renew(SNN, stop_mode='both', switch=False)
            
        elif args.renew_switch_epoch<0: #adaptive stop renew
            
            stop_ratio = abs(float(args.renew_switch_epoch / 100))
            
            tar_w_b, tar_s_b, tar_t_b = args.weight_bit_tar, args.spike_bit_tar, args.spike_step_tar
            tar_bb = tar_w_b * tar_s_b * tar_t_b
            
            init_w_b, init_s_b, init_t_b = args.init_weight_bit, args.init_spike_bit, args.init_spike_step
            init_bb = init_w_b * init_s_b * init_t_b
            
            init_diff_w_b, init_diff_s_b = (init_w_b - tar_w_b), (init_s_b - tar_s_b)
            diff_w_b, diff_s_b = (w_b - tar_w_b), (s_b - tar_s_b)
            threshold_w_b, threshold_s_b = init_diff_w_b * stop_ratio, init_diff_s_b * stop_ratio
            
            if (diff_w_b < threshold_w_b and (not w_stop_renew_flag) and not epoch<args.renew_switch_least_epoch) or threshold_w_b==0 :
                w_stop_renew_flag = True
                print('weight bit renew threshold: {}'.format(threshold_w_b))
                print('weight bit renew stopped')
                switch_renew(SNN, stop_mode='weight', switch=False)
                adaptive_w_stop_e = epoch
             
            if (diff_s_b < threshold_s_b and (not s_stop_renew_flag) and not epoch<args.renew_switch_least_epoch) or threshold_s_b==0:
                s_stop_renew_flag = True
                print('spike bit renew threshold: {}'.format(threshold_s_b))
                print('spike bit renew stopped')
                switch_renew(SNN, stop_mode='act', switch=False)
                adaptive_s_stop_e = epoch
        epoch += 1
    
    print('adaptive_w_stop_e: {}'.format(adaptive_w_stop_e) + ', adaptive_s_stop_e: {}'.format(adaptive_s_stop_e))