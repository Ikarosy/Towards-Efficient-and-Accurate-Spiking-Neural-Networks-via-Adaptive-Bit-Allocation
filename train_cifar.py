import argparse
import os

#os.system('wandb login xxx')
#import wandb
import time
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
from kdutils import seed_all, GradualWarmupScheduler, build_data
#from torchvision.models.resnet import resnet18
from torchvision import transforms
from models import *
from models.resnet import SEWResNet34, ResNet18, ResNet18_cifar
#from models import ImageNet_cnn, ImageNet_snn
#from loss_kd import feature_loss,  logits_loss
#from spikingjelly.clock_driven import functional



parser = argparse.ArgumentParser(description="ImageNet_SNN_Training")

parser.add_argument("--datapath", type=str, default='/mnt/lustre/GPU8/home/yaoxingting/CIFAR10')
parser.add_argument('--arch', default='resnet20_cifar', type=str, help='dataset name',
                    choices=['resnet20_cifar', 'resnet19_cifar', 'resnet20_cifar_modified', 'ResNet18', 'ResNet34', 'resnet20_cifar_real', 'ResNet18_cifar'])
parser.add_argument('--modeltag', type=str, default='SNN', help='decide the name of the exp.')
parser.add_argument("--batch", type=int, default=40)
parser.add_argument("--epochs", type=int, default=320)
parser.add_argument('--lr', default=1e-1, type=float, help='initial learning rate')

parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name',
                    choices=['MNIST', 'CIFAR10', 'CIFAR100'])

parser.add_argument("--warm_up", action='store_true', default=False)
parser.add_argument("--beta", type=float, default=10.)


parser.add_argument('--spiking_mode', action='store_true', help='use spiking network')  
parser.add_argument('--step', default=2, type=int, help='snn step')
parser.add_argument('--seed', type=int, default=9, metavar='S',
                    help='random seed (default: 9)')



args = parser.parse_args()
seed_all(args.seed)

best_acc = 0.3
sta = time.time()

# ----------------------------
for k,v in sorted(vars(args).items()):
    print(k,'=',v)
    
######## input model #######
use_cifar10 = use_cifar10 = args.dataset == 'CIFAR10'

model = eval(args.arch)(num_classes= 10 if use_cifar10 else 100)

#no relu
def replace_relu(model):
    for name, child_module in model.named_children():
        if isinstance(child_module, (torch.nn.ReLU, torch.nn.ReLU6)):
            setattr(model, name, torch.nn.Identity())
        else:
            replace_relu(child_module)
# replace_relu(model)
######## save model #######
model_save_name = './logs/' + args.modeltag + '.pth'


######## load weight #######
#model.load_state_dict(torch.load('raw/ann-resnet18.pth', map_location='cpu'))

######## change to snn #######
if args.spiking_mode is True:
    model = SpikeModel(model, args.step)
    model.set_spike_state(True)
    
######## init bias #######    
#model = init_bias(model)    
SNN = model.cuda()

######## show parameters #######
n_parameters = sum(p.numel() for p in SNN.parameters() if p.requires_grad)
print('number of params:', n_parameters)
print(SNN)

######## parallel #######
model_without_ddp = SNN

######## amp #######
loss_fun = torch.nn.CrossEntropyLoss().cuda()
# scaler = torch.cuda.amp.GradScaler()

######## split BN #######

parameters = split_weights(SNN)
optimer = torch.optim.SGD(params=parameters, lr=args.lr, momentum=0.9, weight_decay=1e-4) 
#optimer = torch.optim.AdamW(params=parameters, lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-3) 
#optimer = torch.optim.AdamW(params=SNN.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=5e-3)

scheduler = CosineAnnealingLR(optimer, T_max=args.epochs, eta_min=0)
scheduler_warm = None
# if args.warm_up:
#     scheduler_warm = GradualWarmupScheduler(optimer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
writer = None

# ------------------------------
traindir = args.datapath + 'train'
valdir = args.datapath + 'val'

###data
train_data, test_data = build_data(cutout=True, use_cifar10=use_cifar10, auto_aug=True, batch_size=args.batch, datapath=args.datapath)

##mics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best = {'acc': 0., 'epoch': -1}
from utils import accuracy, AvgrageMeter
from functools import partial

def save_checkpoint(state, epoch, tag=''):
    if not os.path.exists("./logs/models"):
        os.makedirs("./logs/models")
    filename = os.path.join(
        "./logs/models/{}-checkpoint-{:06}.pth".format(tag, epoch))
    torch.save(state, filename)

save_ck_func = partial(save_checkpoint, tag = args.modeltag)


def train(args, model, device, train_loader, optimizer, epoch, criterion, scaler=None):
    Top1, Top5 = 0.0, 0.0
    model.train()
    t1 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(data,is_drop=False)
                loss = criterion(output, target)
        else:
            output = model(data,is_drop=False)
            loss = criterion(output, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        Top1 += prec1.item() / 100
        Top5 += prec5.item() / 100

        display_interval = 10
        if batch_idx % display_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTop-1 = {:.6f}\tTop-5 = {:.6f}\tTime = {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                           Top1 / display_interval, Top5 / display_interval, time.time() - t1
                )
            )
            Top1, Top5 = 0.0, 0.0
    print('time used in this epoch:{}'.format(time.time() - t1))


def test(args, model, device, test_loader, epoch, criterion, best= best):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()# inactivate BN
    t1 = time.time()
    with torch.no_grad():
        print('start testing')
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, is_drop=False)
            # else:
            #     output = model(data, teas)
            loss = criterion(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        print('TEST Epoch {}: loss = {:.6f},\t'.format(epoch, objs.avg) + \
                  'Top-1  = {:.6f},\t'.format(top1.avg / 100) + \
                  'Top-5  = {:.6f},\t'.format(top5.avg / 100) + \
                  'val_time = {:.6f}\n'.format(time.time() - t1))
     

        if best is not None:
            if top1.avg / 100 > best['acc']:
                best['acc'], best['epoch'] = top1.avg / 100, epoch
                print('saving...')
                save_ck_func({
                    'state_dict': model.state_dict(),
                    'best_epoch': best['epoch'],
                    'best_acc': best['acc'],
                    }, epoch)#"./logs/models"

            print('best acc is {} found in epoch {}'.format(best['acc'], best['epoch']))

            if epoch % 20 == 0:
                print('saving...')
                save_ck_func({
                    'state_dict': model.state_dict(),
                    'best_epoch': best['epoch'],
                    'best_acc': best['acc'],
                    }, epoch)  # "./logs/models"

if __name__ == '__main__':
    epoch = 0
    while (epoch < args.epochs):
        train(args, model=SNN, device=device, train_loader=train_data, optimizer=optimer, epoch=epoch, criterion=loss_fun, scaler=None)
        # if epochs % 1 == 0:
        if args.warm_up:
            scheduler_warm.step()
        else:
            scheduler.step()
        test(args, model=SNN, device=device, test_loader=test_data, epoch=epoch, criterion=loss_fun, best= best)
        epoch += 1