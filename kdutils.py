



import torch
import torchaudio
import torchvision
import os
import random
import numpy as np
import sys
import torch.nn as nn
from spikingjelly.datasets.shd import SpikingHeidelbergDigits
from spikingjelly.datasets import pad_sequence_collate
from typing import Callable, Optional
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

import torchvision.transforms as transforms
from data import CIFAR10Policy, Cutout
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import _LRScheduler

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
            

def build_data(batch_size=128, cutout=False, workers=4, use_cifar10=False, auto_aug=False, datapath=None):

    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if auto_aug:
        aug.append(CIFAR10Policy())

    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root=datapath,
                                train=True, download=True, transform=transform_train)
        val_dataset = CIFAR10(root=datapath,
                              train=False, download=True, transform=transform_test)

    else:   #cifar100
        # mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        # std = [x / 255 for x in [68.2, 65.4, 70.4]]
        
        
        # transform_train = transforms.Compose([
        #                      transforms.RandomCrop(32, padding=4),
        #                      transforms.RandomHorizontalFlip(),
        #                      transforms.ToTensor(),
        #                      transforms.Normalize(mean=mean, std=std)
        #                  ])
        # transform_test = transforms.Compose([
        #                     transforms.ToTensor(),
        #                     transforms.Normalize(mean=mean, std=std)
        #                 ])
        
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        
        train_dataset = CIFAR100(root=datapath,
                                 train=True, download=True, transform=transform_train)
        val_dataset = CIFAR100(root=datapath,
                               train=False, download=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=True)



    return train_loader, val_loader



def build_imagenet(datapath, batch, num_gpu):
    traindir = os.path.join(datapath, 'train')
    valdir =os.path.join(datapath, 'val')
    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]))

    val_dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]))

    train_sampler = DistributedSampler(train_dataset) if num_gpu > 1 else None
    # val_sampler = DistributedSampler(val_dataset)
    
    train_loader = DataLoader(
                dataset=train_dataset,
                sampler=train_sampler,
                batch_size=batch,
                shuffle=(train_sampler is None),
                num_workers=6*num_gpu,
                drop_last=True,
                pin_memory=True)
    val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=batch,
                shuffle=False,
                num_workers=6*num_gpu,
                drop_last=False,
                pin_memory=True)
    
    return train_loader, val_loader



### dvs_data loader
import time
from spikingjelly.datasets import cifar10_dvs
import math

def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


class NumpyToTensor(nn.Module):
    def forward(self, frames):
        assert isinstance(frames, np.ndarray)
        return torch.tensor(frames, requires_grad=False)





def dvs_load_data(dataset_dir, distributed, T, args):
    # Data loading code
    print("Loading data")

    st = time.time()

    # dataset_train, dataset_test = split_to_train_test_set(0.9, origin_set, 10)
    
    #quick load 
    cache_dir = os.path.join(dataset_dir, 'cifar10dvs_cache_spfmer')
    cache_dir = os.path.join(cache_dir, f'frame_T_{T}_spikformer')    
    train_set_pth = os.path.join(cache_dir, f'train_set_{T}.pt')
    test_set_pth = os.path.join(cache_dir, f'test_set_{T}.pt')
    print(cache_dir)
    if os.path.exists(train_set_pth) and os.path.exists(test_set_pth):
        dataset_train = torch.load(train_set_pth)
        dataset_test = torch.load(test_set_pth)
    else:
        origin_set = cifar10_dvs.CIFAR10DVS(root=dataset_dir, data_type='frame', frames_number=T, split_by='number')# origin
        # origin_set = cifar10_dvs.CIFAR10DVS(root=dataset_dir, data_type='frame', frames_number=T, split_by='number')#mine
        dataset_train, dataset_test = split_to_train_test_set(0.9, origin_set, 10)  # origin
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        torch.save(dataset_train, train_set_pth)
        torch.save(dataset_test, test_set_pth)
    print("Took", time.time() - st)
    
    
    
    if args.dsr_da:
        dataset_train.dataset.transform = transforms.Compose([
            NumpyToTensor(),
        transforms.Resize([48, 48], antialias=True),
        transforms.RandomCrop(48, padding=4),
        ])

        dataset_test.dataset.transform = transforms.Compose([
            NumpyToTensor(),
        transforms.Resize([48, 48], antialias=True),
                                                            ])
        
    elif args.dsr_da128:
        dataset_train.dataset.transform = transforms.Compose([
            NumpyToTensor(),
        transforms.Resize([128, 128], antialias=True),
        transforms.RandomCrop(128, padding=4),
        ])

        dataset_test.dataset.transform = transforms.Compose([
            NumpyToTensor(),
        transforms.Resize([128, 128], antialias=True),
                                                            ])
    
    elif args.nda_da or args.nda_da128:
        print('using nda data aug.')
        class nda_da(torch.nn.Module):
            def __init__(self, train=True):
                super().__init__()
                self.resize = transforms.Resize(size=(48, 48), interpolation=torchvision.transforms.InterpolationMode.NEAREST, antialias=True)
                self.rotate = transforms.RandomRotation(degrees=30)
                self.shearx = transforms.RandomAffine(degrees=0, shear=(-30, 30))
                self.training = train
            def forward(self, frames):
                # print(frames.shape)   time * channel * hight * width
                if not args.nda_da128:
                    data = self.resize(frames)
                else:
                    data = frames 
                if self.training:
                    choices = ['roll', 'rotate', 'shear']
                    aug = np.random.choice(choices)
                    if aug == 'roll':
                        off1 = random.randint(-5, 5)
                        off2 = random.randint(-5, 5)
                        data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
                    if aug == 'rotate':
                        data = self.rotate(data)
                    if aug == 'shear':
                        data = self.shearx(data)
                return data
        dataset_train.dataset.transform = transforms.Compose([
            NumpyToTensor(),
            nda_da(train=True),
        ])

        dataset_test.dataset.transform = transforms.Compose([
            NumpyToTensor(),
            nda_da(train=False),
                                                            ])

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True)
        
    return data_loader, data_loader_test


class BinnedSpikingHeidelbergDigits(SpikingHeidelbergDigits):
    def __init__(
            self,
            root: str,
            n_bins: int,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """
        The Spiking Heidelberg Digits (SHD) dataset, which is proposed by `The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks <https://doi.org/10.1109/TNNLS.2020.3044364>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.

        .. admonition:: Note
            :class: note

            Events in this dataset are in the format of ``(x, t)`` rather than ``(x, y, t, p)``. Thus, this dataset is not inherited from :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` directly. But their procedures are similar.

        :class:`spikingjelly.datasets.shd.custom_integrate_function_example` is an example of ``custom_integrate_function``, which is similar to the cunstom function for DVS Gesture in the ``Neuromorphic Datasets Processing`` tutorial.
        """
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
        self.n_bins = n_bins

    def __getitem__(self, i: int):
        if self.data_type == 'event':
            events = {'t': self.h5_file['spikes']['times'][i], 'x': self.h5_file['spikes']['units'][i]}
            label = self.h5_file['labels'][i]
            if self.transform is not None:
                events = self.transform(events)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return events, label

        elif self.data_type == 'frame':
            frames = np.load(self.frames_path[i], allow_pickle=True)['frames'].astype(np.float32)
            label = self.frames_label[i]

            binned_len = frames.shape[1]//self.n_bins
            binned_frames = np.zeros((frames.shape[0], binned_len))
            for i in range(binned_len):
                binned_frames[:,i] = frames[:, self.n_bins*i : self.n_bins*(i+1)].sum(axis=1)

            if self.transform is not None:
                binned_frames = self.transform(binned_frames)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return binned_frames, label



def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # This flag only allows cudnn algorithms that are determinestic unlike .benchmark
    torch.backends.cudnn.deterministic = True

    #this flag enables cudnn for some operations such as conv layers and RNNs, 
    # which can yield a significant speedup.
    torch.backends.cudnn.enabled = False

    # This flag enables the cudnn auto-tuner that finds the best algorithm to use
    # for a particular configuration. (this mode is good whenever input sizes do not vary)
    torch.backends.cudnn.benchmark = False

def binned_SHD_dataloaders(data_path, batch_size, args):
    # train_dataset = BinnedSpikingHeidelbergDigits(data_path, 5, train=True, data_type='frame', frames_number=10, split_by = 'number')
    # test_dataset= BinnedSpikingHeidelbergDigits(data_path, 5, train=False, data_type='frame', frames_number=10, split_by = 'number')
    
    train_dataset = BinnedSpikingHeidelbergDigits(data_path, 5, train=True, data_type='frame', duration=args.init_spike_step)
    test_dataset= BinnedSpikingHeidelbergDigits(data_path, 5, train=False, data_type='frame', duration=args.init_spike_step)
    train_loader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=batch_size, num_workers=4)
    return train_loader, test_loader



def SHD_dataloaders(data_path, batch_size, args):
    # set_seed(0)
    train_dataset = SpikingHeidelbergDigits(data_path, train=True, data_type='frame', frames_number=10, split_by = 'number')
    test_dataset= SpikingHeidelbergDigits(data_path, train=False, data_type='frame', frames_number=10, split_by = 'number')
    
    train_loader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=batch_size, num_workers=4)
    return train_loader, test_loader



def dvs128_load_data(dataset_dir, distributed, T, args):
    # Data loading code
    print("Loading data")

    st = time.time()    
    train_transform, test_transform= None, None
    if args.dsr_da:
        train_transform = transforms.Compose([
            NumpyToTensor(),
        transforms.Resize([48, 48], antialias=True),
        transforms.RandomCrop(48, padding=4),
        ])

        test_transform = transforms.Compose([
            NumpyToTensor(),
        transforms.Resize([48, 48], antialias=True),
                                                            ])
        
    elif args.dsr_da128:
        train_transform = transforms.Compose([
            NumpyToTensor(),
        transforms.Resize([128, 128], antialias=True),
        transforms.RandomCrop(128, padding=4),
        ])

        test_transform = transforms.Compose([
            NumpyToTensor(),
        transforms.Resize([128, 128], antialias=True),
                                                            ])
    
    elif args.nda_da or args.nda_da128 or args.sd_da:
        print('using nda data aug.')
        class nda_da(torch.nn.Module):
            def __init__(self, train=True):
                super().__init__()
                if args.sd_da:
                    imgz = 64
                else:
                    imgz = 48
                self.resize = transforms.Resize(size=(imgz, imgz), interpolation=torchvision.transforms.InterpolationMode.NEAREST, antialias=True)
                self.rotate = transforms.RandomRotation(degrees=30)
                self.shearx = transforms.RandomAffine(degrees=0, shear=(-30, 30))
                self.training = train
            def forward(self, frames):
                # print(frames.shape)   time * channel * hight * width
                if not args.nda_da128:
                    data = self.resize(frames)
                else:
                    data = frames 
                if self.training:
                    choices = ['roll', 'rotate', 'shear']
                    aug = np.random.choice(choices)
                    if aug == 'roll':
                        off1 = random.randint(-5, 5)
                        off2 = random.randint(-5, 5)
                        data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
                    if aug == 'rotate':
                        data = self.rotate(data)
                    if aug == 'shear':
                        data = self.shearx(data)
                return data
            
            
        train_transform = transforms.Compose([
            NumpyToTensor(),
            nda_da(train=True),
        ])

        test_transform = transforms.Compose([
            NumpyToTensor(),
            nda_da(train=False),
                                                            ])


    dataset_train = DVS128Gesture(
        dataset_dir,
        train=True,
        data_type="frame",
        frames_number=T,
        split_by="number",
        transform=train_transform,
    )
    
    dataset_test = DVS128Gesture(
        dataset_dir,
        train=False,
        data_type="frame",
        frames_number=T,
        split_by="number",
        transform=test_transform,
    )
        
    print("Took", time.time() - st)


    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True
        )

    data_loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        pin_memory=True)
        

    return data_loader, data_loader_test

