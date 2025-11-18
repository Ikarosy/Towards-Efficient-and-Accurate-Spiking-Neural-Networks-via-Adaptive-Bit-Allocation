
import subprocess
import random

# random_integer = random.randint(1, 9999)

def list2str(list):
    j = ''
    for i in list:
        if not i == '':
            j += (i + ' ')
    return j

imagenet_datapath = '/mnt/lustre/GPU8/share/ilsvrc12_raw'

# I leave this function here that models are able to be initialized to pretrained checkpoints
# In practice, I find pre-loading pretrained checkpoints is not helpful in improving the accuracies of quantized models.
# Just offering a play station. Play with my codes if you want.

cifar10_res20_pretrained = './pretrained_cks/SEWResNet34_data_aug_step2_seed3407_imagenet_bz64_ep320_using_relu_strict-checkpoint-000313.pth'
#exaple: nohup python -u autoRun_q_imagenet.py > /dev/null 2>&1 &


if __name__ == "__main__":
    

    
    #####pretraining running on gpu6
    # python -u autoRun_q_imagenet.py       
    gpu = '2,3'
    random_integer = int(gpu[0]*4) + 3267
    gpu_num = (len(gpu)+1)//2
    epoch = 200  
    batch_size = 128
    seed = 3407
    arch = 'ResNet18_q' #choice = [resnet20_cifar, resnet19_cifar, resnet20_cifar_modified, ResNet18, ResNet34, resnet20_cifar_q]
    
    learn_weight_bit = True
    learn_spike_bit = True
    learn_time_step = True
    
    
    init_weight_bit=4
    init_spike_bit=4
    init_spike_step=1
    
    
    use_bit_loss=True
    weight_bit_tar=3
    spike_bit_tar=2
    spike_step_tar=1
    
    weight_bit_penalty=4e-2
    spike_bit_penalty=4e-2
    spike_step_penalty=1e-2
    
    bit_shift_lr = 1e-2 #
    
    stop_renew_epoch=-24
    renew_switch_least_epoch=7
    decay = 1.0
    learn_decay=False
    
    T_max=None
    
    logfile = "./imagenet"
    modeltag = '{}_data_aug_seed{}_imagenet_bz{}_ep{}_w{}s{}t{}_learn_w{}s{}t{}_tar_w{}s{}t{}_bit_loss{}'.format(arch, seed, batch_size, epoch, init_weight_bit, init_spike_bit, init_spike_step, learn_weight_bit, learn_spike_bit, learn_time_step, weight_bit_tar, spike_bit_tar, spike_step_tar, use_bit_loss)
    tail = 'renew_stop_at{}'.format(stop_renew_epoch)
    modeltag += tail
    # resume_pth = '/mnt/lustre/GPU8/home/yaoxingting/codes/Ternary-Spike/logs_q_new/models/ResNet18_q_data_aug_seed3407_imagenet_bz128_ep200_w4s4t1_learn_wTruesTruetTrue_tar_w4s4t1_bit_lossTruerenew_stop_at-24-checkpoint-000072.pth'
    # model_name = "{}-checkpoint-{:06}.pth.tar".format(modeltag, epoch)                                        #总的node端口设置                                #gpu 数量
    subprocess.check_call(list2str(["CUDA_VISIBLE_DEVICES=" + gpu, "python", "-m", "torch.distributed.run",  "--master_port", "{}".format(random_integer), "--nproc_per_node", "{}".format(gpu_num), '--nnodes', '1',
                                    "train_cifar_q.py",
                                    "--dataset", 'ImageNet',
                                    "--datapath", imagenet_datapath,
                                    "--arch", arch,
                                    # "--pretrained", cifar10_res20_pretrained,
                                    
                                    "--epochs", "{}".format(epoch),
                                    "--batch", "{}".format(batch_size),
                                    "--seed", "{}".format(seed),
                                    
                                    "--lr", "1e-1", 
                                    # "--lr_spike_scale", "1e-2", 
                                    
                                    
                                    "--weight_quant_per_layer",    # weght 的可学参数是否按照 per layer 分配
                                    "--spike_quant_per_layer",    # spike 的可学参数是否按照 per layer 分配
                                    "--spike_quant_all_positive",    # spike 是否都为正数
                                    
                                    
                                    "--lr_q_weight_bit", "1e-1", 
                                    "--lr_q_weight_scale", "1e-1", 
                                    "--lr_q_weight_shift", "{}".format(bit_shift_lr), 
                                    
                                    "--lr_q_feature_bit", "1e-1",       #e.q. lr_spike_bit
                                    "--lr_q_feature_scale", "1e-1",     #e.q. lr_spike_scale
                                    "--lr_q_feature_shift", "{}".format(bit_shift_lr),     
                                    
                                    "--lr_q_time_step", "1e-2", 
                                    
                                    
                                    "--spiking_mode",
                                    "--init_weight_bit", '{}'.format(init_weight_bit),
                                    "--init_spike_bit", '{}'.format(init_spike_bit),
                                    "--init_spike_step", '{}'.format(init_spike_step),
                                    
                                    "--learn_weight_bit" if learn_weight_bit else '',
                                    "--learn_spike_bit" if learn_spike_bit else '',
                                    "--learn_time_step" if learn_time_step else '',
                                    
                                    
                                    "--use_bit_loss" if use_bit_loss else '',
                                    '--weight_bit_tar', '{}'.format(weight_bit_tar),
                                    '--spike_bit_tar', '{}'.format(spike_bit_tar),
                                    '--spike_step_tar', '{}'.format(spike_step_tar),
                                    
                                    '--weight_bit_penalty', '{}'.format(weight_bit_penalty),
                                    '--spike_bit_penalty', '{}'.format(spike_bit_penalty),
                                    '--spike_step_penalty', '{}'.format(spike_step_penalty),
                                    
                                    #ImageNet args
                                    # '--warm_up',
                                    # '--amp',
                                    # '--resume', resume_pth,
                
                                    '--renew_switch_epoch', '{}'.format(stop_renew_epoch),
                                    '--renew_switch_least_epoch', '{}'.format(renew_switch_least_epoch),

                                    
                                    "--learn_decay" if learn_decay else '',
                                    '--decay', '{}'.format(decay),                                   
                                    "--modeltag", modeltag,
                                    ">",
                                    logfile + "/" + modeltag + '.log',
                                    "2>&1"
                                    ]), shell=True)
    
    
    