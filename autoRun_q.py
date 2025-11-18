import subprocess

def list2str(list):
    j = ''
    for i in list:
        if not i == '':
            j += (i + ' ')
    return j

imagenet_datapath = '/mnt/lustre/GPU8/share/ilsvrc12_raw'
cifar10_datapath = "/mnt/lustre/GPU8/home/yaoxingting/CIFAR10"
cifar100_datapath = '/mnt/lustre/GPU8/home/yaoxingting/CIFAR100'

# cifar10_res20_pretrained = './pretrained_cks/resnet20_cifar_data_aug_step2_seed3407_cifar10_bz128_ep200_using_relu-checkpoint-000193.pth'
cifar10_res20_pretrained = './pretrained_cks/resnet20_cifar_data_aug_step2_seed3407_cifar10_bz128_ep200_using_relu_strict-checkpoint-000193.pth'
cifar100_res20_pretrained = './pretrained_cks/resnet20_cifar_data_aug_step2_seed3407_cifar100_bz128_ep200_using_relu_strict-checkpoint-000197.pth'

# cifar10_res18_pretrained = '/mnt/lustre/GPU8/home/yaoxingting/codes/Ternary-Spike/pretrained_cks/ResNet18_cifar_data_aug_step2_seed3407_cifar10_bz128_ep200_using_relu_strict-checkpoint-000197.pth'
# cifar100_res18_pretrained = '/mnt/lustre/GPU8/home/yaoxingting/codes/Ternary-Spike/pretrained_cks/ResNet18_cifar_data_aug_step2_seed3407_cifar100_bz128_ep200_using_relu_strict-checkpoint-000192.pth'

# I leave this function here that models are able to be initialized to pretrained checkpoints
# In practice, I find pre-loading pretrained checkpoints is not helpful in improving the accuracies of quantized models.
# Just offering a play station. Play with my codes if you want.

#exaple: nohup python -u autoRun_q.py > /dev/null 2>&1 &
if __name__ == "__main__":
    gpu = '3'
    epoch = 200 
    batch_size = 128
    seed = 3407
    arch = 'ResNet18_cifar_q' #choice = [resnet20_cifar, resnet19_cifar, resnet20_cifar_modified, ResNet18, ResNet34, resnet20_cifar_q]
    
    learn_weight_bit = True
    learn_spike_bit = True
    learn_time_step = True
    
    init_weight_bit=4
    init_spike_bit=4
    init_spike_step=2
    
    
    use_bit_loss = True
    weight_bit_tar= 3
    spike_bit_tar= 4
    spike_step_tar= 1
    
    #good when tar bit >= 3
    # weight_bit_penalty=1e-2
    # spike_bit_penalty=2e-2
    # spike_step_penalty=1e-2
    
    
    #testing when tar bit < 3
    weight_bit_penalty=4e-2
    spike_bit_penalty=4e-2
    spike_step_penalty=1e-2
    
    # bit_shift_lr = 1e-4 #good on old shift
    bit_shift_lr = 1e-2 #
    
    stop_renew_epoch=-24
    renew_switch_least_epoch=14
    
    decay = 1.0
    learn_decay=False
    
    T_max=None
    # logfile = "./decay05/RunningLogs_using_strict_pretrained_post_no_shift_both_renew_raw_bit_learn"
    # logfile = "./RunningLogs_mixed_ann_cidar10"
    logfile = "./RunningLogs_res18q_cidar10"
    # logfile = "./RunningLogs_using_strict_pretrained_post_gs_shift_trial_on_new_shift"
    modeltag = '{}_data_aug_seed{}_cifar10_bz{}_ep{}_w{}s{}t{}_learn_w{}s{}t{}_tar_w{}s{}t{}_bit_loss{}'.format(arch, seed, batch_size, epoch, init_weight_bit, init_spike_bit, init_spike_step, learn_weight_bit, learn_spike_bit, learn_time_step, weight_bit_tar, spike_bit_tar, spike_step_tar, use_bit_loss)
    tail = '_act_renew_realresCifar_stop{}_least_valitUtil{}_seed{}_decay{}{}_res18q'.format(stop_renew_epoch, renew_switch_least_epoch, seed,decay,learn_decay)
    modeltag = modeltag + tail
    modeltag = modeltag[49:]
    # modeltag = 'test_test'
    # model_name = "{}-checkpoint-{:06}.pth.tar".format(modeltag, epoch)
    subprocess.check_call(list2str(["CUDA_VISIBLE_DEVICES=" + gpu, "python", "-u", "train_cifar_q.py",
                                    "--dataset", 'CIFAR10',
                                    "--datapath", "/mnt/lustre/GPU8/home/yaoxingting/CIFAR10",
                                    "--arch", arch,
                                    
                                    "--epochs", "{}".format(epoch),
                                    "--batch", "{}".format(batch_size),
                                    "--seed", "{}".format(seed),
                                    



                                    #以下是默认配置不要改，默认输入到
                                    "--lr", "1e-1", 
                                    
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
                                    
                                    
                                    '--renew_switch_epoch', '{}'.format(stop_renew_epoch),
                                    '--renew_switch_least_epoch', '{}'.format(renew_switch_least_epoch),

                                    
                                    "--learn_decay" if learn_decay else '',
                                    '--decay', '{}'.format(decay),
                
                
                                    "--modeltag", modeltag,
                                    ">",
                                    logfile + "/" + modeltag + '.log',
                                    "2>&1"
                                    ]), shell=True)
    
    
    
    




