# To Be Uploaded Right After Code Refactoring

Seems like I do not have enough time for this project to reach a perfect open-source state. The whole codes are just too many. **For Now, Just Help Yourself**. 

This is a fraction of the official codes of ["Towards-Efficient-and-Accurate-Spiking-Neural-Networks-via-Adaptive-Bit-Allocation"](https://arxiv.org/abs/2506.23717), accepted at Neural Networks, Elsevier.

## Dataset
CIFAR

ImageNet.

Other datasets have no ways but to be waiting for uncertain,future updates of this project.

## Get Started

For cifar experiments
```
nohup python -u autoRun_q.py > /dev/null 2>&1 &
```


For cifar experiments
```
nohup python -u autoRun_q_imagenet.py > /dev/null 2>&1 &
```

Of course, you need to read and modify `autoRun_q.py` or `autoRun_q_imagenet.py` in the first place. 

Configure data_path, model_type, and any other hyper-parameters with  `autoRun_q.py` or `autoRun_q_imagenet.py` to fit your experimental environment. And, `python` one of them to start this project and train a specific mixed-precision SNN.

I think  `autoRun_q.py` and `autoRun_q_imagenet.py` are quite easy to read and understand. Given I am busy getting used to my new job, I am not intended to provide a detailed explanation or 'user guideline' for now.



## Notes
Feel free to contact me if any request exists. I am very likely to spare little time to response because I am really really busy and fully occupied with the mess of my new job.

