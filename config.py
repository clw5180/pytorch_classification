import os

class DefaultConfigs(object):
    # set default configs, if you don't understand, don't modify
    seed = 666            # set random seed
    workers = 12           # set number of data loading workers (default: 4)
    beta1 = 0.9           # adam parameters beta1
    beta2 = 0.999         # adam parameters beta2
    mom = 0.9             # momentum parameters
    wd = 1e-4             # weight-decay   # clw note: TODO
    resume = None         # path to latest checkpoint (default: none),should endswith ".pth" or ".tar" if used
    start_epoch = 0       # deault start epoch is zero,if use resume change it

    ########################################################################################
    '''
    文件结构如下： 
        /home/user/dataset/train/0   
        /home/user/dataset/train/1
    '''
    #dataset = "/dataset/df/cloud/data/dataset/"  # dataset folder with train and val
    #dataset = "/home/user/dataset"
    #dataset = "/home/user/dataset/gunzi/v0.2"
    #dataset = "/home/user/dataset/nachi/ai"
    dataset = "/home/user/dataset/kaggle2020_leaf"
    num_classes = len(os.listdir(os.path.join(dataset, 'train')))
    submit_example =  "./submit_example.csv"
    checkpoints = "./checkpoints/"        # path to save checkpoints
    log_dir = "./logs/"                   # path to save log files
    submits = "./submits/"                # path to save submission files

    epochs = 15
    lr_scheduler = "cosine"  # lr scheduler method: "step", "cosine", "adjust","on_loss","on_acc",
    optim = "sgd"        # "adam","radam","novograd",sgd","ranger","ralamb","over9000","lookahead","lamb"
    lr = 2e-2  # 2e-3、1e-1
    bs = 32         # clw note: bs=128, 配合input_size=784, workers = 12，容易超出共享内存大小  报错：ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
    input_size = (512, 384)   #   512、384、784、(800, 600)
    sampler = "RandomSampler"   # "RandomSampler"、"WeightedSampler"、"imbalancedSampler"（和WeightedSampler基本一样）

    model_name = "resnet50"  # "resnet50"、"se_resnext50_32x4d"、"resnext50_32x4d"、"shufflenet_v2_x1_0"、"shufflenetv2_x0.5"、"efficientnet-b4"、“efficientnet-l2”、
    loss_func = "CrossEntropy" #  "LabelSmoothCELoss"、"CELoss"、"BCELoss"、"FocalLoss"、“FocalLoss_clw”、   # clw note: TODO
    gpu_id = "0"           # default gpu id
    fp16 = True          # use float16 to train the model
    opt_level = "O1"      # if use fp16, "O0" means fp32，"O1" means mixed，"O2" means except BN，"O3" means only fp16
    keep_batchnorm_fp32 = False  # if use fp16,keep BN layer as fp32


    def __str__(self):  # 定义打印对象时打印的字符串
        return  "epochs: " + str(self.epochs) + '\n' + \
                "lr: " + str(self.lr) + '\n' + \
                "lr_scheduler: " + str(self.lr_scheduler) + '\n' + \
                "optim: " + self.optim + '\n' + \
                "weight_decay: " + str(self.wd) + '\n' + \
                "bs: " + str(self.bs) + '\n' + \
                "input_size: " + str(self.input_size) + '\n' + \
                "sampler: " + str(self.sampler) + '\n' + \
                "model_name: " + self.model_name + '\n' + \
                "loss_func: " + self.loss_func + '\n' + \
                "fp16: " + ("True" if self.fp16 else "False") + '\n'

configs = DefaultConfigs()
print(str(configs))
