import os

class DefaultConfigs(object):
    # set default configs, if you don't understand, don't modify
    seed = 666            # set random seed
    workers = 12           # set number of data loading workers (default: 4)
    beta1 = 0.9           # adam parameters beta1
    beta2 = 0.999         # adam parameters beta2
    mom = 0.9             # momentum parameters
    #wd = 1e-4             # weight-decay   # clw note: origin is 1e-4, but kaggle top solution use 1e-6 TODO
    wd = 1e-4
    resume = None         # path to latest checkpoint (default: none),should endswith ".pth" or ".tar" if used
    start_epoch = 0       # deault start epoch is zero,if use resume change it

    ########################################################################################
    '''
    文件结构如下： 
        /home/user/dataset/train/0   
        /home/user/dataset/train/1
        /home/user/dataset/train/2
        ......
        /home/user/dataset/val/0   
        /home/user/dataset/val/1
        /home/user/dataset/val/2
        ...
        
    '''
    #dataset = "/dataset/df/cloud/data/dataset/"  # dataset folder with train and val
    #dataset = "/home/user/dataset"
    #dataset = "/home/user/dataset/gunzi/v0.2"
    #dataset = "/home/user/dataset/nachi/ai"
    dataset = "/home/user/dataset/kaggle2020_leaf"
    dataset_merge_csv = "/home/user/dataset/kaggle_cassava_merge/train"
    num_classes = len(os.listdir(os.path.join(dataset, 'train')))
    submit_example =  "./submit_example.csv"
    checkpoints = "./checkpoints/"        # path to save checkpoints
    log_dir = "./logs/"                   # path to save log files
    submits = "./submits/"                # path to save submission files

    model_name = "vit_base_patch16_384"  # "resnet18", "resnet34", "resnet50"、"se_resnext50_32x4d"、"resnext50_32x4d"、
                                    # "shufflenet_v2_x1_0"、"shufflenetv2_x0.5"、"efficientnet-b3"、“efficientnet-b4”、
                                    # “efficientnet-b5”、 vit_base_patch16_384  vit_large_patch16_384
                                    # tf_efficientnet_l2_ns_475

    sampler = "RandomSampler"   # "RandomSampler"、"WeightedSampler"、"imbalancedSampler"（和WeightedSampler基本一样）
    lr_scheduler = "cosine_change_per_epoch" # lr scheduler method: "step", "cosine_change_per_epoch", "cosine_change_per_batch", "adjust","on_loss","on_acc",    adjust不需要配置这里的epoch和lr
    step_milestones = [5, 10, 14]
    step_gamma = 0.1

    epochs = 10
    optim = "sgd"  # "adam","radam","novograd",sgd","ranger","ralamb","over9000","lookahead","lamb"
    if optim == "adam":
        lr = 1e-4  # adam: 1e-4, 3e-4, 5e-4
    elif optim == "sgd":
        lr = 1e-1 if not "vit" in model_name else 1e-2

    bs = 32         # clw note: bs=128, 配合input_size=784, workers = 12，容易超出共享内存大小  报错：ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
    input_size = (512, 512) if "vit" not in model_name else (384, 384)   # clw note：注意是 w, h   512、384、784、(800, 600)

    freeze_bn_epochs = 5
    drop_out_rate = 0.2 if "efficientnet" in model_name else 0.0
    loss_func = "LabelSmoothingLoss" #  "LabelSmoothingLoss"、 "LabelSmoothingLoss_clw", "CELoss"、"BCELoss"、"FocalLoss"、“FocalLoss_clw”、
    label_smooth_epsilon = 0.2
    gpu_id = "0"           # default gpu id
    fp16 = True          # use float16 to train the model
    opt_level = "O1"      # if use fp16, "O0" means fp32，"O1" means mixed，"O2" means except BN，"O3" means only fp16

    do_mixup_in_dataset = 0
    do_cutmix_in_dataset = 0  # in __get_items__()
    do_cutmix_in_batch = 0


    def __str__(self):  # 定义打印对象时打印的字符串
        return  "epochs: " + str(self.epochs) + '\n' + \
                "lr: " + str(self.lr) + '\n' + \
                "lr_scheduler: " + str(self.lr_scheduler) + '\n' + \
                ("step_milestone: " + str(self.step_milestones) + '\n' if self.lr_scheduler == "step" else "") + \
                ("step_gamma: "  + str(self.step_gamma) + '\n' if self.lr_scheduler == "step" else "") + \
                "optim: " + self.optim + '\n' + \
                "weight_decay: " + str(self.wd) + '\n' + \
                "bs: " + str(self.bs) + '\n' + \
                "input_size: " + str(self.input_size) + '\n' + \
                "sampler: " + str(self.sampler) + '\n' + \
                "model_name: " + self.model_name + '\n' + \
                "drop_out_rate: " + str(self.drop_out_rate) + '\n' + \
                "freeze_bn_epochs: " + str(self.freeze_bn_epochs) + '\n' + \
                "loss_func: " + self.loss_func + '\n' + \
                ("label_smooth_epsilon: " + str(self.label_smooth_epsilon) + '\n'  if self.loss_func.startswith("LabelSmoothingLoss") else "") + \
                "fp16: " + ("True" if self.fp16 else "False") + '\n' + \
                "do_mixup_in_dataset: " + str(self.do_mixup_in_dataset) + '\n' + \
                "do_cutmix_in_dataset: " + str(self.do_cutmix_in_dataset) + '\n' + \
                "do_cutmix_in_batch: " + str(self.do_cutmix_in_batch)

configs = DefaultConfigs()
