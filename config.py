class DefaultConfigs(object):
    # set default configs, if you don't understand, don't modify
    seed = 666            # set random seed
    workers = 16           # set number of data loading workers (default: 4)
    beta1 = 0.9           # adam parameters beta1
    beta2 = 0.999         # adam parameters beta2
    mom = 0.9             # momentum parameters
    wd = 1e-4             # weight-decay   # clw note: TODO
    resume = None         # path to latest checkpoint (default: none),should endswith ".pth" or ".tar" if used
    evaluate = False      # just do evaluate
    start_epoch = 0       # deault start epoch is zero,if use resume change it
    split_online = False  # split dataset to train and val online or offline

    # set changeable configs, you can change one during your experiment
    ########################################################################################
    ### clw note: 文件结构如下： /home/user/dataset/train/0   /home/user/dataset/train/1
    #dataset = "/dataset/df/cloud/data/dataset/"  # dataset folder with train and val
    #dataset = "/home/user/dataset"
    dataset = "/home/user/dataset/gunzi/v0.2"
    #dataset = "/home/user/dataset/nachi/ai"
    ########################################################################################
    #test_folder =  "/dataset/df/cloud/data/test/"      # test images' folder
    test_folder =  "/home/user/dataset/gunzi/test_ng/1"
    #test_folder =  "/home/user/dataset/nachi/ai/test"
    #submit_example =  "/dataset/df/cloud/data/submit_example.csv"    # submit example file
    submit_example =  "./submit_example.csv"
    checkpoints = "./checkpoints/"        # path to save checkpoints
    log_dir = "./logs/"                   # path to save log files
    submits = "./submits/"                # path to save submission files

    #epochs = 40           # train epochs
    #epochs = 100           # clw modify
    epochs = 50           # clw modify
    lr_scheduler = "step"  # lr scheduler method,"adjust","on_loss","on_acc","step"  # clw note: TODO
    step_size = 20    # clw modify
    #step_size = 20    # clw modify
    #lr = 2e-3             # learning rate
    #lr = 2e-2             # clw modify
    lr = 1e-1             # clw modify
    #bs = 16
    bs = 32               # batch size

    input_size = 512      # model input size or image resied
    #input_size = 384      # clw modify
    #num_classes = 9       # num of classesstep_size
    num_classes = 2
    #num_classes = 6
    gpu_id = "0"          # default gpu id
    #model_name = "se_resnext50_32x4d-model-sgd-512"      # model name to use
    #model_name = "resnet50"
    #model_name = "resnext50_32x4d"
    model_name = "shufflenet_v2_x1_0"
    #model_name = "shufflenetv2_x0_5"
    #model_name = "shufflenetv2_x1_5"
    optim = "sgd"        # "adam","radam","novograd",sgd","ranger","ralamb","over9000","lookahead","lamb"
    #fp16 = True          # use float16 to train the model
    fp16 = False
    opt_level = "O1"      # if use fp16, "O0" means fp32，"O1" means mixed，"O2" means except BN，"O3" means only fp16
    keep_batchnorm_fp32 = False  # if use fp16,keep BN layer as fp32
    loss_func = "CrossEntropy" # "CrossEntropy"、"FocalLoss"、"LabelSmoothCE"   # clw note: TODO
    #loss_func = "FocalLoss_clw"
    #loss_func = "FocalLoss"


    
configs = DefaultConfigs()
