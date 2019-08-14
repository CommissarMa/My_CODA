from easydict import EasyDict as edict
import yaml


def parse_params_and_print(cfg_path):
    '''
    从配置文件中读取参数并打印
    cfg_path：配置文件路径
    '''
    with open(cfg_path, 'r') as f:
        cfg = edict(yaml.load(f))        
    dataset = cfg.DATASET
    data_path = cfg[dataset].DATA_PATH
    target_data_path = cfg[dataset].TARGET_DATA_PATH
    log_path = cfg[dataset].LOG_PATH
    pre_trained_path = cfg[dataset].PRE_TRAINED_PATH
    batch_size = cfg[dataset].BATCH_SIZE
    lr = float(cfg[dataset].LEARNING_RATE)
    epoch_num = cfg[dataset].EPOCH_NUM
    steps = cfg[dataset].STEPS
    decay_rate = cfg[dataset].DECAY_RATE
    start_epoch = cfg[dataset].START_EPOCH
    snap_shot = cfg[dataset].SNAP_SHOT
    resize = cfg[dataset].RESIZE
    val_size = cfg[dataset].VAL_SIZE
    
    # 打印参数
    print("Choosen parameters:")
    print("-------------------")
    print("Dataset: ", dataset)
    print("Data location: ", data_path)
    print("Target Data location: ", target_data_path)
    print("Tensorboard log root: ", log_path)
    print("Pre-trained model path:", pre_trained_path)
    print("Batch size:", batch_size)
    print("Learning rate: ", lr)
    print("Total epoch number: ", epoch_num)
    print("Learning rate steps: ", steps)
    print("Learning rate decay rate: ", decay_rate)
    print("Start epoch: ", start_epoch)
    print("Snap shot: ", snap_shot)
    print("Image resize: ", resize)
    print("Validation size: ", val_size)
    print("===================")
    print("")

    return dataset, data_path, target_data_path, log_path, pre_trained_path, batch_size, lr, epoch_num, steps, decay_rate, start_epoch, snap_shot, resize, val_size
