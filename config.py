from utils import get_weight_path,get_weight_list

__all__ = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152","resnext50_32x4d","resnext101_32x8d","resnext101_64x4d","wide_resnet50_2","wide_resnet101_2",
           "vit_b_16","vit_b_32","vit_l_16","vit_l_32","vit_h_14"]

NET_NAME = 'resnext101_32x8d'
DATASET = 'CUB_200'
# DATASET = 'Stanford_Dogs'
VERSION = 'v3.0'
DEVICE = '1'
# Must be True when pre-training and inference
PRE_TRAINED = False
# 1,2,3,4,5
CURRENT_FOLD = 1                    # 当前进行训练或验证的折数，通常用于k折交叉验证中
GPU_NUM = len(DEVICE.split(','))    
FOLD_NUM = 5                        # 总的折数，用于k折交叉验证中

# 表示CUB-200-2011数据集的训练集的均值和标准差 这些值通常用于对数据进行标准化或归一化处理
CUB_TRAIN_MEAN = [0.48560741861744905, 0.49941626449353244, 0.43237713785804116]
CUB_TRAIN_STD = [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]

# 表示stanford dogs数据集的训练集的均值和标准差
DOGS_TRAIN_MEAN = [0.4761, 0.4518, 0.3910]
DOGS_TRAIN_STD = [0.2268, 0.2219, 0.2203]

CKPT_PATH = './ckpt/{}/fold{}'.format(VERSION,CURRENT_FOLD)     # 检查点路径
WEIGHT_PATH = get_weight_path(CKPT_PATH)                        # 根据检查点路径获取权重文件路径
# print(WEIGHT_PATH)

if PRE_TRAINED:
    WEIGHT_PATH_LIST = get_weight_list('./ckpt/{}/'.format(VERSION))
else:
    WEIGHT_PATH_LIST = None

if DATASET == 'CUB_200':
    num_classes = 200
    train_mean = CUB_TRAIN_MEAN
    train_std = CUB_TRAIN_STD
elif DATASET == 'Stanford_Dogs':
    num_classes = 120
    train_mean = DOGS_TRAIN_MEAN
    train_std = DOGS_TRAIN_STD
else:
    raise ValueError(f"Unknown dataset: {DATASET}")


# Arguments when trainer initial    始化训练器时使用的各种参数配置
INIT_TRAINER = {
    'net_name':NET_NAME,
    'lr':5e-5, 
    'n_epoch':200,
    'num_classes':num_classes,
    'image_size':256,
    'batch_size':16,
    'train_mean':train_mean,    # 训练集的均值和标准差，用于数据标准化。
    'train_std':train_std,      # 训练集的均值和标准差，用于数据标准化。
    'num_workers':2,                # 数据加载时使用的工作进程数量
    'device':DEVICE,
    'pre_trained':PRE_TRAINED,
    'weight_path':WEIGHT_PATH,
    'weight_decay': 1e-3,           # 权重衰减参数，用于优化器
    'momentum': 0.9,                # 动量参数，用于优化器
    'gamma': 0.1,                   # 用于学习率调整的参数
    'milestones': [30,60,90],       # 用于学习率调整的时期列表
#     'milestones': [40, 80],       # 用于学习率调整的时期列表
    'T_max':5,                      # 在学习率调整策略中用于控制余弦退火的参数
    'use_fp16':True,                # 是否使用半精度浮点数
    'dropout':0.01                  # 丢弃率
 }

# Arguments when perform the trainer 
SETUP_TRAINER = {
    'output_dir':'./ckpt/{}'.format(VERSION),
    'log_dir':'./log/{}'.format(VERSION),
    'optimizer':'AdamW',            # 优化器的名称
    'loss_fun':'Cross_Entropy',     # 损失函数的名称
    'class_weight':None,            # 类别权重，用于在损失函数中平衡不同类别样本的影响
    'lr_scheduler':'MultiStepLR'    # 学习率调整策略的名称
}

