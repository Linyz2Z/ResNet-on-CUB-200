import os
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time
import shutil
import random

from trainer import VolumeClassifier
from data_utils.csv_reader import csv_reader_single
from config import INIT_TRAINER, SETUP_TRAINER, VERSION, CURRENT_FOLD, WEIGHT_PATH_LIST, FOLD_NUM, DATASET
from converter.common_utils import save_as_hdf5

# 生成多折交叉验证数据集 train_path, validation_path
def get_cross_validation(path_list, fold_num, current_fold):

    _len_ = len(path_list) // fold_num
    train_id = []
    validation_id = []
    end_index = current_fold * _len_    # validation end index 
    start_index = end_index - _len_     # validation start index
    if current_fold == fold_num:
        validation_id.extend(path_list[start_index:])
        train_id.extend(path_list[:start_index])
    else:
        validation_id.extend(path_list[start_index:end_index])
        train_id.extend(path_list[:start_index])
        train_id.extend(path_list[end_index:])

    print(f'train sample number:{len(train_id)}, val sample number:{len(validation_id)}')
    return train_id, validation_id

def get_cross_balance_validation(path_list, label_dict, fold_num, current_fold):
    train_id = []
    val_id = []

    for label in range(0,200):
        temp_list = []
        for id in path_list:
            if label_dict[id] == label:
                temp_list.append(id)
        _len_ = len(temp_list) // fold_num
        end_index = current_fold * _len_
        start_index = end_index - _len_
        random.seed(1)
        random.shuffle(temp_list)
        if current_fold == fold_num:
            val_id.extend(temp_list[start_index:])
            train_id.extend(temp_list[:start_index])
        else:
            val_id.extend(temp_list[start_index:end_index])
            train_id.extend(temp_list[:start_index])
            train_id.extend(temp_list[end_index:])

    return train_id, val_id

# 用于计算给定神经网络模型的参数数量
def get_parameter_number(net):
    # 计算模型中所有参数的数量
    total_num = sum(p.numel() for p in net.parameters())
    # 计算模型中所有可训练参数的数量
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":

    # 使用了 Python 的 argparse 库创建一个命令行参数解析器。
    # 这个解析器用于解析命令行输入的参数，并提供给程序在运行时使用。
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--mode',
                        default='train',
                        choices=["train-cross","train", "inf","inf-cross"],
                        help='choose the mode',
                        type=str)
    parser.add_argument('-s',
                        '--save',
                        default='yes',
                        choices=['no', 'n', 'yes', 'y'],
                        help='save the forward middle features or not',
                        type=str)
    parser.add_argument('-n',
                        '--net_name',
                        #default=None,
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152","resnext50_32x4d","resnext101_32x8d","resnext101_64x4d","wide_resnet50_2","wide_resnet101_2",
           "vit_b_16","vit_b_32","vit_l_16","vit_l_32","vit_h_14"],
                        help='override the INIT_TRAINER[\'net_name\'] of config.py',
                        type=str)
    parser.add_argument('-l',
                        '--lr',
                        #default=None,
                        help='override the INIT_TRAINER[\'lr\'] of config.py',
                        type=float)
    parser.add_argument('-e',
                        '--n_epoch',
                        #default=None,
                        help='override the INIT_TRAINER[\'n_epoch\'] of config.py',
                        type=int)
    parser.add_argument('-c',
                        '--num_classes',
                        #default=None,
                        help='override the INIT_TRAINER[\'num_classes\'] of config.py',
                        type=int)
    parser.add_argument('-is',
                        '--image_size',
                        #default=None,
                        help='override the INIT_TRAINER[\'image_size\'] of config.py',
                        type=int)
    parser.add_argument('-bs',
                        '--batch_size',
                        #default=None,
                        help='override the INIT_TRAINER[\'batch_size\'] of config.py',
                        type=int)
    args = parser.parse_args()
    
    if args.net_name is not None:
        INIT_TRAINER['net_name'] = args.net_name
    if args.lr is not None:
        INIT_TRAINER['lr'] = args.lr
    if args.n_epoch is not None:
        INIT_TRAINER['n_epoch'] = args.n_epoch
    if args.num_classes is not None:
        INIT_TRAINER['num_classes'] = args.num_classes
    if args.image_size is not None:
        INIT_TRAINER['image_size'] = args.image_size
    if args.batch_size is not None:
        INIT_TRAINER['batch_size'] = args.batch_size

    # Set data path & classifier
    
    if args.mode != 'train-cross' and args.mode != 'inf-cross':
        classifier = VolumeClassifier(**INIT_TRAINER)
        # ** 运算符用于将一个字典（例如 INIT_TRAINER）展开为关键字参数（keyword arguments）传递给函数。
        # 也就是说，INIT_TRAINER 字典中的每个键值对将被转换为 VolumeClassifier 类的参数。
        print(get_parameter_number(classifier.net))

    # Training
    ###############################################
    if 'train' in args.mode:
        ###### modification for new data
        if DATASET == 'CUB_200':
            csv_path = './csv_file/cub_200_2011.csv_train.csv'
        elif DATASET == 'Stanford_Dogs':
            csv_path = './csv_file/Stanford_Dogs_train.csv'

        label_dict = csv_reader_single(csv_path, key_col='id', value_col='label')   # label_dict = path : label
        path_list = list(label_dict.keys())     # list of image path
        random.seed(1)                          # 保证实验的可重复性
        random.shuffle(path_list)

        if args.mode == 'train-cross':          # 多折交叉验证训练
            INIT_TRAINER['train_cross'] = True
            for fold in range(1,FOLD_NUM+1):    # from 1 to FOLD_NUM
                print('===================fold %d==================='%(fold))
                if INIT_TRAINER['pre_trained']:
                    INIT_TRAINER['weight_path'] = WEIGHT_PATH_LIST[fold-1]
                
                classifier = VolumeClassifier(**INIT_TRAINER)
                train_path, val_path = get_cross_validation(path_list, FOLD_NUM, fold)
                # train_path, val_path = get_cross_balance_validation(path_list, label_dict, FOLD_NUM, fold)
                SETUP_TRAINER['train_path'] = train_path
                SETUP_TRAINER['val_path'] = val_path
                SETUP_TRAINER['label_dict'] = label_dict
                SETUP_TRAINER['cur_fold'] = fold

                start_time = time.time()
                classifier.trainer(**SETUP_TRAINER)

                print('run time:%.4f' % (time.time() - start_time))
        
        elif args.mode == 'train':              # 普通训练
            train_path, val_path = get_cross_validation(path_list, FOLD_NUM, CURRENT_FOLD)
            # train_path, val_path = get_cross_balance_validation(path_list, label_dict, FOLD_NUM, CURRENT_FOLD)
            SETUP_TRAINER['train_path'] = train_path
            SETUP_TRAINER['val_path'] = val_path
            SETUP_TRAINER['label_dict'] = label_dict
            SETUP_TRAINER['cur_fold'] = CURRENT_FOLD

            start_time = time.time()
            classifier.trainer(**SETUP_TRAINER)

            print('run time:%.4f' % (time.time() - start_time))
    ###############################################

    # Inference
    ###############################################
    elif 'inf' in args.mode:

        if DATASET == 'CUB_200':
            test_csv_path = './csv_file/cub_200_2011.csv_test.csv'
        elif DATASET == 'Stanford_Dogs':
            test_csv_path = './csv_file/Stanford_Dogs_test.csv'
        label_dict = csv_reader_single(test_csv_path, key_col='id', value_col='label')   # label_dict = path : label
        test_path = list(label_dict.keys())     # 测试集路径path
        print('test len:',len(test_path))
        #########

        save_dir = './analysis/result/{}'.format(VERSION)
        feature_dir = './analysis/mid_feature/{}'.format(VERSION)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if args.mode == 'inf':
            save_path = os.path.join(save_dir,f'fold{str(CURRENT_FOLD)}.csv')
            start_time = time.time()
            if args.save == 'no' or args.save == 'n':   # 不保存参数
                result, _, _ = classifier.inference(test_path, label_dict)
                # result = {'true': [], 'pred': [], 'prob': []}
                print('run time:%.4f' % (time.time() - start_time))
            else:                                       # 保存参数
                result, feature_in, feature_out = classifier.inference(
                    test_path, label_dict, hook_fn_forward=True)    # ？？？
                print('run time:%.4f' % (time.time() - start_time))
                # save the avgpool output
                print(feature_in.shape, feature_out.shape)
                feature_save_path = os.path.join(feature_dir,f'fold{str(CURRENT_FOLD)}') # mid_feature\v1.0\fold{CURRENT_FOLD}
                if not os.path.exists(feature_save_path):
                    os.makedirs(feature_save_path)
                else:
                    shutil.rmtree(feature_save_path)
                    # shutil.rmtree(feature_save_path) 会删除指定路径 feature_save_path 处的目录以及其包含的所有文件和子目录
                    # 覆盖之前保存的内容
                    os.makedirs(feature_save_path)
                for i in range(len(test_path)):     # 保存mid_feature
                    name = os.path.basename(test_path[i])
                    feature_path = os.path.join(feature_save_path, name.split(".")[0]) # mid_feature\v1.0\fold{CURRENT_FOLD}\ name(remove .jpg)
                    save_as_hdf5(feature_in[i], feature_path, 'feature_in')
                    save_as_hdf5(feature_out[i], feature_path, 'feature_out')
            result['path'] = test_path
            # result = {'true': [], 'pred': [], 'prob': [], 'path': []}
            csv_file = pd.DataFrame(result)
            csv_file.to_csv(save_path, index=False)
            # 生成foldx_report.csv
            #report 
            cls_report = classification_report(
                result['true'],
                result['pred'],
                output_dict=True)
            #fc weight
            if INIT_TRAINER['net_name'].startswith('res') or INIT_TRAINER['net_name'].startswith('wide_res'):

                fc_weight_save_path = os.path.join(save_dir,f'fold{str(CURRENT_FOLD)}_fc_weight.npy')
                np.save(fc_weight_save_path, classifier.net.state_dict()['fc.weight'].cpu().numpy())
            
            #save as csv
            report_save_path = os.path.join(save_dir,f'fold{str(CURRENT_FOLD)}_report.csv')
            report_csv_file = pd.DataFrame(cls_report)
            report_csv_file.to_csv(report_save_path)
        

        elif args.mode == 'inf-cross':  # 多折推理 （使用多折训练时每一折的权重作为分类器权重进行推理）

            for fold in range(1,FOLD_NUM+1):
                print('===================fold %d==================='%(fold))
                print('weight path %s'%WEIGHT_PATH_LIST[fold-1])
                INIT_TRAINER['weight_path'] = WEIGHT_PATH_LIST[fold-1]      # 模型预训练权重
                classifier = VolumeClassifier(**INIT_TRAINER)
                
                save_path = os.path.join(save_dir,f'fold{str(fold)}.csv')
                start_time = time.time()
                if args.save == 'no' or args.save == 'n':                   # 以下基本同‘inf’
                    result, _, _ = classifier.inference(test_path, label_dict)
                    print('run time:%.4f' % (time.time() - start_time))
                else:
                    result, feature_in, feature_out = classifier.inference(
                        test_path, label_dict, hook_fn_forward=True)
                    print('run time:%.4f' % (time.time() - start_time))
                    # save the avgpool output
                    print(feature_in.shape, feature_out.shape)
                    feature_save_path = os.path.join(feature_dir,f'fold{str(fold)}')
                    if not os.path.exists(feature_save_path):
                        os.makedirs(feature_save_path)
                    else:
                        shutil.rmtree(feature_save_path)
                        os.makedirs(feature_save_path)
                    for i in range(len(test_path)):
                        name = os.path.basename(test_path[i])
                        feature_path = os.path.join(feature_save_path, name.split(".")[0])
                        save_as_hdf5(feature_in[i], feature_path, 'feature_in')
                        save_as_hdf5(feature_out[i], feature_path, 'feature_out')
                result['path'] = test_path
                csv_file = pd.DataFrame(result)
                csv_file.to_csv(save_path, index=False)
                #report
                cls_report = classification_report(
                    result['true'],
                    result['pred'],
                    output_dict=True)
                
                #fc weight
                if INIT_TRAINER['net_name'].startswith('res') or INIT_TRAINER['net_name'].startswith('wide_res'):
                    fc_weight_save_path = os.path.join(save_dir,f'fold{str(fold)}_fc_weight.npy')
                    np.save(fc_weight_save_path, classifier.net.state_dict()['fc.weight'].cpu().numpy())
 
                
                #save as csv
                report_save_path = os.path.join(save_dir,f'fold{str(fold)}_report.csv')
                report_csv_file = pd.DataFrame(cls_report)
                report_csv_file.to_csv(report_save_path)
    ###############################################
