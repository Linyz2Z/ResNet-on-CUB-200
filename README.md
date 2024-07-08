# ResNet-on-CUB-200
## SYSU 2023春 人工神经网络 wrs 部分内容

### 项目结构

本项目文件由以下结构组成

|-- DL2024_proj/

    |-- analysis/ 模型结果输出的文件夹，以及数据统计工具类
    
        |-- result/ 最终结果输出文件夹
        
            |-- xxx/ 按版本号输出的文件夹
            
                |-- foldk.csv 第k折交叉验证的真实标签'true'，预测标签'pred'，预测概率'prob'，图像路径'path'
                
                |-- foldk_report.csv 第k折交叉验证的分类报告
                
                |-- foldk_fc_weight.npy 第k折交叉验证的全连接层权重
                
                神经网络的第k折交叉验证（k-fold cross-validation）是一种用于评估模型性能的方法。
                它通过将数据集分成 k 个子集（折），然后在 k  次迭代中，每次使用其中一个子集作为
                测试集，而其余 k-1 个子集作为训练集进行训练和测试。通过这种方式，可以有效地衡量
                模型在不同数据子集上的表现。
        
        |-- mid_feature/ 钩子（hook）获取的中间结果（某层的输入输出特征）输出文件夹
        
            |-- xxx/ 按版本号输出的文件夹
            
                |-- foldk/ 按第k折交叉验证输出的文件夹
                
                    |-- xxx 逐个输出测试集样本的中间结果
                    
        |-- analysis_tools.py 绘出混淆矩阵、roc曲线、计算CAM、保存热力图的类
        
        |-- statistic_result.py 统计评价指标和auc的类
                
    |-- ckpt/ 模型参数输出的文件夹
    
        |-- xxx/ 按版本号输出的文件夹
        
            |-- foldk/ 按第k折交叉验证输出的文件夹
            
                |-- epoch=xxx.pth 模型参数输出，包含迭代次数'epoch'，保存路径'save_dir'，模型参数字典'state_dict'，优化器参数字典'optimizer'
        
    |-- converter/ 文件读取的工具类的文件夹
            
    |-- csv_file/ csv格式的数据集目录的文件夹
                
    |-- data_utils/ 数据读取的工具类的文件夹
    
        |-- data_loader.py 数据读取
    
        |-- transform.py 数据转换和数据增强
            
    |-- datasets/ 数据集的文件夹
    
        |-- CUB_200_2011/ CUB 200 2011数据集
        
            |-- attributes.txt 
            
            |-- CUB_200_2011/ 
            
                |-- attributes 
                
                |-- images 图像数据
                
                |-- parts 
                
                |-- bounding_boxes.txt 
                
                |-- classes.txt.txt 
                
                |-- image_class_labels.txt 
                
                |-- images.txt 目录文件
                
                |-- README
                
                |-- train_test_split.txt 训练集和测试集的划分文件
           
    |-- log/ 模型运行的过程数据输出的文件夹
    
        |-- xxx/ 按版本号输出的文件夹
        
            |-- foldk/ 按第k折交叉验证输出的文件夹
            
                |-- data/ 过程数据输出文件夹
                
                |-- events.xxxxx 命令行输出结果
        
    |-- model/ 模型文件夹
                
    |-- config.py 模型配置
    
    |-- LICENSE 权利声明
    
    |-- main.py 项目模型入口
    
    |-- requirements.txt 依赖库
    
    |-- make_csv.py 制作csv格式的数据集目录的函数
    
    |-- trainer.py 训练器，训练过程主要代码
    
    |-- utils.py 工具类

### 参数设置
```
INIT_TRAINER = {
    'net_name':'resnext101_32x8d',
    'lr':5e-5, 
    'n_epoch':200,
    'num_classes':200,
    'image_size':256,
    'batch_size':16,
    'train_mean':[0.48560741861744905, 0.49941626449353244, 0.43237713785804116],   
    'train_std':[0.2321024260764962, 0.22770540015765814, 0.2665100547329813],     
    'num_workers':2,        
    'device':DEVICE,
    'pre_trained':False,
    'weight_decay': 1e-3,          
    'momentum': 0.9,             
    'gamma': 0.1,                 
    'milestones': [30,60,90],         
    'T_max':5,                    
    'use_fp16':True,             
    'dropout':0.01              
 }
```

### 项目代码运行
```
在项目根目录下，

单折、多折训练
~~~bash
python main.py -m train
python main.py -m train-cross
~~~

单折、多折推理
~~~bash
python main.py -m inf
python main.py -m inf-cross
~~~

单折、多折推理（不需要结果输出）
~~~bash
python main.py -m inf -s n
python main.py -m inf-cross -s n
~~~
```

### 训练结果
![training_trick](./res_pic/training%20trick.jpg)

![res](./res_pic/resnext101_32x8d_fold1_Accuracy.jpg)
