import os
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
import numpy as np
import math
import shutil
import random

from torch.nn import functional as F

import data_utils.transform as tr
from data_utils.data_loader import DataGenerator

from utils import dfs_remove_weight
from torch.cuda.amp import autocast, GradScaler
# GPU version.

import matplotlib.pyplot as plt

from config import VERSION


class VolumeClassifier(object):
    '''
    Control the training, evaluation, and inference process.
    Args:
    - net_name: string, __all__ = ["resnet18", "resnet34", "resnet50",...].
    - lr: float, learning rate.
    - n_epoch: integer, the epoch number
    - num_classes: integer, the number of class
    - image_size: integer, input size
    - batch_size: integer
    - num_workers: integer, how many subprocesses to use for data loading.
    - device: string, use the specified device
    - pre_trained: True or False, default False
    - weight_path: weight path of pre-trained model
    '''
    def __init__(self,
                 net_name=None,
                 lr=1e-3,
                 n_epoch=1,
                 num_classes=3,
                 image_size=None,
                 batch_size=6,
                 train_mean=0,
                 train_std=0,
                 num_workers=0,
                 device=None,
                 pre_trained=False,
                 weight_path=None,
                 weight_decay=0.,
                 momentum=0.95,
                 gamma=0.1,
                 milestones=[40, 80],
                 T_max=5,
                 use_fp16=True,
                 dropout=0.01):
        super(VolumeClassifier, self).__init__()

        self.net_name = net_name
        self.lr = lr
        self.n_epoch = n_epoch
        self.num_classes = num_classes
        self.image_size = image_size
        self.batch_size = batch_size
        self.train_mean = train_mean
        self.train_std = train_std
        
        self.num_workers = num_workers
        self.device = device

        self.pre_trained = pre_trained
        self.weight_path = weight_path
        self.start_epoch = 0
        self.global_step = 0
        self.loss_threshold = 1.0
        self.metric_threshold = 0.0
        # save the middle output
        self.feature_in = []
        self.feature_out = []

        self.weight_decay = weight_decay
        self.momentum = momentum
        self.gamma = gamma
        self.milestones = milestones
        self.T_max = T_max
        self.use_fp16 = use_fp16
        self.dropout = dropout
        self.mixup_enable = False

        os.environ['CUDA_VISIBLE_DEVICES'] = self.device

        self.net = self._get_net(self.net_name)

        self.train_acc_record = []
        self.train_loss_record = []
        self.val_acc_record = []
        self.val_loss_record = []

        if self.pre_trained:
            self._get_pre_trained(self.weight_path)

    def trainer(self,
                train_path,
                val_path,
                label_dict,
                output_dir=None,
                log_dir=None,
                optimizer='Adam',
                loss_fun='Cross_Entropy',
                class_weight=None,
                lr_scheduler=None,
                cur_fold=0):

        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        print('Device:{}'.format(self.device))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        pic_save_dir = os.path.join('res_pic',f'{VERSION}')
        log_dir = os.path.join(log_dir, f'fold{str(cur_fold)}')
        output_dir = os.path.join(output_dir, f'fold{str(cur_fold)}')

        if os.path.exists(log_dir):
            if not self.pre_trained:
                shutil.rmtree(log_dir)
                os.makedirs(log_dir)
        else:
            os.makedirs(log_dir)

        if os.path.exists(output_dir):
            if not self.pre_trained:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
        else:
            os.makedirs(output_dir)

        if not os.path.exists(pic_save_dir):
            os.makedirs(pic_save_dir)

        self.writer = SummaryWriter(log_dir)
        self.global_step = self.start_epoch * math.ceil(
            len(train_path) / self.batch_size)

        net = self.net
        lr = self.lr
        loss = self._get_loss(loss_fun, class_weight)

        if len(self.device.split(',')) > 1:
            net = DataParallel(net)

        # dataloader setting
        train_transformer = transforms.Compose([
            tr.ToCVImage(),
            tr.RandomResizedCrop(size=self.image_size, scale=(1.0, 1.0)),
            tr.RandomHorizontalFlip(),
            # tr.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
            # tr.RandomErasing(),
            # tr.CutOut(56),
            tr.ToTensor(),
            tr.Normalize(self.train_mean, self.train_std)
        ])
        # train_transformer = transforms.Compose([
        #     tr.ToCVImage(),
        #     tr.Resize(resized=self.image_size),
        #     tr.CutOut(32),
        #     tr.ToTensor()
        # ])

        train_dataset = DataGenerator(train_path,
                                      label_dict,
                                      transform=train_transformer)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        # copy to gpu
        net = net.cuda()
        loss = loss.cuda()

        # optimizer setting
        optimizer = self._get_optimizer(optimizer, net, lr)
        scaler = GradScaler()

        if self.pre_trained:
            checkpoint = torch.load(self.weight_path)
            optimizer.load_state_dict(checkpoint['optimizer'])

        if lr_scheduler is not None:
            lr_scheduler = self._get_lr_scheduler(lr_scheduler, optimizer)

        early_stopping = EarlyStopping(patience=20,
                                       verbose=True,
                                       monitor='val_acc',
                                       best_score=self.metric_threshold,
                                       op_type='max')

        for epoch in range(self.start_epoch, self.n_epoch):

            train_loss, train_acc = self._train_on_epoch(epoch, net, loss, optimizer, train_loader, scaler)
            # train_loss, train_acc = self._train_on_epoch_adversarially(epoch, net, loss, optimizer, train_loader, scaler)

            self.train_acc_record.append(train_acc)
            self.train_loss_record.append(train_loss)

            torch.cuda.empty_cache()

            val_loss, val_acc = self._val_on_epoch(epoch, net, loss, val_path,label_dict)

            self.val_acc_record.append(val_acc)
            self.val_loss_record.append(val_loss)

            if lr_scheduler is not None:
                lr_scheduler.step()

            print('epoch:{},train_loss:{:.5f},val_loss:{:.5f}'.format(epoch, train_loss, val_loss))

            print('epoch:{},train_acc:{:.5f},val_acc:{:.5f}'.format(epoch, train_acc, val_acc))

            self.writer.add_scalars('data/loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('data/acc', {
                'train': train_acc,
                'val': val_acc
            }, epoch)
            self.writer.add_scalar('data/lr', optimizer.param_groups[0]['lr'],epoch)

            early_stopping(val_acc)

            if val_acc > self.metric_threshold:
                self.metric_threshold = val_acc

                if len(self.device.split(',')) > 1:
                    state_dict = net.module.state_dict()
                else:
                    state_dict = net.state_dict()

                saver = {
                    'epoch': epoch,
                    'save_dir': output_dir,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict()
                }

                file_name = 'epoch={}-train_loss={:.5f}-val_loss={:.5f}-train_acc={:.5f}-val_acc={:.5f}.pth'.format(
                    epoch, train_loss, val_loss, train_acc, val_acc)
                print('Save as :', file_name)
                save_path = os.path.join(output_dir, file_name)

                torch.save(saver, save_path)

            #early stopping
            if early_stopping.early_stop:
                print('Early Stopping!')
                break

        plt.clf()    

        iterations = list(range(1, len(self.train_acc_record) + 1))
        plt.plot(iterations, self.train_acc_record, label='train')
        plt.plot(iterations, self.val_acc_record, label='val')
        plt.title(f'Accuracy of {self.net_name} on CUB_200')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.legend()
        max_val_acc = max(self.val_acc_record)
        max_val_index = self.val_acc_record.index(max_val_acc)
        plt.text(iterations[max_val_index], self.val_acc_record[max_val_index], f'{max_val_acc:.4f}', ha='left', va='bottom')
        # plt.text(iterations[-1], self.train_acc_record[-1], f'{self.train_acc_record[-1]:.4f}', ha='left', va='bottom')
        # plt.text(iterations[-1], self.val_acc_record[-1], f'{self.val_acc_record[-1]:.4f}', ha='left', va='bottom') 
        ylist = [0.20000*i for i in range(0,6)]
        ylist.append(max(self.val_acc_record))
        plt.yticks(ylist)
        plt.tight_layout()
        plt.grid()
        plt.savefig(os.path.join(pic_save_dir, f'{self.net_name}_fold{str(cur_fold)}_Accuracy.jpg'))

        plt.clf()

        plt.plot(iterations, self.train_loss_record, label='train')
        plt.plot(iterations, self.val_loss_record, label='val')
        plt.title(f'Loss of {self.net_name} on CUB_200')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        # plt.text(iterations[-1], self.train_loss_record[-1], f'{self.train_loss_record[-1]:.4f}', ha='left', va='bottom')
        # plt.text(iterations[-1], self.val_loss_record[-1], f'{self.val_loss_record[-1]:.4f}', ha='left', va='bottom')
        plt.grid()
        plt.savefig(os.path.join(pic_save_dir, f'{self.net_name}_fold{str(cur_fold)}_Loss.jpg'))

        self.writer.close()
        dfs_remove_weight(output_dir, 5)

    def _train_on_epoch(self, epoch, net, criterion, optimizer, train_loader,
                        scaler):

        net.train()

        train_loss = AverageMeter()
        train_acc = AverageMeter()

        for step, sample in enumerate(train_loader):

            data = sample['image']
            target = sample['label']

            data = data.cuda()
            target = target.cuda()

            if self.mixup_enable:
                data, targets_a, targets_b, lam = mixup_data(data, target, alpha=1.0)

            with autocast(self.use_fp16):
                output = net(data)
                print(output.shape)
                # loss = criterion(output, target)
                if self.mixup_enable:
                    loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
                else:
                    loss = criterion(output, target)

            optimizer.zero_grad()
            if self.use_fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            output = F.softmax(output, dim=1)
            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            acc = accuracy(output.data, target)[0]
            train_loss.update(loss.item(), data.size(0))
            train_acc.update(acc.item(), data.size(0))

            torch.cuda.empty_cache()

            print('epoch:{},step:{},train_loss:{:.5f},train_acc:{:.5f},lr:{}'.
                  format(epoch, step, loss.item(), acc.item(),
                         optimizer.param_groups[0]['lr']))

            if self.global_step % 10 == 0:
                self.writer.add_scalars('data/train_loss_acc', {
                    'train_loss': loss.item(),
                    'train_acc': acc.item()
                }, self.global_step)

            self.global_step += 1

        return train_loss.avg, train_acc.avg
    
    def fgsm_attack(self, data, epsilon, data_grad):
        sign_data_grad = data_grad.sign()
        perturbed_data = data + epsilon * sign_data_grad
        perturbed_data = torch.clamp(perturbed_data, 0, 1)
        return perturbed_data
    
    def _train_on_epoch_adversarially(self, epoch, net, criterion, optimizer, train_loader,
                        scaler, epsilon=0.1, alpha=0.7):
        net.train()

        train_loss = AverageMeter()
        train_acc = AverageMeter()

        for step, sample in enumerate(train_loader):

            data = sample['image']
            target = sample['label']

            data = data.cuda()
            target = target.cuda()

            data.requires_grad = True

            with autocast(self.use_fp16):
                output = net(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            if self.use_fp16:
                scaler.scale(loss).backward(retain_graph=True)   # 保留计算图
            else:
                loss.backward(retain_graph=True)

            # 生成对抗样本
            data_grad = data.grad.data
            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

            with autocast(self.use_fp16):
                perturbed_output = net(perturbed_data)
                perturbed_loss = criterion(perturbed_output, target)

            # 将原始数据和对抗样本的损失加权合并
            combined_loss = alpha * loss + (1 - alpha) * perturbed_loss

            optimizer.zero_grad()
            # 累积对抗样本的损失并进行反向传播和优化
            if self.use_fp16:
                scaler.scale(combined_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                combined_loss.backward()
                optimizer.step()

            # 使用对抗样本的损失和输出更新记录
            loss = combined_loss
            output = perturbed_output

            # 计算损失和精度
            output = F.softmax(output, dim=1)
            output = output.float()
            loss = loss.float()

            acc = accuracy(output.data, target)[0]
            train_loss.update(loss.item(), data.size(0))
            train_acc.update(acc.item(), data.size(0))

            torch.cuda.empty_cache()

            print('epoch:{}, step:{}, train_loss:{:.5f}, train_acc:{:.5f}, lr:{}'.
                format(epoch, step, loss.item(), acc.item(),
                        optimizer.param_groups[0]['lr']))

            if self.global_step % 10 == 0:
                self.writer.add_scalars('data/train_loss_acc', {
                    'train_loss': loss.item(),
                    'train_acc': acc.item()
                }, self.global_step)

            self.global_step += 1

        return train_loss.avg, train_acc.avg

    def _val_on_epoch(self, epoch, net, criterion, val_path, label_dict):

        net.eval()

        val_transformer = transforms.Compose([
            tr.ToCVImage(),
            # tr.RandomResizedCrop(size=self.image_size, scale=(1.0, 1.0)),
            tr.CenterCrop(cropped=self.image_size),
            tr.ToTensor(),
            tr.Normalize(self.train_mean, self.train_std)
        ])
        # val_transformer = transforms.Compose([
        #     tr.ToCVImage(),
        #     tr.Resize(resized=self.image_size),
        #     tr.ToTensor()
        # ])

        val_dataset = DataGenerator(val_path,
                                    label_dict,
                                    transform=val_transformer)

        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True)

        val_loss = AverageMeter()
        val_acc = AverageMeter()

        with torch.no_grad():
            for step, sample in enumerate(val_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda()

                with autocast(self.use_fp16):
                    output = net(data)
                    loss = criterion(output, target)

                output = F.softmax(output, dim=1)
                output = output.float()
                loss = loss.float()

                # measure accuracy and record loss
                acc = accuracy(output.data, target)[0]
                val_loss.update(loss.item(), data.size(0))
                val_acc.update(acc.item(), data.size(0))

                torch.cuda.empty_cache()

                print('epoch:{},step:{},val_loss:{:.5f},val_acc:{:.5f}'.format(
                    epoch, step, loss.item(), acc.item()))

        return val_loss.avg, val_acc.avg

    def hook_fn_forward(self, module, input, output):

        for i in range(input[0].size(0)):
            self.feature_in.append(input[0][i].cpu().numpy())
            self.feature_out.append(output[i].cpu().numpy())

    def inference(self,
                  test_path,
                  label_dict,
                  net=None,
                  hook_fn_forward=False):

        if net is None:
            net = self.net

        if hook_fn_forward:
            net.avgpool.register_forward_hook(self.hook_fn_forward)

        net = net.cuda()
        net.eval()

        test_transformer = transforms.Compose([
            tr.ToCVImage(),
            # tr.RandomResizedCrop(size=self.image_size, scale=(1.0, 1.0)),
            tr.CenterCrop(cropped=self.image_size),
            tr.ToTensor(),
            tr.Normalize(self.train_mean, self.train_std)
        ])
        # test_transformer = transforms.Compose([
        #     tr.ToCVImage(),
        #     tr.Resize(resized=self.image_size),
        #     tr.ToTensor()
        # ])

        test_dataset = DataGenerator(test_path,
                                     label_dict,
                                     transform=test_transformer)

        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)

        result = {'true': [], 'pred': [], 'prob': []}

        test_acc = AverageMeter()

        with torch.no_grad():
            for step, sample in enumerate(test_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda()  #N
                with autocast(self.use_fp16):
                    output = net(data)
                output = F.softmax(output, dim=1)
                output = output.float()  #N*C

                acc = accuracy(output.data, target)[0]
                test_acc.update(acc.item(), data.size(0))

                result['true'].extend(target.detach().tolist())
                result['pred'].extend(torch.argmax(output, 1).detach().tolist())
                result['prob'].extend(output.detach().tolist())

                print('step:{},test_acc:{:.5f}'.format(step, acc.item()))

                torch.cuda.empty_cache()

        print('average test_acc:{:.5f}'.format(test_acc.avg))

        return result, np.array(self.feature_in), np.array(self.feature_out)

    def _get_net(self, net_name):
        if net_name.startswith('res') or net_name.startswith('wide_res'):
            import model.resnet as resnet
            # net = resnet.__dict__[net_name](
            #     num_classes=self.num_classes
            # )
            net = resnet.__dict__[net_name](
                pretrained=True      # 使用预训练模型
            )

            for param in net.parameters():            # 冻结全部层
                param.requires_grad = False
            
            # 解冻部分层，可以根据具体情况进行调整
            for param in net.layer2.parameters():
                param.requires_grad = True

            for param in net.layer3.parameters():
                param.requires_grad = True

            for param in net.layer4.parameters():
                param.requires_grad = True

            net.fc = nn.Linear(in_features=net.fc.in_features, out_features=self.num_classes)

            # # 使用dropout
            # net.fc = nn.Sequential(
            #     nn.Dropout(p=0.5), 
            #     nn.Linear(in_features=net.fc.in_features, out_features=self.num_classes)
            #     )

        elif net_name.startswith('vit_'):
           import model.vision_transformer as vit
           net = vit.__dict__[net_name](
               num_classes=self.num_classes,
               image_size=self.image_size,
               dropout=self.dropout
           )

        return net

    def _get_loss(self, loss_fun, class_weight=None):
        if class_weight is not None:
            class_weight = torch.tensor(class_weight)

        if loss_fun == 'Cross_Entropy':
            loss = nn.CrossEntropyLoss(class_weight, label_smoothing=0.1)

        return loss

    def _get_optimizer(self, optimizer, net, lr):
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(),
                                         lr=lr,
                                         weight_decay=self.weight_decay)

        elif optimizer == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(),
                                        lr=lr,
                                        momentum=self.momentum,
                                        weight_decay=self.weight_decay)

        elif optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(net.parameters(),
                                         lr=lr,weight_decay=self.weight_decay)

        return optimizer

    def _get_lr_scheduler(self, lr_scheduler, optimizer):
        if lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, verbose=True)
        elif lr_scheduler == 'MultiStepLR':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.milestones, gamma=self.gamma)
        elif lr_scheduler == 'CosineAnnealingLR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.T_max)
        elif lr_scheduler == 'CosineAnnealingWarmRestarts':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 20, T_mult=2)

        return lr_scheduler

    def _get_pre_trained(self, weight_path):
        checkpoint = torch.load(weight_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
    
          
    def _load_cross_train(self, weight_path):
        checkpoint = torch.load(weight_path)
        self.net.load_state_dict(checkpoint['state_dict'])


# computing tools


class AverageMeter(object):
    '''
    Computes and stores the average and current value
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    '''
    Computes the precision@k for the specified values of k
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""
    def __init__(self,
                 patience=10,
                 verbose=True,
                 delta=0,
                 monitor='val_loss',
                 best_score=None,
                 op_type='min'):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement. 
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type

        if self.op_type == 'min':
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):

        score = -val_score if self.op_type == 'min' else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0

    def print_and_update(self, val_score):
        '''print_message when validation score decrease.'''
        if self.verbose:
            print(
                self.monitor,
                f'optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...'
            )
        self.val_score_min = val_score

def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)