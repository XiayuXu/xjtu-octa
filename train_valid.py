# Input file Data train
# Output file Checkpoint Train_log
# train2是对valid可视化， 这样可以减小内存消耗，增大batch。但是出现了随机变化， 这个问题需要查找
import os
import numpy as np
import time
from PIL import Image
from tqdm import tqdm

import torch
import torchsummary
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter

from Model.Data_Loader import Images_Dataset
from Model.Unet_Family import U_Net
from Model.Make_floder import makefloder
from Model.Losses import calc_loss, calc_dice
import copy

# GPU 检查GPU是否可用
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('Cuda is not available. Training on CPU')
else:
    print('Cuda is available. Training on GPU')
device = torch.device("cuda:0" if train_on_gpu else "cpu")

""" 超参数设置 """
Data_path = 'Data'  # 数据路径
result_path = './Result_64'
makefloder(result_path)  # 创建存放结果的文件夹
# writer = SummaryWriter('./train_log')

Batch_Size = 4
Epoch = 30
initial_learning_rate = 0.01
Num_Workers = 4        # 使用线程数目,负责并发
shuffle = True
Random_seed = 50
pin_memory = False     # Dataloader 参数. 如果GPU可以使用,则设置为True.
if train_on_gpu:
    pin_memory = True  # If ``True``, the data loader will copy tensors into CUDA pinned memory before returning them.
valid_loss_min = np.inf

"""　参数选择与设置　"""
# U-Net 模型选择与设置(Input_channel and Output_channel)
Model = {'U_Net': U_Net}
In_ch = 4
Out_ch = 3
# model = Model['AttU_Net'](In_ch, Out_ch)
model = Model['U_Net'](In_ch, Out_ch)
model.to(device)

torchsummary.summary(model, (4, 320, 320))

# 训练数据集路径
train_path = os.path.join(Data_path, 'train')
train_label_path = os.path.join(Data_path, 'train_GT')
valid_path = os.path.join(Data_path, 'test')
valid_label_path = os.path.join(Data_path, 'valid_GT')
# 结果路径设置

""" 数据加载　"""
# Dataset 将训练数据集加载到tensor
Train_data = Images_Dataset(train_dir=train_path, label_dir=train_label_path)
Valid_data = Images_Dataset(train_dir=valid_path, label_dir=valid_label_path)
print(f'Train data number is {len(Train_data)};')
print(f'Train images number is {len(Train_data.train_A_images)};')
print(f'Label images number is {len(Train_data.label_images)};')

# 分割 Training 与 Validation 集合

train_idx = list(range(len(Train_data)))
valid_idx = list(range(len(Valid_data)))
train_sampler = SubsetRandomSampler(train_idx)  # 无放回地按照给定的索引列表采样样本元素。
valid_sampler = SubsetRandomSampler(valid_idx)

# DataLoader 按批加载数据
train_loader = torch.utils.data.DataLoader(Train_data, batch_size=Batch_Size, sampler=train_sampler,
                                           pin_memory=pin_memory)  # 注销了线程，不知道是不是版本问题
valid_loader = torch.utils.data.DataLoader(Valid_data, batch_size=Batch_Size, sampler=valid_sampler,
                                           pin_memory=pin_memory)
valid_loader = copy.deepcopy(valid_loader)
""" 训练模型的配置（损失函数和优化器）　"""
# Optimizer 优化器
opt = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)

# PyTorch 在torch.optim.lr_scheduler包中提供了一些调整学习率的技术

T_Max = 60  # 原程序中设置是不正确的
eta_min = 1e-5
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_Max, eta_min)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3, verbose=True)
# Loss  损失函数
loss = calc_loss
Train_loss = np.inf
Valid_loss = np.inf

log = {'train_loss': [], 'valid_loss': [], 'dice_values': [], 'lr': []}
""" Train """
for i in range(Epoch):

    train_loss = 0.0
    valid_loss = 0.0
    dice = 0
    start = time.time()
    # lr = scheduler.get_lr()
    # 训练
    model.train()
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        output = model(x)
        lossT = loss(output, y)
        train_loss += lossT.item() * x.size(0)
        lossT.backward()
        opt.step()
    Train_loss = train_loss / len(train_idx)
    print('Epoch: {}/{} \t Learning Rate: {:.3f} \t Training Loss: {:.6f} \t'
          .format(i + 1, Epoch, opt.param_groups[0]['lr'], Train_loss))

    log['train_loss'].append(Train_loss)
    log['lr'].append(opt.param_groups[0]['lr'])
    # 学习率更新
    scheduler.step(i)
    # 验证
    model.eval()
    torch.no_grad()
    if not i % 1:
        for step, (x, y) in enumerate(valid_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            lossT = calc_loss(output, y)
            dice += calc_dice(output, y).item() * x.size(0)
            valid_loss += lossT.item() * x.size(0)
            if step == 0:
                # 可视化
                logit = torch.softmax(output, dim=1).detach().cpu()  # detach 拷贝，共享内存，脱离计算图， clone不共享，仍在计算图内
                temp = logit[0].clone()
                temp[0, :, :] = logit[0][2, :, :]
                temp[1, :, :] = logit[0][0, :, :]
                temp[2, :, :] = logit[0][1, :, :]
                logit_img = torchvision.transforms.ToPILImage()(temp)
                logit_img.save(result_path + '/Process/Epoch_' + str(i + 1) + '_loss_'
                               + str(round(float(valid_loss/Batch_Size), 4)) + '.tif')

        Valid_loss = valid_loss / len(valid_idx)
        dice_values = dice / len(valid_idx)
        print('Validation Loss: {:.6f} \t Dice:{:.6f}'.format(Valid_loss, dice_values))
        log['valid_loss'].append(Valid_loss)
        log['dice_values'].append(dice_values)
        """ 模型存储 """
        if Valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, Valid_loss))
            torch.save(model.state_dict(), result_path + '/CheckPoint/Epoch_' +
                       str(i + 1) + '_' + str(round(Valid_loss, 4)) + '.pkl')
            if Valid_loss <= valid_loss_min:
                valid_loss_min = Valid_loss

import pandas as pd
df = pd.DataFrame.from_dict(log)
df.to_excel(os.path.join(result_path, 'log.xlsx'))




