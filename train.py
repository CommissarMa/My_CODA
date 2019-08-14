#%%
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import sys
import time

from utils.config import parse_params_and_print
from utils.logger import Logger
from dataset import loading_data
from model import CSRNet
from utils.lr_policy import update_lr


if __name__ == '__main__':
    
    # 参数配置
    cfg_path = './config/gcc2shhb.yml'
    (dataset, data_path, target_data_path, log_path, pre_trained_path, batch_size, lr, epoch_num, steps, decay_rate, start_epoch, snap_shot, resize, val_size) = parse_params_and_print(cfg_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = Logger(log_path)

    train_loader, _ = loading_data(data_path, mode='train', batch_size=batch_size)
    val_loader, _ = loading_data(target_data_path, mode='val')

    net = CSRNet().to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9)
    criterion_dens = nn.MSELoss(size_average=False)
    criterion_count = nn.L1Loss(size_average=False)

    # 开始训练和验证
    best_mae = sys.maxsize
    for epoch in range(start_epoch,epoch_num+1):
        print('Epoch {}/{}'.format(epoch, epoch_num))
        # 训练阶段
        optimizer = update_lr(optimizer, epoch, steps, decay_rate)
        net.train()

        running_loss = 0.0
        running_mse = 0.0
        running_mae = 0.0
        totalnum = 0
        for idx, (image,densityMap) in enumerate(train_loader):
            image = image.to(device)
            densityMap = densityMap.to(device)

            optimizer.zero_grad()
            duration = time.time()
            predDensityMap = net(image)
            predDensityMap = torch.squeeze(predDensityMap)
            densityMap = torch.squeeze(densityMap)
            loss = criterion_dens(predDensityMap, densityMap)
            time_elapsed = time.time() - duration

            outputs_np = predDensityMap.data.cpu().numpy()
            densityMap_np = densityMap.data.cpu().numpy()

            pre_dens = np.sum(outputs_np.reshape((outputs_np.shape[0],-1)),1)
            gt_count = np.sum(densityMap_np.reshape((densityMap_np.shape[0],-1)),1)

            totalnum += outputs_np.shape[0]

            running_mae += np.sum(np.abs(pre_dens-gt_count))
            running_mse += np.sum((pre_dens-gt_count)**2)

            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()
        
        epoch_loss = running_loss / totalnum
        epoch_mae = running_mae/totalnum
        epoch_mse = np.sqrt(running_mse/totalnum)
        print('Train Loss: {:.4f} density MAE: {:.4f}  density MSE: {:.4f}'.format(epoch_loss,epoch_mae,epoch_mse))
        logger.scalar_summary('Train_loss', epoch_loss, epoch)
        logger.scalar_summary('Train_dens_MAE', epoch_mae, epoch)
        logger.scalar_summary('Train_dens_MSE', epoch_mse, epoch)
        
        # 验证阶段
        net.eval()
        running_loss = 0.0
        running_mse = 0.0
        running_mae = 0.0
        totalnum = 0
        for idx, (image,densityMap) in enumerate(val_loader):
            image = image.to(device)
            densityMap = densityMap.to(device)

            optimizer.zero_grad()
            duration = time.time()
            predDensityMap = net(image)

            outputs_np = predDensityMap.data.cpu().numpy()
            densityMap_np = densityMap.data.cpu().numpy()

            pre_dens = np.sum(outputs_np.reshape((outputs_np.shape[0],-1)),1)
            gt_count = np.sum(densityMap_np.reshape((densityMap_np.shape[0],-1)),1)

            totalnum += outputs_np.shape[0]

            running_mae += np.sum(np.abs(pre_dens-gt_count))
            running_mse += np.sum((pre_dens-gt_count)**2)

            running_loss += loss.data.item()
        
        epoch_loss = running_loss / totalnum
        epoch_mae = running_mae/totalnum
        epoch_mse = np.sqrt(running_mse/totalnum)
        print('Val Loss: {:.4f} density MAE: {:.4f}  density MSE: {:.4f}'.format(epoch_loss,epoch_mae,epoch_mse))
        logger.scalar_summary('Val_loss', epoch_loss, epoch)
        logger.scalar_summary('Val_dens_MAE', epoch_mae, epoch)
        logger.scalar_summary('Val_dens_MSE', epoch_mse, epoch)

#%%
