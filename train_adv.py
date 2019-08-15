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
from model import CSRNet, Discriminator
from utils.lr_policy import update_lr
from utils import loadModel


def lr_poly(base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate_D(optimizer, i_iter, lr_D, epoch_num, power):
    lr = lr_poly(lr_D, i_iter, epoch_num, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return optimizer


if __name__ == '__main__':
    
    # 参数配置
    cfg_path = './config/gcc2shhb_adv.yml'
    (dataset, data_path, target_data_path, log_path, pre_trained_path, batch_size, lr, epoch_num, steps, decay_rate, start_epoch, snap_shot, resize, val_size) = parse_params_and_print(cfg_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = Logger(log_path)

    lr_D = 1e-3
    train_loader, _ = loading_data(data_path, mode='train', batch_size=batch_size)
    target_loader, _ = loading_data(target_data_path, mode='train', batch_size=batch_size)
    val_loader, _ = loading_data(target_data_path, mode='val')

    net = CSRNet().to(device)
    net_D = Discriminator(1).to(device)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9)
    optimizer_D = optim.Adam(net_D.parameters(), lr=lr_D, betas=(0.9, 0.99))
    
    #load model
    if pre_trained_path['density'] != '':
        net = loadModel(net, pre_trained_path['density'])
    if pre_trained_path['discriminator'] != '':
        net_D = loadModel(net_D,pre_trained_path['discriminator'])

    criterion_dens = nn.MSELoss(size_average=False)
    #self.criterion_count = nn.L1Loss(size_average=False)
    criterion_disc = nn.BCEWithLogitsLoss()
    power = 0.9
    source_label = 0
    target_label = 1
    lambda_adv = 0.001 

    # 训练阶段
    trainloader_iter = enumerate(train_loader)
    targetloader_iter = enumerate(target_loader)
    best_mae = sys.maxsize
    loss_dens_value = 0.0
    loss_adv_value = 0.0
    loss_D_value = 0.0

    running_mse = 0.0
    running_mae = 0.0
    totalnum = 0
    iter_count = 0
    for epoch in range(start_epoch,epoch_num+1):
        iter_count = iter_count + 1
        print('Iteration {}/{}'.format(epoch, epoch_num))

        optimizer = update_lr(optimizer, epoch, steps, decay_rate)
        net.train(True)
        net_D.train(True)
        optimizer_D = adjust_learning_rate_D(optimizer_D, epoch, lr_D, epoch_num, power)
        optimizer.zero_grad()
        optimizer_D.zero_grad()
        
        # train G
        for param in net_D.parameters():
            param.requires_grad = False
        
        # train with source
        _, (image,Dmap) = next(trainloader_iter)
        image = image.to(device)
        Dmap = Dmap.to(device)

        duration = time.time()
        predDmap = net(image)
        loss = criterion_dens(torch.squeeze(predDmap,1), Dmap)
        time_elapsed = time.time() - duration

        outputs_np = predDmap.data.cpu().numpy()
        Dmap_np = (Dmap).data.cpu().numpy()

        # calculate iter size
        iter_size = outputs_np.shape[0]
        totalnum += iter_size
        # backpropogate G
        loss.backward()
        #self.optimizer.step()
        loss_dens_value += loss.data.item()/iter_size

        # train with target
        _, (image_t,_) = next(targetloader_iter)
        image_t = image_t.to(device)
        predDmap_t = net(image_t)
        D_out_t = net_D(predDmap_t)
        loss = lambda_adv * criterion_disc(D_out_t, torch.Tensor(
            torch.FloatTensor(D_out_t.data.size()).fill_(source_label)).to(device))
        
        loss.backward()
        loss_adv_value += loss / iter_size
        #self.optimizer.step()

        #calculating mae & mse
        pre_dens = np.sum(outputs_np.reshape((outputs_np.shape[0], -1)), 1)
        gt_count = np.sum(Dmap_np.reshape((Dmap_np.shape[0], -1)), 1)

        running_mae += np.sum(np.abs(pre_dens - gt_count))
        running_mse += np.sum((pre_dens - gt_count) ** 2)

        # train D
        # bring back requires_grad
        for param in net_D.parameters():
            param.requires_grad = True
        # train with source
        predDmap = predDmap.detach()
        D_out = net_D(predDmap)

        loss = criterion_disc(D_out,torch.Tensor(torch.FloatTensor(D_out.data.size()).fill_(source_label)).to(device))

        loss.backward()
        loss_D_value += loss.data.item() / iter_size

        # train with target
        predDmap_t = predDmap_t.detach()
        D_out_t = net_D(predDmap_t)
        loss = criterion_disc(D_out_t,torch.Tensor(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).to(device))
        loss.backward()

        loss_D_value += loss.data.item() / iter_size
        # optimizer
        optimizer.step()
        optimizer_D.step()

        print('Train density Loss: {:.4f} Density Adversarial Loss: {:.4f}  Discriminator Loss: {:.4f}'.format(loss_dens_value/iter_count, loss_adv_value/iter_count, loss_D_value/iter_count))
        logger.scalar_summary('Temporal/train_density_loss', loss_dens_value, epoch)
        logger.scalar_summary('Temporal/train_adv_loss', loss_adv_value, epoch)
        logger.scalar_summary('Temporal/train_D_loss', loss_D_value, epoch)
        #test mae & mse on train set
        epoch_mae = running_mae / totalnum
        epoch_mse = np.sqrt(running_mse /totalnum)
        print('Training Iteration:{} MAE: {:.4f} MSE: {:.4f}'.format(epoch,epoch_mae, epoch_mse))

        # 验证阶段
        net.eval()
        net_D.eval()

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
