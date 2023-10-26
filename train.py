import pandas as pd
import argparse
import os
from collections import OrderedDict
from glob import glob
import yaml

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import albumentations as albu
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from loss import BCEDiceLoss
from dataset import LidcDataset
from metrics import dice_train
from utils import AverageMeter

from models.unet_model_1 import UNet_1
from models.unet_model_2 import UNet_2

#configs
lr = 1e-5
weight_decay = 1e-4
batch_size = 12
epochs = 100


def train(train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target in train_loader:
        input = input.cuda()
        target = target.cuda()
        
        output = model(input)
        loss = criterion(output, target)
        dice = dice_train(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('dice',avg_meters['dice'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('dice',avg_meters['dice'].avg)])
    
def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter()}
    
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target in val_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)
            dice = dice_train(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('dice',avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('dice',avg_meters['dice'].avg)])

def main():
    os.makedirs('model_outputs', exist_ok = True)
    criterion = BCEDiceLoss().cuda()
    cudnn.benchmark = True
    
    model = UNet_1(n_channels = 1, n_classes = 1, bilinear = True)
    model = model.cuda()
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.Adam(params, lr = lr, weight_decay = weight_decay)
    
    IMAGE_DIR = '//Image/'
    MASK_DIR = '//Mask/' 
    
    meta = pd.read_csv('meta.csv')
    
    meta['original_image']= meta['original_image'].apply(lambda x:IMAGE_DIR+ x +'.npy')
    meta['mask_image'] = meta['mask_image'].apply(lambda x:MASK_DIR+ x +'.npy')
    
    train_meta = meta[meta['Segmentation_train']==True]
    val_meta = meta[meta['Segmentation_train']==False]
    
    train_image_paths = list(train_meta['original_image'])
    train_mask_paths = list(train_meta['mask_image'])
    val_image_paths = list(val_meta['original_image'])
    val_mask_paths = list(val_meta['mask_image'])
    
    train_dataset = LidcDataset(train_image_paths, train_mask_paths)
    val_dataset = LidcDataset(val_image_paths,val_mask_paths)
    
    batch_size = 12
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        pin_memory = True,
        drop_last = True,
        num_workers = 6)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False,
        pin_memory = True,
        drop_last = False,
        num_workers = 6)
    
    log = pd.DataFrame(index=[],columns= ['epoch','lr','loss','dice','val_loss','val_dice'])

    best_dice = 0
    trigger = 0
    
    for epoch in range(epochs):
        train_log = train(train_loader, model, criterion, optimizer)
        val_log = validate(val_loader, model, criterion)
        print('Training epoch [{}/{}], Training BCE loss:{:.4f}, Training DICE:{:.4f}, Validation BCE loss:{:.4f}, Validation Dice:{:.4f}'.format(
            epoch + 1, epochs, train_log['loss'], train_log['dice'], val_log['loss'], val_log['dice']))
        
        tmp = pd.Series([
            epoch,
            lr,
            train_log['loss'],
            train_log['dice'],
            val_log['loss'],
            val_log['dice']
        ], index=['epoch', 'lr', 'loss', 'dice', 'val_loss', 'val_dice'])
        
        log = log.append(tmp, ignore_index=True)
        log.to_csv('model_outputs/log.csv', index=False)

        trigger += 1
        
        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), 'model_outputs/trained_model.pth')
            best_dice = val_log['dice']
            print("=> saved best model as validation DICE is greater than previous best DICE")
            trigger = 0
        
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()