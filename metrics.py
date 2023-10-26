import numpy as np
import torch
import torch.nn.functional as F

def dice_train(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    
    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
        
def dice_val(output, target):
    smooth = 1e-5

    output = output.view(-1)
    output = (output>0.5).float().cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    
    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)