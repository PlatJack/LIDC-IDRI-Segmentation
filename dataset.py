import os
import numpy as np
import glob

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
import torchvision
from torchvision import transforms
import albumentations as albu
from albumentations.pytorch import ToTensorV2

class LidcDataset(Dataset):
    def __init__(self, IMAGES_PATHS, MASK_PATHS, Albumentation = False):
        self.image_paths = IMAGES_PATHS
        self.mask_paths= MASK_PATHS
        self.albumentation = Albumentation


        self.albu_transformations =  albu.Compose([
            albu.ElasticTransform(alpha = 1.1, alpha_affine = 0.5, sigma = 5, p = 0.15),
            albu.HorizontalFlip(p = 0.15),
            ToTensorV2()
        ])
        self.transformations = transforms.Compose([transforms.ToTensor()])
    
    def transform(self, image, mask):
        if self.albumentation:
            image = image.reshape(512, 512, 1)
            mask = mask.reshape(512, 512, 1)
            mask = mask.astype('uint8')
            augmented=  self.albu_transformations(image = image, mask = mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.reshape([1, 512, 512])
        else:
            image = self.transformations(image)
            mask = self.transformations(mask)

        image, mask = image.type(torch.FloatTensor), mask.type(torch.FloatTensor)
        return image, mask
    
    def __getitem__(self, index):
        image = np.load(self.image_paths[index])
        mask = np.load(self.mask_paths[index])
        image, mask = self.transform(image, mask)
        return image, mask

    def __len__(self):
        return len(self.image_paths)