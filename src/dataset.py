import os
import cv2
import numpy as np

import torch
from torchvision import transforms
import torch.nn.functional as F


# CIFAR IMAGE CONSTANTS(RGB order)
MEAN = (125.3/255, 123.0/255, 113.9/255)
STD = (63.0/255, 62.1/255, 66.7/255)

class Dataset(torch.utils.data.Dataset):
    """
    Create the training dataset and test dataset, based on the data_path and the split of "Train" or "Validation"
    """
    def __init__(self, data_path, opt, split='Train'):
        inputs=np.load(os.path.join(data_path, 'data{}.npy'.format(split)))
        labels = np.load(os.path.join(data_path, 'label{}.npy'.format(split)))
        self.opt = opt
        self.inputs = torch.from_numpy(inputs).float()
        self.labels = torch.from_numpy(labels).long()
        self.split = split
        self.data_path = data_path
        self.do_augment = (split=='Train' and opt.doAugmentaion)
        if self.do_augment:
            #Do normalization and augmentation on the data
            self.augmentation_norm = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=0.5),
#                    transforms.RandomResizedCrop(32, scale=(0.9, 1.0), ratio=(0.9, 1.1), interpolation=2),
#                    transforms.RandomPerspective(0.2),
                    transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1), fillcolor=0),
#                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
            #        transforms.RandomErasing(p=0.5, scale=(0.005, 0.01), ratio=(0.5, 2.0), value=0, inplace=False),
#                    transforms.Normalize(MEAN,STD)
                ])
        else:
            #Do only normalization on the data
            self.norm = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
#                    transforms.Normalize(MEAN,STD)
                ])
        
        print('Loaded {} images for {}'.format(len(self.inputs), split))
        
    def __getitem__(self, index):
        inp, label = self.inputs[index], self.labels[index]
        if self.do_augment:
            inp = self.augmentation_norm(inp)
        else:
            inp = self.norm(inp)        
        return inp, label

    def __len__(self):
        return len(self.inputs)