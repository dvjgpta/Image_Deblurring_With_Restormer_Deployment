import os
import cv2
import numpy as np
import torch
import glob
from torch.utils.data import Dataset
from data.augmentations import paired_random_crop, augment
from data.utils import img2tensor, normalize

# general class for a paired dataset
class PairedDataset(Dataset):

    def __init__(self,lq_root,gt_root,crop_size=256,scale=1,augment=True,mean=None,std=None):
        self.scale=scale
        self.lq_files=sorted(glob.glob(os.path.join(lq_root,'*.png')))
        self.gt_files=sorted(glob.glob(os.path.join(gt_root,'*.png')))

        assert len(self.lq_files)==len(self.gt_files), "Mismatch LQ/GT files"
        self.crop_size=crop_size
        self.mean=mean
        self.augment=augment
        self.std=std

    def __len__(self):
        return len(self.lq_files)
    
    def __getitem__(self, idx):
        lq_path=self.lq_files[idx]
        gt_path=self.gt_files[idx]

        lq= cv2.imread(lq_path)[:,:,::-1] #BGR TO RGB
        gt= cv2.imread(gt_path)[:,:,::-1]

        #paired random crop
        lq,gt = paired_random_crop(lq,gt,self.crop_size,scale=self.scale)

        #augment
        if self.augment:
            lq, gt = augment(lq, gt, hflip=True, vflip=True, rot=True)


        #tensor conversion
        lq,gt=img2tensor([lq,gt])

        #normalization
        if self.mean is not None or self.std is not None:
            normalize(lq,self.mean,self.std)
            normalize(gt,self.mean,self.std)

        return {'lq':lq,   'gt': gt , 'lq_path':lq_path}