import os
import numpy as np
import argparse
from tqdm import tqdm
import utils

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils
import cv2

from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte

from archs.restormer_arch import Restormer


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='/mnt/DATA/EE22B013/Btech_project/Model/Raw_data',type=str, help='input test image dir')
parser.add_argument('--results_dir',default='/mnt/DATA/EE22B013/Btech_project/Model/results', type=str, help='dir to save results')
parser.add_argument('--weights', default='/mnt/DATA/EE22B013/Btech_project/Model/experiments/Restormer_GOPRO/models/restormer_latest.pth',type=str, help='path to weights')
parser.add_argument('--dataset',default='Gopro', type=str, help='Dataset Name') # gopro,hide,realblur

args=parser.parse_args()

yaml_file='/mnt/DATA/EE22B013/Btech_project/Model/train_config.yml'

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


x=yaml.load(open(yaml_file,'r'),Loader=Loader)
s=x['network_g']

restored_model=Restormer(**x['network_g'])
checkpoint=torch.load(args.weights)
restored_model.load_state_dict(checkpoint['model'])
print(f'Testing using weights: {args.weights}')
restored_model.cuda()
#restored_model=torch.nn.DataParallel(restored_model)
restored_model.eval()


factor=8
dataset=args.dataset
results_dir=os.path.join(args.results_dir,dataset)
os.makedirs(results_dir,exist_ok=True)

inp_dir=os.path.join(args.input_dir,dataset,'test','blur')
input_files=natsorted(glob(os.path.join(inp_dir,'*.png'))+glob(os.path.join(inp_dir,'*.jpg')))

with torch.no_grad():
    for file in tqdm(input_files):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

       

        img = cv2.imread(file)  # BGR format by default
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
        img = np.float32(img) / 255.0
        img=torch.from_numpy(img).permute(2,0,1)
        img=img.unsqueeze(0).cuda()

        h,w=img.shape[2],img.shape[3]
        H,W=((h+factor)//factor*factor,(w+factor)//factor*factor)
        padh=H-h if h%factor!=0 else 0
        padw=W-w if w%factor!=0 else 0
        img=F.pad(img,(0,padw,0,padh),'reflect')

        out=restored_model(img)
        out=out[:,:,:h,:w]
        out=torch.clamp(out,0,1).cpu().detach().permute(0,2,3,1).squeeze(0).numpy()

        cv2.imwrite(os.path.join(results_dir, os.path.splitext(os.path.split(file)[-1])[0]+'.png'), cv2.cvtColor(img_as_ubyte(out), cv2.COLOR_RGB2BGR))