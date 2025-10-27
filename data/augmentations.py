import random
import numpy as np
import cv2

# img_path='/mnt/DATA/EE22B013/Btech_project/Model/patches/gopro/train/blurred/0-4.png'
# img=cv2.imread(img_path)

# if img is not None:
#     height,width,_ = img.shape
#     print(f"IMage dimensions: {height} x {width}")

# else:
#     print("Error: Could not read the image.")


# mod crop function to match dimensions images while upsampling or downsampling like if img dimensons is 563,259,3 and scale is 4 so it will make it divisible by 4
def mod_crop(img,scale):
    h,w=img.shape[:2]
    h_remainder, w_remainder=h%scale, w%scale
    return img[:h_remainder,:w_remainder]

#generates random crops paired out of the patches to diversify
def paired_random_crop(lq_img,gt_img,gt_patch_size,scale):
    h,w = lq_img.shape[:2]
    lq_patch_size=gt_patch_size//scale

    top=random.randint(0,h-lq_patch_size)
    left=random.randint(0,w-lq_patch_size)

    lq_patch=lq_img[top:top+lq_patch_size,left:left+lq_patch_size,:]
    gt_patch=gt_img[top*scale:top*scale +gt_patch_size, left*scale:left*scale + gt_patch_size, :]

    return lq_patch, gt_patch

#apply random flip and rotation 90
def augment(lq, gt, hflip=True, vflip=True, rot=True):
    if hflip and random.random() < 0.5:
        lq = lq[:, ::-1, :]
        gt = gt[:, ::-1, :]
    if vflip and random.random() < 0.5:
        lq = lq[::-1, :, :]
        gt = gt[::-1, :, :]
    if rot and random.random() < 0.5:
        lq = lq.transpose(1,0,2)
        gt = gt.transpose(1,0,2)
    return lq, gt


