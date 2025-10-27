import os
import numpy as np
from tqdm import tqdm
from natsort import natsorted
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from glob import glob


restored_dir = '/mnt/DATA/EE22B013/Btech_project/Model/results/Gopro'  # folder with restored images
gt_dir = '/mnt/DATA/EE22B013/Btech_project/Model/Raw_data/Gopro/test/sharp'   # ground truth folder
extensions = ('*.png', '*.jpg')


# sort file lists
gt_files = natsorted(sum([glob(os.path.join(gt_dir, e)) for e in extensions], []))
restored_files = natsorted(sum([glob(os.path.join(restored_dir, e)) for e in extensions], []))

psnr_list, ssim_list = [], []
print("GT files:", len(gt_files))
print("Restored files:", len(restored_files))

for gt_path, res_path in tqdm(zip(gt_files, restored_files), total=len(gt_files), desc="Evaluating", ncols=100):
    gt_img = imread(gt_path).astype(np.float32) / 255.0
    res_img = imread(res_path).astype(np.float32) / 255.0

    psnr_val = compare_psnr(gt_img, res_img, data_range=1.0)
    ssim_val = compare_ssim(
                gt_img,
                res_img,
                channel_axis=2,  # for HWC images
                data_range=1.0,
                win_size=3
                )   


    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)

avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)

print(f"Average PSNR: {avg_psnr:.2f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")
