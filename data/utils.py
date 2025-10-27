import torch
import numpy as np

def img2tensor(img, bgr2rgb=False, float32=True):
    single_input = False
    # Wrap single image into a list
    if not isinstance(img, list):
        imgs = [img]  # fixed: use img, not imgs
        single_input = True  # fixed typo
    else:
        imgs = img

    tensor_list = []
    for im in imgs:
        if bgr2rgb and im.shape[2] == 3:
            im = im[:, :, ::-1]  # BGR to RGB

        im = np.transpose(im, (2, 0, 1))  # HWC to CHW
        if float32:
            im = im.astype(np.float32)/255.0

        tensor_list.append(torch.from_numpy(im))

    if single_input:
        return tensor_list[0]
    else:
        return tensor_list


    
def normalize(tensor,mean,std,inplace=False):

    if not inplace:
        tensor=tensor.clone()

    for t,m,s in zip(tensor,mean,std):
        t.sub_(m).div_(s)

    return tensor
