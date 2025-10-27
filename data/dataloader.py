import math
import random
import torch
from torch.utils.data import DataLoader
from data.gopro_dataset import PairedDataset
from data.data_sampler import EnlargedSampler
from data.prefetch_loader import CPUPrefetcher,CUDAPrefetcher

#create train and val dataloaders for paired image restoration
def create_train_val_loader(opt,logger):

    train_loader, val_loader=None,None
    train_sampler=None

    for phase, dataset_opt in opt['datasets'].items():
        if phase=='train':
            dataset_enlarge_ratio=dataset_opt.get('dataset_enlarge_ratio',1)
            # train_set = PairedDataset(
            #     lq_root=dataset_opt['dataroot_lq'],
            #     gt_root=dataset_opt['dataroot_gt'],
            #     crop_size=dataset_opt.get('crop_size', 256),
            #     augment=dataset_opt.get('augment', True),
            #     mean=dataset_opt.get('mean', None),
            #     std=dataset_opt.get('std', None)
            # )
            # stage_idx = opt['current_stage']  # you need to track this in your main loop
            crop_size = dataset_opt.get('gt_size', 256)
            batch_size = dataset_opt.get('batch_size_per_gpu', 16)

            train_set = PairedDataset(
                lq_root=dataset_opt['dataroot_lq'],
                gt_root=dataset_opt['dataroot_gt'],
                crop_size=dataset_opt.get('gt_size', 256),
                augment=dataset_opt.get('augment', True),
                mean=dataset_opt.get('mean', None),
                std=dataset_opt.get('std', None)
            )



            # train_sampler= EnlargedSampler(
            #     train_set,
            #     num_replicas=opt['world_size'],
            #     rank=opt['rank'],
            #     enlarge_ratio=dataset_enlarge_ratio
            # )
            train_sampler = EnlargedSampler(
                train_set,
                enlarge_ratio=dataset_enlarge_ratio
            )

            train_loader=DataLoader(
                train_set,
                batch_size=dataset_opt['batch_size_per_gpu'],
                sampler=train_sampler if opt['dist'] else None,
                shuffle=(train_sampler is None),
                num_workers=dataset_opt.get('num_workers',8),
                pin_memory=True,
                drop_last=True
            )

            num_iter_per_epoch=math.ceil(
                len(train_set)* dataset_enlarge_ratio/
                (dataset_opt['batch_size_per_gpu'] * opt['world_size'])
            )

            total_iters = int(opt['train']['total_iter'])
            total_epochs=math.ceil(total_iters/num_iter_per_epoch)

            logger.info(
                f"Training Dataset: {len(train_set)} images\n"
                f"Dataset enlarge ratio: {dataset_enlarge_ratio}\n"
                f"Batch size per GPU: {dataset_opt['batch_size_per_gpu']}\n"
                f"World size (GPU number): {opt['world_size']}\n"
                f"Iters per epoch: {num_iter_per_epoch}\n"
                f"Total epochs: {total_epochs}, Total iters: {total_iters}"
            )

        elif phase =='val':
            val_set = PairedDataset(
                lq_root=dataset_opt['dataroot_lq'],
                gt_root=dataset_opt['dataroot_gt'],
                crop_size=dataset_opt.get('crop_size', 256),
                augment=False,
                mean=dataset_opt.get('mean', None),
                std=dataset_opt.get('std', None)
            )
            val_loader=DataLoader(
                val_set,
                batch_size=1,
                shuffle=False,
                num_workers=dataset_opt.get('num_workers',8),
                pin_memory=True
            )
            logger.info(f"Validation Dataset: {len(val_set)} images")

        else:
            raise ValueError(f"Dataset phase {phase} not recognized.")

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


def create_prefetcher(loader,mode='cpu',opt=None):
    if mode=='cpu' or mode is None:
        prefetcher=CPUPrefetcher(loader)
    elif mode=='cuda':
        prefetcher = CUDAPrefetcher(loader,opt)
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError("pin_memory must be True for CUDAPrefetcher")
    else:
        raise ValueError(f"Unknown prefetcher mode: {mode}")
    return prefetcher