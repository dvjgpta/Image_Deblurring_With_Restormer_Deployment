# # train_restormer.py
# import os
# import time
# import datetime
# import torch
# import random
# import math

# from torch.optim import AdamW
# from schedulers import CosineAnnealingRestartCyclicLR
# from tqdm import tqdm
# from options import parse_options
# from logger import get_root_logger, init_tb_logger, init_wandb_logger
# from data.dataloader import create_train_val_loader
# from models.base_model import RestormerTrainer
# from archs.restormer_arch import Restormer  

# # ==========================
# # Main training script
# # ==========================
# def main():
#     # -----------------------------
#     # 1. Parse options & seeds
#     # -----------------------------
#     opt = parse_options(is_train=True)
#     torch.backends.cudnn.benchmark = True

#     seed = opt['manual_seed']
#     random.seed(seed)
#     torch.manual_seed(seed)

#     # -----------------------------
#     # 2. Initialize logger
#     # -----------------------------
#     log_file = os.path.join(opt['path']['log'], f"train_{opt['name']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
#     logger = get_root_logger(log_file=log_file)
#     logger.info(f"Training options:\n{opt}")

#     tb_logger, wb_logger = None, None
#     if opt['logger'].get('use_tb_logger', True):
#         tb_logger = init_tb_logger(log_dir=os.path.join('tb_logger', opt['name']))
#     if opt['logger'].get('wandb', {}).get('project'):
#         wb_logger = init_wandb_logger(opt)

#     # -----------------------------
#     # 3. Create Datasets & Dataloaders
#     # -----------------------------
#     train_loader, train_sampler, val_loader, total_epochs, total_iters = create_train_val_loader(opt, logger)

#     # -----------------------------
#     # 4. Create model
#     # -----------------------------
#     net_opt = opt['network_g']  # This is your Restormer-specific settings in YAML

#     restormer_net = Restormer(
#     inp_channels=int(net_opt['inp_channels']),
#     out_channels=int(net_opt['out_channels']),
#     dim=int(net_opt['dim']),
#     num_blocks=list(net_opt['num_blocks']),
#     num_refinement_blocks=net_opt['num_refinement_blocks'],
#     heads=list(net_opt['heads']),
#     ffn_expansion_factor=float(net_opt['ffn_expansion_factor']),
#     bias=bool(net_opt['bias']),
#     LayerNorm_type=str(net_opt['LayerNorm_type']),
#     dual_pixel_task=bool(net_opt.get('dual_pixel_task', False))
# )
    
#     optim_params = opt['train']['optim_g']
#     optimizer = AdamW(
#         restormer_net.parameters(),
#         lr=optim_params['lr'],
#         betas=optim_params['betas'],
#         weight_decay=optim_params['weight_decay']
#     )
    
#     sched_params = opt['train']['scheduler']
#     scheduler = CosineAnnealingRestartCyclicLR(
#         optimizer,
#         periods=sched_params['periods'],
#         restart_weights=sched_params['restart_weights'],
#         eta_mins=sched_params['eta_mins']
#     )

#     model = RestormerTrainer(restormer_net, opt, optimizer=optimizer, scheduler=scheduler)

#     start_epoch, current_iter = 0, 0

#     # Resume if checkpoint exists
#     if opt['path'].get('resume_state'):
#         checkpoint = torch.load(opt['path']['resume_state'])
#         model.load_state_dict(checkpoint['model'])
#         start_epoch = checkpoint['epoch']
#         current_iter = checkpoint['iter']
#         logger.info(f"Resuming from epoch {start_epoch}, iter {current_iter}")

#     # -----------------------------
#     # 5. Prefetcher
#     # -----------------------------
#     prefetch_mode = opt['datasets']['train'].get('prefetch_mode', 'cpu')
#     if prefetch_mode == 'cpu':
#         from data.prefetch_loader import CPUPrefetcher
#         prefetcher = CPUPrefetcher(train_loader)
#     elif prefetch_mode == 'cuda':
#         from data.prefetch_loader import CUDAPrefetcher
#         prefetcher = CUDAPrefetcher(train_loader, opt)
#     else:
#         raise ValueError(f'Invalid prefetch_mode: {prefetch_mode}')

#     # -----------------------------
#     # 6. Training loop with progressive learning
#     # -----------------------------
#     start_time = time.time()
#     epoch = start_epoch

#     # Precompute iteration groups for progressive learning
#     iters = opt['datasets']['train'].get('iters', [total_iters])
#     batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
#     mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes', [batch_size])
#     gt_size = opt['datasets']['train'].get('gt_size')
#     mini_gt_sizes = opt['datasets']['train'].get('gt_sizes', [gt_size])
#     groups = [sum(iters[0:i + 1]) for i in range(len(iters))]
#     logger_j = [True] * len(groups)

#     # pbar=tqdm(total=total_iters,desc="Training Progress", ncols=100)
#     while current_iter < total_iters:
#         if train_sampler:
#             train_sampler.set_epoch(epoch)
#         prefetcher.reset()
#         train_data = prefetcher.next()

#         # Count how many iterations are left in this epoch
#         epoch_iter_count = min(total_iters - current_iter, len(train_loader))

#         # Epoch-level tqdm
#         # with tqdm(total=epoch_iter_count, desc=f"Epoch {epoch}", ncols=120) as pbar_epoch:
#         while train_data is not None:
#                 current_iter += 1
#                 if current_iter > total_iters:
#                     break

#                 # ------------------
#                 # Progressive patch & batch
#                 # ------------------
#                 j = next((i for i, g in enumerate(groups) if current_iter <= g), len(groups)-1)
#                 mini_gt_size = mini_gt_sizes[j]
#                 mini_batch_size = mini_batch_sizes[j]

#                 # At the beginning of each stage
#                 if current_iter == groups[j]:  # stage just finished
#                     if j+1 < len(groups):  # next stage exists
#                         opt['datasets']['train']['gt_size'] = mini_gt_sizes[j+1]
#                         opt['datasets']['train']['batch_size_per_gpu'] = mini_batch_sizes[j+1]
#                         train_loader, train_sampler, _, _, _ = create_train_val_loader(opt, logger)
#                         prefetcher = CUDAPrefetcher(train_loader, opt)

#                 if logger_j[j]:
#                     logger.info(f"\nUpdating Patch_Size to {mini_gt_size} and Batch_Size to {mini_batch_size*torch.cuda.device_count()}\n")
#                     logger_j[j] = False

#                 # # Random crop for smaller patch size if needed
#                 # lq, gt = train_data['lq'], train_data['gt']
#                 # if mini_batch_size < batch_size:
#                 #     indices = random.sample(range(batch_size), mini_batch_size)
#                 #     lq = lq[indices]
#                 #     gt = gt[indices]

#                 # if mini_gt_size < gt_size:
#                 #     x0 = int((gt_size - mini_gt_size) * random.random())
#                 #     y0 = int((gt_size - mini_gt_size) * random.random())
#                 #     x1 = x0 + mini_gt_size
#                 #     y1 = y0 + mini_gt_size
#                 #     lq = lq[:, :, x0:x1, y0:y1]
#                 #     gt = gt[:, :, x0*opt['scale']:x1*opt['scale'], y0*opt['scale']:y1*opt['scale']]
#                 # Robust sampling + cropping using actual tensor sizes (avoid stale variables)
#                 lq, gt = train_data['lq'], train_data['gt']  # shapes: (B, C, H, W) and (B, C, H*scale, W*scale)

#                 # 1) sample mini-batch using actual batch size
#                 actual_bs = lq.size(0)
#                 if mini_batch_size < actual_bs:
#                     indices = random.sample(range(actual_bs), mini_batch_size)
#                     lq = lq[indices]
#                     gt = gt[indices]

#                 # 2) crop using actual spatial dims of lq
#                 #    make sure mini_gt_size fits in current lq spatial dims
#                 _, _, cur_h, cur_w = lq.shape
#                 if mini_gt_size < cur_h and mini_gt_size < cur_w:
#                     max_x = cur_h - mini_gt_size
#                     max_y = cur_w - mini_gt_size
#                     # use randint so inclusive range works; safe even if max == 0 is avoided above
#                     x0 = random.randint(0, max_x)
#                     y0 = random.randint(0, max_y)
#                     x1 = x0 + mini_gt_size
#                     y1 = y0 + mini_gt_size
#                     lq = lq[:, :, x0:x1, y0:y1]
#                     s = int(opt.get('scale', 1))
#                     gt = gt[:, :, x0 * s : x1 * s, y0 * s : y1 * s]
#                 else:
#                     # if the current lq is smaller than mini_gt_size, pad (or skip cropping)
#                     # simpler: pad symmetrically to mini_gt_size
#                     pad_h = max(0, mini_gt_size - cur_h)
#                     pad_w = max(0, mini_gt_size - cur_w)
#                     if pad_h > 0 or pad_w > 0:
#                         # pad on right and bottom
#                         lq = torch.nn.functional.pad(lq, (0, pad_w, 0, pad_h), mode='reflect')
#                         gt = torch.nn.functional.pad(gt, (0, pad_w * s, 0, pad_h * s), mode='reflect')
#                         # now crop top-left region of required size
#                         lq = lq[:, :, 0:mini_gt_size, 0:mini_gt_size]
#                         gt = gt[:, :, 0:mini_gt_size * s, 0:mini_gt_size * s]



#                 # Feed data
#                 model.feed_train_data({'lq': lq, 'gt': gt})
#                 train_loss = model.optimize_parameters(current_iter)
#                 lr = model.get_current_learning_rate()
                
#                 # # pbar_epoch.set_postfix({"Iter": current_iter, "Loss": f"{loss:.6f}", "LR": f"{model.get_current_learning_rate():.6f}"})
#                 # pbar_epoch.set_postfix({"Iter": current_iter})
#                 # pbar_epoch.update(1)

#                 # Logging, checkpointing, validation
#                 # if current_iter % opt['logger']['print_freq'] == 0:
#                 #     train_loss = model.get_current_loss()
#                 #     log_vars = {'epoch': epoch, 'iter': current_iter, 'lr': model.get_current_learning_rate(),}
#                 #     tqdm.write(f"[Epoch {epoch}][Iter {current_iter}] LR: {model.get_current_learning_rate():.10f}, Train Loss: {model.get_current_loss():.6f}")
#                 if current_iter % opt['logger']['print_freq'] == 0:
#                     val_loss = model.validate_loss_only(val_loader)
#                     msg = (f"[Epoch {epoch}][Iter {current_iter}] "
#                         f"LR: {lr:.8f}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
#                     print(msg)
#                     logger.info(msg)
#                 if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
#                     logger.info("Saving checkpoint...")
#                     save_dir = opt['path'].get('models', 'experiments/Restormer_GOPRO/models')
#                     os.makedirs(save_dir, exist_ok=True)  # create if not exists
#                     model.save_model(os.path.join(save_dir, f'restormer_iter_{current_iter}.pth'))


#                 if val_loader and current_iter % opt['val']['val_freq'] == 0:
#                     with torch.no_grad():
#                         val_loss, avg_psnr, avg_ssim = model.validate(val_loader)
#                         msg = (f"[Iter {current_iter}] Val Loss: {val_loss:.6f}, "
#                             f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
#                         print(msg)
#                         logger.info(msg)
#                         torch.cuda.empty_cache()

#                 train_data = prefetcher.next()
#         epoch += 1

#     # -----------------------------
#     # 7. Finish training
#     # -----------------------------
#     total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
#     logger.info(f"Training completed in {total_time}")
#     model.save_model(os.path.join(opt['path']['models'], 'restormer_latest.pth'))

#     if tb_logger:
#         tb_logger.close()


# if __name__ == "__main__":
#     main()
# train_restormer.py
import os
import time
import datetime
import torch
import random
import math

from torch.optim import AdamW
from schedulers import CosineAnnealingRestartCyclicLR
from tqdm import tqdm
from options import parse_options
from logger import get_root_logger, init_tb_logger, init_wandb_logger
from data.dataloader import create_train_val_loader
from models.base_model import RestormerTrainer
from archs.restormer_arch import Restormer  


# Main training script
def main():
    
    # 1. Parse options & seeds
    opt = parse_options(is_train=True)
    torch.backends.cudnn.benchmark = True

    seed = opt['manual_seed']
    random.seed(seed)
    torch.manual_seed(seed)

    # 2. Initialize logger
    log_file = os.path.join(opt['path']['log'], f"train_{opt['name']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    logger = get_root_logger(log_file=log_file)
    logger.info(f"Training options:\n{opt}")

    tb_logger, wb_logger = None, None
    if opt['logger'].get('use_tb_logger', True):
        tb_logger = init_tb_logger(log_dir=os.path.join('tb_logger', opt['name']))
    if opt['logger'].get('wandb', {}).get('project'):
        wb_logger = init_wandb_logger(opt)

    # 3. Create Datasets & Dataloaders
    train_loader, train_sampler, val_loader, total_epochs, total_iters = create_train_val_loader(opt, logger)

    # 4. Create model
    net_opt = opt['network_g']  

    restormer_net = Restormer(
    inp_channels=int(net_opt['inp_channels']),
    out_channels=int(net_opt['out_channels']),
    dim=int(net_opt['dim']),
    num_blocks=list(net_opt['num_blocks']),
    num_refinement_blocks=net_opt['num_refinement_blocks'],
    heads=list(net_opt['heads']),
    ffn_expansion_factor=float(net_opt['ffn_expansion_factor']),
    bias=bool(net_opt['bias']),
    LayerNorm_type=str(net_opt['LayerNorm_type']),
    dual_pixel_task=bool(net_opt.get('dual_pixel_task', False))
)
    
    optim_params = opt['train']['optim_g']
    optimizer = AdamW(
        restormer_net.parameters(),
        lr=optim_params['lr'],
        betas=optim_params['betas'],
        weight_decay=optim_params['weight_decay']
    )
    
    sched_params = opt['train']['scheduler']
    scheduler = CosineAnnealingRestartCyclicLR(
        optimizer,
        periods=sched_params['periods'],
        restart_weights=sched_params['restart_weights'],
        eta_mins=sched_params['eta_mins']
    )

    model = RestormerTrainer(restormer_net, opt, optimizer=optimizer, scheduler=scheduler)

    start_epoch, current_iter = 0, 0

    # Resume if checkpoint exists
    if opt['path'].get('resume_state'):
        checkpoint = torch.load(opt['path']['resume_state'])
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        current_iter = checkpoint['iter']
        logger.info(f"Resuming from epoch {start_epoch}, iter {current_iter}")

    
    # 5. Prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode', 'cpu')
    if prefetch_mode == 'cpu':
        from data.prefetch_loader import CPUPrefetcher
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        from data.prefetch_loader import CUDAPrefetcher
        prefetcher = CUDAPrefetcher(train_loader, opt)
    else:
        raise ValueError(f'Invalid prefetch_mode: {prefetch_mode}')

   
    # 6. Training loop with progressive learning
    start_time = time.time()
    epoch = start_epoch

    # Precompute iteration groups for progressive learning
    iters = opt['datasets']['train'].get('iters', [total_iters])
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes', [batch_size])
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes', [gt_size])
    groups = [sum(iters[0:i + 1]) for i in range(len(iters))]
    logger_j = [True] * len(groups)

    # Main training progress bar
    pbar_total = tqdm(
        total=total_iters, 
        desc="Training", 
        ncols=80, 
        position=0,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    pbar_total.update(current_iter)

    while current_iter < total_iters:
        if train_sampler:
            train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        # Count how many iterations are left in this epoch
        epoch_iter_count = min(total_iters - current_iter, len(train_loader))

        # Epoch-level progress bar
        pbar_epoch = tqdm(total=epoch_iter_count, desc=f"Epoch {epoch}", ncols=100, position=1, leave=False)

        while train_data is not None:
            current_iter += 1
            if current_iter > total_iters:
                break

            
            # Progressive patch & batch
            j = next((i for i, g in enumerate(groups) if current_iter <= g), len(groups)-1)
            mini_gt_size = mini_gt_sizes[j]
            mini_batch_size = mini_batch_sizes[j]

            # At the beginning of each stage
            if current_iter == groups[j]:  # stage just finished
                if j+1 < len(groups):  # next stage exists
                    opt['datasets']['train']['gt_size'] = mini_gt_sizes[j+1]
                    opt['datasets']['train']['batch_size_per_gpu'] = mini_batch_sizes[j+1]
                    train_loader, train_sampler, _, _, _ = create_train_val_loader(opt, logger)
                    prefetcher = CUDAPrefetcher(train_loader, opt)

            if logger_j[j]:
                logger.info(f"\nUpdating Patch_Size to {mini_gt_size} and Batch_Size to {mini_batch_size*torch.cuda.device_count()}\n")
                logger_j[j] = False

            lq, gt = train_data['lq'], train_data['gt']  # shapes: (B, C, H, W) and (B, C, H*scale, W*scale)

            # 1) sample mini-batch using actual batch size
            actual_bs = lq.size(0)
            if mini_batch_size < actual_bs:
                indices = random.sample(range(actual_bs), mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]

            # 2) crop using actual spatial dims of lq
            #    make sure mini_gt_size fits in current lq spatial dims
            _, _, cur_h, cur_w = lq.shape
            if mini_gt_size < cur_h and mini_gt_size < cur_w:
                max_x = cur_h - mini_gt_size
                max_y = cur_w - mini_gt_size
                # use randint so inclusive range works; safe even if max == 0 is avoided above
                x0 = random.randint(0, max_x)
                y0 = random.randint(0, max_y)
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq = lq[:, :, x0:x1, y0:y1]
                s = int(opt.get('scale', 1))
                gt = gt[:, :, x0 * s : x1 * s, y0 * s : y1 * s]
            else:
                # if the current lq is smaller than mini_gt_size, pad (or skip cropping)
                # simpler: pad symmetrically to mini_gt_size
                pad_h = max(0, mini_gt_size - cur_h)
                pad_w = max(0, mini_gt_size - cur_w)
                if pad_h > 0 or pad_w > 0:
                    # pad on right and bottom
                    lq = torch.nn.functional.pad(lq, (0, pad_w, 0, pad_h), mode='reflect')
                    gt = torch.nn.functional.pad(gt, (0, pad_w * s, 0, pad_h * s), mode='reflect')
                    # now crop top-left region of required size
                    lq = lq[:, :, 0:mini_gt_size, 0:mini_gt_size]
                    gt = gt[:, :, 0:mini_gt_size * s, 0:mini_gt_size * s]

            # Feed data
            model.feed_train_data({'lq': lq, 'gt': gt})
            train_loss = model.optimize_parameters(current_iter)
            lr = model.get_current_learning_rate()
            
            # # Update progress bars
            # pbar_epoch.update(1)
            # pbar_total.update(1)
            # Update progress bar only occasionally to reduce flickering
            if current_iter % 10 == 0:  # Update every 10 iterations
                pbar_total.update(10)

                # Update description with current metrics
                desc = f"Training [E{epoch} S{j}] Loss:{train_loss:.4f} LR:{lr:.2e}"
                pbar_total.set_description(desc)
                
            # Store latest metrics for logging
            last_train_loss = train_loss
            last_val_loss = 0  # Will be updated if validation runs
            last_lr = lr

            # Update progress bar descriptions with current metrics
            # pbar_epoch.set_postfix({
            #     "Loss": f"{train_loss:.4f}", 
            #     "LR": f"{lr:.2e}",
            #     "Stage": j
            # })
            # pbar_total.set_postfix({
            #     "Epoch": epoch,
            #     "Loss": f"{train_loss:.4f}",
            #     "LR": f"{lr:.2e}"
            # })

            # Logging, checkpointing, validation
            if current_iter % opt['logger']['print_freq'] == 0:
                val_loss = model.validate_loss_only(val_loader)
                last_val_loss = val_loss
                msg = (f"[Epoch {epoch}][Iter {current_iter}] "
                    f"LR: {lr:.8f}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                print(f"\n{msg}")  
                logger.info(msg)

            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info("Saving checkpoint...")
                save_dir = opt['path'].get('models', 'experiments/Restormer_GOPRO/models')
                os.makedirs(save_dir, exist_ok=True)  # create if not exists
                model.save_model(os.path.join(save_dir, f'restormer_iter_{current_iter}.pth'))

            if val_loader and current_iter % opt['val']['val_freq'] == 0:
                with torch.no_grad():
                    val_loss, avg_psnr, avg_ssim = model.validate(val_loader)
                    last_val_loss = val_loss
                    msg = (f"[Iter {current_iter}] Val Loss: {val_loss:.6f}, "
                        f"PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
                    print(f"\n{msg}")  
                    logger.info(msg)
                    torch.cuda.empty_cache()

            train_data = prefetcher.next()
        
        # Close epoch progress bar
        epoch += 1

    # Close main progress bar
    pbar_total.close()

   
    # 7. Finish training
    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f"Training completed in {total_time}")
    model.save_model(os.path.join(opt['path']['models'], 'restormer_latest.pth'))

    if tb_logger:
        tb_logger.close()


if __name__ == "__main__":
    main()