import torch
import torch.nn as nn
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from schedulers import CosineAnnealingRestartCyclicLR, ReduceLROnPlateau

# class RestormerTrainer:
#     def __init__(self, net, device='cuda', optimizer=None, loss_fn=None):
#         self.net = net.to(device)
#         self.device = device
#         self.optimizer = optimizer
#         self.loss_fn = loss_fn
#         self.is_train = True

#     def feed_train_data(self, data):
#         self.lq = data['lq'].to(self.device)
#         self.gt = data['gt'].to(self.device)

#     def optimize_parameters(self, current_iter=None):
#         self.net.train()
#         self.optimizer.zero_grad()
#         output = self.net(self.lq)
#         loss = self.loss_fn(output, self.gt)
#         loss.backward()
#         self.optimizer.step()
#         return loss.item()

#     def get_current_learning_rate(self):
#         return [group['lr'] for group in self.optimizer.param_groups]

#     def save(self, epoch=-1, current_iter=-1, path='./checkpoints'):
#         os.makedirs(path, exist_ok=True)
#         save_path = f"{path}/restormer_iter{current_iter}.pth"
#         torch.save(self.net.state_dict(), save_path)

    # -----------------------------
    # Local validation function
    # -----------------------------


class RestormerTrainer:
    def __init__(self, model, opt,optimizer=None, scheduler=None):
        self.model = model
        self.opt=opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.current_loss=0.0
        
        # ---- Optimizer ----
        if optimizer is None:
            optim_opt = opt['train']['optim_g']
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(optim_opt['lr']),
                betas=tuple(optim_opt.get('betas', (0.9, 0.999))),
                weight_decay=float(optim_opt.get('weight_decay', 0))
            )
        else:
            self.optimizer = optimizer
        
        # ---- Scheduler ----
        if scheduler is None:
            sched_opt = opt['train']['scheduler']
            self.scheduler = CosineAnnealingRestartCyclicLR(
                self.optimizer,
                periods=sched_opt['periods'],
                restart_weights=sched_opt.get('restart_weights', [1]*len(sched_opt['periods'])),
                eta_mins=sched_opt.get('eta_mins', [0]*len(sched_opt['periods']))
            )
        else:
            self.scheduler = scheduler
        # Using PyTorch CosineAnnealingLR as placeholder, implement your custom cyclic LR if needed
        # total_iters = opt['train']['total_iter']

        # # We'll create cumulative iteration boundaries
        # self.stage_iters = np.cumsum(opt['datasets']['train']['iters'])  # [92000, 156000, ...]

        # # Set up min LR per stage
        # eta_mins = sched_opt['eta_mins']  # e.g., [3e-4, 1e-6]

        # # Initialize LR array for each iteration
        # self.lr_array = np.zeros(total_iters, dtype=np.float32)

        # current_lr = float(eta_mins[0])  # starting LR
        # iter_pointer = 0
        # for stage_idx, stage_total in enumerate(opt['datasets']['train']['iters']):
        #     # Cycle 1 (fixed LR) for first stage if stage_idx==0
        #     if stage_idx == 0:
        #         self.lr_array[iter_pointer:iter_pointer+stage_total] = current_lr
        #     else:
        #         # Cosine decay for remaining stages
        #         T = stage_total
        #         eta_min = float(eta_mins[-1])
        #         for i in range(T):
        #             self.lr_array[iter_pointer + i] = eta_min + 0.5*(current_lr - eta_min)*(1 + np.cos(np.pi * i / T))
        #         current_lr = self.lr_array[iter_pointer + stage_total - 1]  # update for next stage
        #     iter_pointer += stage_total
        
        # ---- Loss ----
        pixel_opt = opt['train']['pixel_opt']
        self.criterion = nn.L1Loss(reduction=pixel_opt.get('reduction', 'mean'))
        self.loss_weight = pixel_opt.get('loss_weight', 1.0)
        
        # ---- Grad clip ----
        self.use_grad_clip = opt['train'].get('use_grad_clip', False)
        
    def feed_train_data(self, data):
        # Expect data dict: {'lq': tensor, 'gt': tensor}
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
    
    def optimize_parameters(self, current_iter=None):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self.lq)
        loss = self.criterion(output, self.gt) * self.loss_weight
        loss.backward()
        
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        # if current_iter is not None:
        #     lr = self.lr_array[current_iter-1]  # current_iter starts at 1
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = lr

        self.current_loss = loss.item()
        return self.current_loss
    def save_model(self, path):
        torch.save({'model': self.model.state_dict()}, path)

    

    def validate(self, val_loader, scale=1):
        self.model.eval()
        psnr_list, ssim_list, loss_list = [], [], []

        with torch.no_grad():
            for data in val_loader:
                lq = data['lq'].to(self.device)
                gt = data['gt'].to(self.device)

                output = self.model(lq)  # fixed typo
                loss = self.criterion(output, gt) * self.loss_weight
                loss_list.append(loss.item())

                output_np = output.clamp(0,1).cpu().numpy()
                gt_np = gt.clamp(0,1).cpu().numpy()

                for i in range(output_np.shape[0]):
                    out_img = np.transpose(output_np[i], (1,2,0))
                    gt_img = np.transpose(gt_np[i], (1,2,0))
                    # psnr_list.append(compare_psnr(gt_img, out_img, data_range=1.0))
                    psnr_list.append(calculate_psnr(gt_img, out_img, data_range=1.0))
                    ssim_list.append(compare_ssim(
                        gt_img, out_img,
                        channel_axis=2,       # replace multichannel
                        data_range=1.0,
                        win_size=min(7, min(gt_img.shape[:2]))  # ensure win_size <= image height/width
                    ))

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_loss = np.mean(loss_list)
        return avg_loss, avg_psnr, avg_ssim

    def validate_loss_only(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                lq = data['lq'].to(self.device)
                gt = data['gt'].to(self.device)
                output = self.model(lq)
                loss = self.criterion(output, gt) * self.loss_weight
                total_loss += loss.item()
        self.model.train()
        return total_loss / len(val_loader)

    
    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']
    def get_current_loss(self):
        return self.current_loss

# def calculate_psnr(img1, img2, border=0, data_range=255.0):
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')
#     h, w = img1.shape[:2]
#     img1 = img1[border:h-border, border:w-border]
#     img2 = img2[border:h-border, border:w-border]

#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     mse = np.mean((img1 - img2)**2)
#     if mse == 0:
#         return float('inf')
#     return 20 * math.log10(data_range / math.sqrt(mse))

class Nested_UNet_Trainer:
    def __init__(self,model,opt,optimizer=None,scheduler=None):
        self.model=model
        self.opt=opt
        self.device=torch.device('cuda'if torch.cuda.is_available()else'cpu')
        self.model.to(self.device)
        self.current_loss=0.0

        if optimizer is None:
            optim_opt=opt['train']['optim_g']
            self.optimizer=torch.optim.AdamW(
                self.model.parameters(),
                lr=float(optim_opt['lr']),
                betas=tuple(optim_opt.get('betas',(0.9,0.999))),
                weight_decay=float(optim_opt.get('weight_decay',0))
            )
        else:
            self.optimizer=optimizer
        
        if scheduler is None:
            sched_opt=opt['train']['scheduler']
            self.scheduler=ReduceLROnPlateau(
                self.optimizer,
                mode=sched_opt.get('mode','min'),
                factor=float(sched_opt.get('factor',0.5)),
                patience=int(sched_opt.get('patience',5)),
                threshold=float(sched_opt.get('threshold',1e-4)),
                min_lr=float(sched_opt.get('min_lr',1e-6)),
                verbose=bool(sched_opt.get('verbose',True))
            )
        else:
            self.scheduler=scheduler

        pixel_opt=opt['train']['pixel_opt']
        self.criterion=nn.L1Loss(reduction=pixel_opt.get('reduction','mean'))
        self.loss_weight=pixel_opt.get('loss_weight',1.0)

        self.use_grad_clip=opt['train'].get('use_grad_clip',False)

        def feed_train_data(self,data):
            self.lq=data['lq'].to(self.device)
            self.gt=data['gt'].to(self.device)

        def optimize_paramaters(self,current_iter=None):
            self.model.train()
            self.optimizer.zero_grad()
            output=self.model(self.lq)
            loss=self.criterion(output,self.gt)*self.loss_weight
            loss.backward()

            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=1.0)
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            self.current_loss=loss.item()
            return self.current_loss
        def save_model(self,path):
            torch.save({'model':self.model.state_dict()},path)

        def validate(self,val_loader,scale=1):
            self.model.eval()
            psnr_list,ssim_list,loss_list=[],[],[]

            with torch.no_grad():
                for data in val_loader:
                    lq=data['lq'].to(self.device)
                    gt=data['gt'].to(self.device)

                    output=self.model(lq)
                    loss=self.criterion(output,gt)*self.loss_weight
                    loss_list.append(loss.item())

                    output_np=output.clamp(0,1).cpu().numpy()
                    gt_np=gt.clamp(0,1).cpu().numpy()

                    for i in range(output_np.shape[0]):
                        out_img=np.transpose(output_np[i],(1,2,0))
                        gt_img=np.transpose(gt_np[i],(1,2,0))

                        psnr_list.append(calculate_psnr(gt_img,out_img,data_range=1.0))
                        ssim_list.append(compare_ssim(
                            gt_img,out_img,
                            channel_axis=2,
                            data_range=1.0,
                            win_size=min(7,min(gt_img.shape[:2]))
                        ))
            avg_psnr=np.mean(psnr_list)
            avg_ssim=np.mean(ssim_list)
            avg_loss=np.mean(loss_list)
            return avg_loss,avg_psnr,avg_ssim
        
        def validate_loss_only(self,val_loader):
            self.model.eval()
            total_loss=0.0
            with torch.no_grad():
                for data in val_loader:
                    lq=data['lq'].to(self.device)
                    gt=data['gt'].to(self.device)
                    output=self.model(lq)
                    loss=self.criterion(output,gt)*self.loss_weight
                    total_loss+=loss.item()
            self.model.train()
            return total_loss/len(val_loader)
    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']
    def get_current_loss(self):
        return self.current_loss



def calculate_psnr(img1, img2, data_range=1.0, eps=1e-10):
    """
    Robust PSNR calculation.
    - img1, img2: numpy arrays, same shape, float (expected in range [0, data_range]).
    - data_range: maximum possible pixel value (1.0 for normalized, 255.0 for uint8).
    - eps: small value to avoid divide-by-zero when mse==0.
    Returns PSNR in dB (float), or np.inf when mse==0.
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse <= 0:
        return float('inf')
    return 10.0 * np.log10((data_range ** 2) / (mse + eps))