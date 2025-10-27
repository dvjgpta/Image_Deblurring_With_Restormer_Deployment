import math
from torch.optim.lr_scheduler import _LRScheduler

def get_position_from_periods(iteration, cumulative_periods):
    for i, cp in enumerate(cumulative_periods):
        if iteration < cp:
            return i
    return len(cumulative_periods) - 1

class CosineAnnealingRestartCyclicLR(_LRScheduler):
    def __init__(self, optimizer, periods, restart_weights=(1,), eta_mins=(0,), last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        self.cumulative_period = [sum(self.periods[0:i + 1]) for i in range(len(self.periods))]
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        return [
            eta_min + current_weight * 0.5 * (base_lr - eta_min) *
            (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]

class ReduceLROnPlateau(_LRScheduler):
    def __init__(self,optimizer,mode='min',factor=0.5,patience=5,threshold=1e-4,min_lr=1e-6,verbose=True):
        self.optimizer=optimizer
        self.scheduler=ReduceLROnPlateau(
            optimizer=optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr,
            verbose=verbose
        )
        self.last_epoch=-1
    
    def step(self,metric):
        self.last_epoch+=1
        self.scheduler.step(metric)

    def get_last_lr(self):
        #return last computed learning rate by scheduler
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        # return state dict of scheduler
        return self.scheduler.state_dict()
    
    def load_state_dict(self,state_dict):
        #load state dict to scheduler
        self.scheduler.load_state_dict(state_dict)