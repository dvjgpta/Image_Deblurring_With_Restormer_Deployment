import math
import torch
from easydict import EasyDict


class CPUPrefetcher:
    """Prefetch batches from CPU dataloader."""
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = None

    def next(self):
        try:
            data = next(self.loader)
        except StopIteration:
            return None
        return data

    def reset(self):
        self.loader = iter(self.loader)


def move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    elif isinstance(data, dict) or isinstance(data, EasyDict):
        return type(data)({k: move_to_device(v, device) for k, v in data.items()})
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    else:
        return data  # leave other types unchanged



class CUDAPrefetcher:
    """Prefetch batches and move them to GPU asynchronously."""

    def __init__(self, loader, opt):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        """Load the next batch and move it to GPU asynchronously."""
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return None

        with torch.cuda.stream(self.stream):
            self.batch = move_to_device(self.batch, self.device)

    def next(self):
        """Return the current batch and preload the next one."""
        if self.batch is None:
            return None
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        """Reset the loader iterator."""
        self.loader = iter(self.ori_loader)
        self.preload()
