import math
import torch
from torch.utils.data import Sampler

class EnlargedSampler(Sampler):
    """Sampler that enlarges a dataset for iteration-based training.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset to sample from.
        world_size (int): Number of processes (for DDP), 1 for single GPU.
        rank (int): Rank of the current process.
        enlarge_ratio (int): Factor to virtually enlarge the dataset.
        shuffle (bool): Whether to shuffle the dataset each epoch.
    """
    def __init__(self, dataset, world_size=1, rank=0, enlarge_ratio=1, shuffle=True):
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        self.enlarge_ratio = enlarge_ratio
        self.shuffle = shuffle
        self.num_samples = int(math.ceil(len(dataset) * enlarge_ratio / world_size))
        self.total_size = self.num_samples * world_size

    def __iter__(self):
        indices = list(range(len(self.dataset))) * self.enlarge_ratio
        if self.shuffle:
            torch.manual_seed(torch.initial_seed())
            indices = torch.randperm(len(indices)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample for this rank
        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """For reproducibility in DDP (optional)."""
        self.epoch = epoch
