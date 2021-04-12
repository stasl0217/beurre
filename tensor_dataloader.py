from typing import *
import math
from torch.utils.data import Sampler, TensorDataset, DataLoader
import torch
######################################
# Custom Sampler
######################################

class TensorBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=False, drop_last=False):
        if not isinstance(data_source, TensorDataset):
            raise ValueError(f"data_source should be an instance of torch.utils.data.TensorDataset, but got data_source={data_source}")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if not isinstance(shuffle, bool):
            raise ValueError(f"shuffle should be a boolean value, but got shuffle={shuffle}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        if self.shuffle:
            idxs = torch.randperm(len(self.data_source)).split(self.batch_size)
        else:
            idxs = torch.arange(len(self.data_source)).split(self.batch_size)
        if self.drop_last and len(self.data_source) % self.batch_size != 0:
            idxs = idxs[:-1]
        return iter(idxs)

    def __len__(self):
        return (math.floor if self.drop_last else math.ceil)(len(self.data_source) / self.batch_size)

def unwrap_collate_fn(batch):
    return batch[0]

class TensorDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, *, drop_last=False, collate_fn=None, **kwargs):
        if sampler is not None or batch_sampler is not None or collate_fn is not None:
            raise ValueError("TensorDataLoader does not support alternate samplers, batch samplers, or collate functions.")
        sampler = TensorBatchSampler(dataset, batch_size, shuffle, drop_last)
        super().__init__(dataset, batch_size=1, shuffle=False, sampler=sampler,
                         drop_last=False, collate_fn = unwrap_collate_fn, **kwargs)
