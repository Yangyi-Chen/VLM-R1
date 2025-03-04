import torch
from torch.utils.data.distributed import DistributedSampler
import numpy as np




class BaseDataModule():
    def train_dataloader(self, distributed=True):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["data"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            collate_fn=self.collate_fn,
            pin_memory=True,
            # sampler=DistributedSampler(dataset=self.train_dataset) if distributed else None
        )