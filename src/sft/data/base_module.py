import torch
from torch.utils.data.distributed import DistributedSampler
from .image_utils import load_base64_to_PILImage
import numpy as np


IGNORE_INDEX = -100
class BaseDataModule():
    def tokenize_example(self, example, prob=None):
        # example: ['describe image', 'xx xxxx xxx xxxx']
        end_token = self.tokenizer.eos_token
        tags = [i for _ in range(len(example) // 2) for i in ["User", "Assistant"]]
        labels = []
        tokenized_ids = []
        probs = []
        # add 577 [1] to the beginning of the probs as dummy
        probs += [1] * 577
        for i, c in enumerate(example):
            if i % 2 == 1:
                # model
                c_input = self.start_token + tags[i] + ": "
                tokenized = self.tokenizer(c_input, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += [IGNORE_INDEX] * len(tokenized["input_ids"])
                probs += [1] * len(tokenized["input_ids"])
                
                c_generate = c + end_token
                tokenized = self.tokenizer(c_generate, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += tokenized["input_ids"]
                if prob is not None:
                    assert len(tokenized["input_ids"]) == len(prob)
                    probs += prob
            else:
                # user
                if i == 0:
                    c_new = self.tokenizer.bos_token + tags[i] + ": " + c + end_token
                else:
                    c_new = self.start_token + tags[i] + ": " + c + end_token
                tokenized = self.tokenizer(c_new, add_special_tokens=False)
                tokenized_ids += tokenized["input_ids"]
                labels += [IGNORE_INDEX] * len(tokenized["input_ids"])
                probs += [1] * len(tokenized["input_ids"])
                
                
        if prob is not None:
            assert len(tokenized_ids) == len(labels)
            return {"input_ids": torch.LongTensor(tokenized_ids), "labels": torch.LongTensor(labels), "probs": torch.FloatTensor(probs)}
        else:
            assert len(tokenized_ids) == len(labels)
            return {"input_ids": torch.LongTensor(tokenized_ids), "labels": torch.LongTensor(labels)}
        



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