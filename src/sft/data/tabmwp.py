import os
import torch
import json
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from .base_module import BaseDataModule
import numpy as np
from datasets import load_dataset
import jsonlines
from qwen_vl_utils import process_vision_info







class TabMWPModule(BaseDataModule):
    class TabMWPDataset(torch.utils.data.Dataset):
        def __init__(self, config: dict):
            self.config = config
            self.data_file_path = self.config["data"]["data_file"]
            self.image_folder = self.config["data"]["image_folder"]
            self.data_file = self.read_jsonl(self.data_file_path)
            for item in self.data_file:
                item['image'] = os.path.join(self.image_folder, item['image'])
        

    
            

        def __len__(self):
            return len(self.data_file)
        
        def __getitem__(self, idx):
            sample = self.data_file[idx]
            return sample
        
        
        

        def read_jsonl(self, path):
            read_data = []
            with jsonlines.open(path) as reader:
                for obj in reader:
                    read_data.append(obj)
            return read_data
    
    
    def collate_fn(self, batch):
        assert len(batch) == 1
        sample = batch[0]
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample['image'],
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            },
            {"role": "assistant", "content": sample["text_description"]}
        ]
        text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        labels = inputs["input_ids"].clone()
        inputs["labels"] = labels
        
        
        return inputs
    
        




    def __init__(self, config: dict, processor):
        super().__init__()
        self.config = config
        self.processor = processor
        self.train_dataset = self.TabMWPDataset(config)
        
        
    