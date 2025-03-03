import os
import torch
import json
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from .base_module import BaseDataModule
import numpy as np
from datasets import load_dataset
from .image_utils import load_base64_to_PILImage
from transformers import CLIPImageProcessor
import random
import jsonlines
from datasets import Dataset


IGNORE_INDEX = -100
B_INST, E_INST = "[INST]", "[/INST]"


class LlavaSFTModule(BaseDataModule):
    class LlavaSFTDataset(torch.utils.data.IterableDataset):
        def read_jsonlines(self, path):
            with jsonlines.open(path) as reader:
                data = [item for item in reader]
            return data
    
    
        def __init__(self, config: dict, tokenizer):
            self.config = config
            self.tokenizer = tokenizer
             
            
            self.visual_data = self.read_jsonlines("/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/raw_data/the_cauldron/all_data.jsonl")
            self.sharegpt4v = self.read_jsonlines('/shared/nas/data/m1/yangyic3/MultimodalAgent/data/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k_filter.jsonl')
            self.gpt4laion = self.read_jsonlines("/shared/nas/data/m1/yangyic3/MultimodalAgent/data/gpt4laion/gpt4laion_filter.jsonl") # yes
            self.lvis_instruct = self.read_jsonlines("/shared/nas/data/m1/yangyic3/MultimodalAgent/data/LVIS-Instruct4V/lvis_instruct4v_220k.jsonl")
            self.gqa = self.read_jsonlines("/shared/nas/data/m1/yangyic3/MultimodalAgent/data/gqa.jsonl")
            self.textocr_gpt4v = self.read_jsonlines("/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/raw_data/textocr/textocr-gpt4v/textocr_gpt4v.jsonl")
            self.llavar = self.read_jsonlines("/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/raw_data/LLaVAR/chat_llavar_filtered.jsonl")
            self.pixmo_askanything = self.read_jsonlines("/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/raw_data/pixmo/pixmo.jsonl") 
            
            for item in self.visual_data:
                item['image'] = "/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/raw_data/the_cauldron/images/" + item['image']
            for item in self.sharegpt4v:
                item['image'] = "/shared/nas/data/m1/yangyic3/MultimodalAgent/data/" + item['image']
                item['source'] = "sharegpt4v"
            for item in self.gpt4laion:
                item['image'] = "/shared/nas/data/m1/yangyic3/MultimodalAgent/data/" + item['image']
                item['source'] = "gpt4laion"
            for item in self.lvis_instruct:
                item['image'] = "/shared/nas/data/m1/yangyic3/MultimodalAgent/data/" + item['image']
                item['source'] = "lvis_instruct"
            for item in self.gqa:
                item['image'] = "/shared/nas/data/m1/yangyic3/MultimodalAgent/data/" + item['image']
                item['source'] = "gqa"
            for item in self.textocr_gpt4v:
                item['source'] = "textocr_gpt4v"
                item['image'] = "/shared/nas/data/m1/yangyic3/Multimodal-Mistral/" + item['image']
            for item in self.pixmo_askanything:
                item['image'] = "/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/raw_data/" + item['image']
                item['source'] = "pixmo_askanything"
            
            ds = self.visual_data + self.gqa + self.textocr_gpt4v + self.sharegpt4v + self.gpt4laion + self.lvis_instruct + self.pixmo_askanything
            random.seed(32)
            random.shuffle(ds)
                
            # load into huggingface dataset 
            TOTAL_NUM = 4000
            ds_dict = {"id": [], "image": [], "conversations": []}
            count = 0
            for idx, item in enumerate(ds):
                image_path = item['image']
                # check if image exists
                if not os.path.exists(image_path):
                    continue
                ds_dict["id"].append(idx)
                ds_dict["conversations"].append(item['conversations'])
                ds_dict["image"].append(image_path)
                count += 1 
                if count == TOTAL_NUM:
                    break
            
            self.raw_data = Dataset.from_dict(ds_dict) 
            self.data = iter(self.raw_data)
               
        
        def __len__(self):
            return self.config['training']['max_steps'] * self.config['data']['batch_size']
        
        def __iter__(self):
            return self.data 
    
    
    def collate_fn(self, batch):
        collect_list = []
        for sample in batch:
            image_path = sample['image']
            conversation = sample['conversations']
            # read the image
            image = Image.open(image_path).convert("RGB")
            conversation = [item['value'] for item in conversation]
            
            input_label = self.tokenize_example(conversation)
            
            input_ids = input_label["input_ids"]
            labels = input_label["labels"]
            length = input_ids.shape[0]
            
            save_dict = {
                "imgs": image,
                "length": length,
                "orig_instruction_tokenized": input_ids,
                "labels": labels,
            }
            collect_list.append(save_dict)
        
        
        batch = collect_list
            
        try:
            encoder_input_ids = torch.nn.utils.rnn.pad_sequence(
                [d["orig_instruction_tokenized"] for d in batch],
                batch_first=True,
                padding_value=0
            )
            encoder_input_lengths = [d["length"] for d in batch]
        

            attention_mask = encoder_input_ids.ne(0)
            decoder_target_ids = torch.nn.utils.rnn.pad_sequence(
                [d["labels"] for d in batch],
                batch_first=True,
                padding_value=IGNORE_INDEX
            )

            patch_images = [d['imgs'] for d in batch]
            preprocess_images = self.preprocess.preprocess(patch_images, return_tensors='pt')
            patch_images = preprocess_images['pixel_values'] # batch_size, 3, 336, 336


                    
            return {
                "encoder_input_ids": encoder_input_ids,
                "encoder_input_lengths": encoder_input_lengths,
                "attention_mask": attention_mask,
                "decoder_target_ids": decoder_target_ids,
                "patch_images": patch_images,
                "probs": None
            }
        except Exception as e:
            print(e)
            return None




    



    def __init__(self, config: dict, tokenizer):
        super().__init__()
        
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = self.LlavaSFTDataset(
            config, tokenizer
        )
        self.start_token = '\n'
        self.preprocess = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    
    
    
    