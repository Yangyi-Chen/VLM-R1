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

IGNORE_INDEX = -100
B_INST, E_INST = "[INST]", "[/INST]"


class LlavaPretrainingModule(BaseDataModule):
    class LlavaPretrainingDataset(torch.utils.data.IterableDataset):
        def __init__(self, config: dict, tokenizer):
            self.config = config
            self.tokenizer = tokenizer
            self.data_dir = '/shared/nas/data/m1/yangyic3/pre-training-scripts-analysis/filter_capfusion/'
            self.raw_data = load_dataset("/shared/nas/data/m1/yangyic3/pre-training-scripts-analysis/filter_capfusion/", split="train", streaming=True).shuffle(seed=32) 
            self.data = iter(self.raw_data)
            

        def __len__(self):
            return self.config['training']['max_steps'] * self.config['data']['batch_size']
        
        def __iter__(self):
            return self.data 
    
    def collate_fn(self, batch):
        collect_list = []
        for sample in batch:
            loss_map = sample['loss_map'] # list
            content = sample['content']
            content_bef = content[0]
            content_aft = content[1]
            if content_bef['type'] == 'text':
                caption = content_bef['text']
                image = content_aft['image_url']['url']
            else:
                caption = content_aft['text']
                image = content_bef['image_url']['url']
            image = load_base64_to_PILImage(image)
    
            question = self.general_caption_instruction_list[
                np.random.randint(0, len(self.general_caption_instruction_list))]
            
            input_label = self.tokenize_example([question, caption], loss_map)
            
            input_ids = input_label["input_ids"]
            labels = input_label["labels"]
            if "probs" in input_label:
                probs = input_label["probs"]
            else:
                probs = None
            
            length = input_ids.shape[0]

            save_dict = {
                "imgs": image,
                "orig_instruction": question,
                "length": length,
                "orig_instruction_tokenized": input_ids,
                "labels": labels,
                "probs": probs
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
            if batch[0]["probs"] is not None:
                probs = torch.nn.utils.rnn.pad_sequence(
                    [d["probs"] for d in batch],
                    batch_first=True,
                    padding_value=1
                )
            else:
                probs = None
                    
            return {
                "encoder_input_ids": encoder_input_ids,
                "encoder_input_lengths": encoder_input_lengths,
                "attention_mask": attention_mask,
                "decoder_target_ids": decoder_target_ids,
                "patch_images": patch_images,
                "probs": probs
            }
        except Exception as e:
            print(e)
            return None




    def __init__(self, config: dict, tokenizer):
        super().__init__()
        self.general_caption_instruction_list = [
                "Write a caption for this image."
                "Provide a description for the image.",
                "Compose a caption that fits this image.",
                "Write a brief explanation of what's shown in the image.",
                "Craft a short text to accompany the image.",
                "Generate a caption that matches the content of the image.",
                "Offer a textual representation of the image.",
                "Create a concise caption for the given picture.",
                "Construct a descriptive sentence for this image.",
                "Formulate a caption that suits this visual.",
                "Come up with a short explanation for the image.",
                "Produce a caption that captures the essence of the image.",
                "Write a few words that describe what you see in the image.",
                "Develop a caption to accompany the picture.",
                "Compose a text that relates to the content of the image.",
                "Summarize the image in a brief caption.",
                "Craft a descriptive sentence for this picture.",
                "Generate a caption that represents the image accurately.",
                "Offer a brief explanation of the visual content.",
                "Create a concise caption describing the image.",
                "Compose a caption that best fits the image."
            ]
        
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = self.LlavaPretrainingDataset(
            config, tokenizer
        )
        self.start_token = '\n'
        self.preprocess = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    
    
    
    