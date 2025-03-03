from PIL import Image
import torch
IGNORE_INDEX = -100
from typing import List
import warnings
warnings.filterwarnings("ignore")
import torch
from torch import nn
from transformers import CLIPModel, AutoTokenizer, LlamaForCausalLM
from torch.nn import functional as F




    
class Llava(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gemma = config['model']['gemma']
        self.vision_language_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir="./").vision_model
        self.tokenizer = AutoTokenizer.from_pretrained("/shared/nas/data/m1/yangyic3/pre-training-scripts-analysis/cache/Llama-3.2-3B/")
        self.language_model = LlamaForCausalLM.from_pretrained("/shared/nas/data/m1/yangyic3/pre-training-scripts-analysis/cache/Llama-3.2-3B/", torch_dtype=torch.bfloat16)
      
        self.mlp = self.build_mlp_projection()           
        self.set_require_grad(mode=0)
    
      
        
    
    def build_mlp_projection(self):
        self.linear1 = nn.Linear(1024, 4096)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(4096, 3072)
        return nn.Sequential(self.linear1, self.activation, self.linear2)
    
    
        

    def set_require_grad(self, mode=0):
        if mode == 0:
            for param in self.language_model.parameters():
                param.requires_grad = False
            for param in self.vision_language_model.parameters():
                param.requires_grad = False
            for param in self.mlp.parameters():
                param.requires_grad = True
        
            
    def forward(self, batch, device):
 
        encoder_input_ids = batch["encoder_input_ids"]
        attention_mask = batch['attention_mask']
        decoder_target_ids = batch["decoder_target_ids"]
        patch_images = batch["patch_images"]
        probs = batch["probs"] # shape: batch_size, MAX_LENGTH
        
        inputs_embeds = self.language_model.get_input_embeddings()(encoder_input_ids)  
        vision_output = self.vision_language_model(patch_images).last_hidden_state # TOTAL_BATCH_SIZE, 577, 1024

        mapping_output = self.mlp(vision_output)  # batch_size, 577, 3072
        inputs_embeds = torch.cat([mapping_output, inputs_embeds], dim=1)  # batch_size, 577 + max_len, 2048
        decoder_target_ids = torch.cat([torch.ones_like(mapping_output[:, :, 0], dtype=torch.long) * -100, decoder_target_ids], dim=1)

        attention_mask = torch.cat([torch.ones_like(mapping_output[:, :, 0], dtype=torch.long), attention_mask], dim=1)
        
        if probs is None: 
            outputs = self.language_model(
                inputs_embeds=inputs_embeds.bfloat16(),
                attention_mask=attention_mask,
                labels=decoder_target_ids
            )
            return outputs['loss']
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds.bfloat16(),
                attention_mask=attention_mask,
            )
            pred = outputs.logits[:, :-1, :]
            labels = decoder_target_ids[:, 1:]
            probs = probs[:, 1:]
            original_loss_avg = F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))
            
            loss_all = F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1), reduction='none', ignore_index=-100)    # batch_size * max_len 
            # loss_ref = -torch.log(prob)
            salient = 1 - probs 
            # flatten 
            salient = salient.flatten()
            # apply the GEOMMA
            salient = salient ** self.gemma
            loss_all_ref = salient * loss_all
            loss_all_ref = loss_all_ref[loss_all_ref != 0.0]
            
            loss_all_ref_avg = torch.mean(loss_all_ref)
            scale = original_loss_avg / loss_all_ref_avg
            # convert the scale to a scalar without gradient
            scale = scale.detach()
            final_loss = torch.mean(loss_all_ref * scale) 
            return final_loss
    
    
    
    
    
    def generate(self, raw_img, encoder_input_ids, encoder_input_lengths, **generate_kwargs):
       
        
        vision_output = self.vision_language_model(raw_img, 'cuda')
        mapping_output = self.linear(vision_output)  # batch_size, 32, 2048
        inputs_embeds = self.language_model.get_input_embeddings()(encoder_input_ids.cuda())  # batch_size, max_len, 2048
        mask_emb = self.language_model.get_input_embeddings()(
            torch.tensor([[self.tokenizer.pad_token_id]]).cuda())
        total_length = mapping_output.size(1) + inputs_embeds.size(1)

        
        input_emb_li = []
        for i in range(inputs_embeds.shape[0]):
            single_mapping_output = mapping_output[i][:mapping_output.shape[1], :]  # 9, 2048
            single_inputs_embed = inputs_embeds[i][:encoder_input_lengths[i], :]  # 12, 2048
            if total_length - mapping_output.shape[1] - encoder_input_lengths[i] == 0:
                single_input_emb = torch.cat([single_mapping_output, single_inputs_embed], dim=0)
            else:
                concat_mask_embedding = torch.cat(
                    [mask_emb] * (total_length - mapping_output.shape[1] - encoder_input_lengths[i]),
                    dim=0).squeeze()  # length_, 2048
                # print(concat_mask_embedding.shape, single_mapping_output.shape, single_inputs_embed.shape)
                if concat_mask_embedding.shape[0] == 5120:
                    concat_mask_embedding = concat_mask_embedding.unsqueeze(0)
                # print(concat_mask_embedding.shape, single_mapping_output.shape, single_inputs_embed.shape)
                single_input_emb = torch.cat([concat_mask_embedding, single_mapping_output, single_inputs_embed],
                                             dim=0)  # total_length, 2048

            input_emb_li.append(single_input_emb)
        inputs_embeds = torch.stack(input_emb_li, dim=0)  # 16, total_length, 2048
        attention_mask = torch.ones_like(inputs_embeds[:, :, 0], dtype=torch.long, device='cuda')
        for i in range(attention_mask.shape[0]):
            if mapping_output.shape[1] + encoder_input_lengths[i] == total_length:
                continue
            attention_mask[i][ :total_length - mapping_output.shape[1] - encoder_input_lengths[i]] = 0

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds.cuda().bfloat16(),
            attention_mask=attention_mask.cuda(),
            **generate_kwargs,
        )
        return outputs, total_length



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config["training"]["optimizer"]["lr"],
            weight_decay=self.config["training"]["optimizer"]["weight_decay"],
        )
        return optimizer