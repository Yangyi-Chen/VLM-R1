import argparse
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import json
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Optional, Tuple


def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            # logger.warning_once(
            #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            #     "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
            #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
            #     "removed and `position_embeddings` will be mandatory."
            # )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward



def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--template", type=str, default="answer")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")

    args = parser.parse_args()

    template = args.template
    model_path = args.model_path

    IMAGE_FOLDER = "/scratch/azureml/cr/j/f01af20a3317416d9343927e368a55a6/exe/wd/PromptPG/data/tabmwp/tables"



    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    test_data = read_json("/blob/v-yangyi/data/data_files/tabmwp/problems_test1k.json")

    for k, item in tqdm(test_data.items()):
        image_path = os.path.join(IMAGE_FOLDER, k + ".png")
        qustion = item['question']
        table = item['table']
        answer = item['answer']


        messages = [
            {
            "role": "user",
            "content":  [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": "<image>"+ "Describe this image."},
                    ],
                },
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=1280)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text[0])





