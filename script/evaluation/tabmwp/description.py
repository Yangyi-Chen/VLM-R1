import argparse
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import json
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import torch
from typing import Optional, Tuple
from Levenshtein import ratio
from statistics import mean as average
import re

def clean_text(text, exclue_chars=['\n', '\r']):
    # Extract content between <answer> and </answer> if present
    if template == 'answer':
        answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    elif template == 'visual':
        answer_matches = re.findall(r'<visual>(.*?)</visual>', text, re.DOTALL)
    else:
        raise ValueError(f"Unknown template: {template}")
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]
    
    for char in exclue_chars:
        if char in ['\n', '\r']:
            # If there is a space before the newline, remove the newline
            text = re.sub(r'(?<=\s)' + re.escape(char), '', text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r'(?<!\s)' + re.escape(char), ' ', text)
        else:
            text = text.replace(char, ' ')
    
    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip('.').lower()



def batch_generate(test_data, processor, model, image_folder, batch_size):
    # Get all test keys
    all_keys = list(test_data.keys())

    metric_list = []
    # Process in batches
    for i in tqdm(range(0, len(all_keys), batch_size)):
        batch_keys = all_keys[i:i+batch_size]
        batch_messages = []
        target_list = []

        # Prepare batch inputs
        for k in batch_keys:
            image_path = os.path.join(image_folder, k + ".png")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": "<image>Describe this image."},
                    ],
                }
            ]
            batch_messages.append(messages)
            target_list.append(test_data[k]["table"])

        
        # Process batch
        batch_text = []
        batch_image_inputs = []
        # batch_video_inputs = []
        
        for messages in batch_messages:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            batch_text.append(text)
            batch_image_inputs.extend(image_inputs)
            # batch_video_inputs.extend(video_inputs)
        
        # Create inputs for the whole batch
        inputs = processor(
            text=batch_text,
            images=batch_image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            padding_side="left"
        )
        inputs = inputs.to("cuda")
        
        # Generate for the whole batch
        with torch.no_grad():  # Add this to save memory during inference
            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)


        # Process outputs for each item in the batch
        for j, k in enumerate(batch_keys):
            # Get the input and output IDs for this batch item
            in_ids = inputs.input_ids[j]
            out_ids = generated_ids[j]
            
            # Trim the input IDs from the output
            generated_ids_trimmed = out_ids[len(in_ids):]
            
            # Decode to get the output text
            output_text = processor.batch_decode(
                [generated_ids_trimmed], 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            # Store the result
            output_text = clean_text(output_text)

            # compute the metric
            # print(output_text)
            # print("*"*50)
            # print(clean_text(target_list[j]))
            # print("---"*50)
            metric_list.append(ratio(output_text, clean_text(target_list[j])))
        print(average(metric_list))
    return average(metric_list)




def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--template", type=str, default="answer")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    template = args.template
    model_path = args.model_path
    BATCH_SIZE = args.batch_size


    IMAGE_FOLDER = "/scratch/azureml/cr/j/f01af20a3317416d9343927e368a55a6/exe/wd/PromptPG/data/tabmwp/tables"



    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    test_data = read_json("/blob/v-yangyi/data/data_files/tabmwp/problems_test1k.json")

    metric = batch_generate(
        test_data=test_data,
        processor=processor,
        model=model,
        image_folder=IMAGE_FOLDER,
        batch_size=BATCH_SIZE,
    )
    print(metric)


   