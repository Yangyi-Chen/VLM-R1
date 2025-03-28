import argparse
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
import json
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import torch
from typing import Optional, Tuple
from Levenshtein import ratio
from statistics import mean as average
import re
from math_verify import parse, verify

def call_gpt(prompt):
    from openai import OpenAI

    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content


def ai_check(question, table, gt, text):
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]
    CHECK_PROMPT = "You need to verify the answer generated by another model to a given question that asks information about a table. \n The question is: {} \n The table is: {} \n The ground truth answer is: {} \n The answer generated by the model is: {} \n Please verify the answer by first generating your reason, and then output you verification as True or False within the following format: \n <answer>True</answer> or <answer>False</answer> \n"
    prompt = CHECK_PROMPT.format(question, table, gt, text)
    response = call_gpt(prompt)
    verification = re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if verification:
        return 1.0 if verification[-1].strip().lower() == "true" or verification[-1].strip().lower() == "True" else 0.0
    else:
        print("Verification failed")
        return 0.0



def clean_text(text, exclue_chars=['\n', '\r']):
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
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


def numeric_reward(content, sol, **kwargs):
    content = clean_text(content)
    sol = clean_text(sol)
    try:
        content, sol = float(content), float(sol)
        return 1.0 if content == sol else 0.0
    except:
        return None




def default_accuracy_reward(content, sol, **kwargs):
    reward = 0.0
    # Try symbolic verification first for numeric answers
    try:
        answer = parse(content)
        if float(verify(answer, parse(sol))) > 0:
            reward = 1.0
    except Exception:
        pass  # Continue to next verification method if this fails
    
    # If symbolic verification failed, try string matching or fuzzy matching
    if reward == 0.0:
        try:
            # Extract answer from solution if it has think/answer tags
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            
            # Extract answer from content if it has think/answer tags
            content_matches = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
            student_answer = content_matches[-1].strip() if content_matches else content.strip()
            
            # Check if ground truth contains numbers
            has_numbers = bool(re.search(r'\d', ground_truth))
            # Check if it's a multiple choice question
            # has_choices = extract_choice(ground_truth)
            
            if has_numbers:
                # For numeric answers, use exact matching
                reward = numeric_reward(student_answer, ground_truth)
                if reward is None:
                    reward = 0
                # if reward is None:
                #     reward = ratio(clean_text(student_answer), clean_text(ground_truth))
            else:
                # text answers 
                if ground_truth == student_answer or ground_truth in student_answer:
                    reward = 1.0
                
            # elif has_choices:
            #     # For multiple choice, extract and compare choices
            #     correct_choice = has_choices.upper()
            #     student_choice = extract_choice(student_answer)
            #     if student_choice:
            #         reward = 1.0 if student_choice == correct_choice else 0.0
            # else:
            #     # For text answers, use fuzzy matching
            #     reward = ratio(clean_text(student_answer), clean_text(ground_truth))
        except Exception:
            pass  # Keep reward as 0.0 if all methods fail

    return reward





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
            if image_folder is None:
                messages = [
                    {'role': 'user', 'content': test_data[k]['question'] + " <visual> " +  test_data[k]['table'] + " </visual>"}
                ]
            else:
                image_path = os.path.join(image_folder, k + ".png")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path,
                            },
                            {"type": "text", "text": "<image>" + test_data[k]['question']},
                        ],
                    }
                ]
            batch_messages.append(messages)
            target_list.append(test_data[k]["answer"])

        
        # Process batch
        batch_text = []
        batch_image_inputs = []
       

        
        for messages in batch_messages:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_text.append(text)
            if image_folder is not None:
                image_inputs, video_inputs = process_vision_info(messages)
                batch_image_inputs.extend(image_inputs)
        
        # Create inputs for the whole batch
        if image_folder is not None:
            inputs = processor(
                text=batch_text,
                images=batch_image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
                padding_side="left"
            )
        else:
            inputs = processor(
                text=batch_text,
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
            # print(output_text)
            # Store the result
            metric = default_accuracy_reward(output_text, target_list[j])
            # compute the metric
            if metric != 1.0:
                metric = ai_check(test_data[k]['question'], test_data[k]['table'], test_data[k]['answer'], output_text)
            # if metric != 1.0:
            #     print(test_data[k]['question'])
            #     print(test_data[k]['table'])
            #     print(test_data[k]['answer'])
            #     print("-"*90)
            #     print(output_text)
            #     print("*"*50)

            metric_list.append(metric)
        
        print(average(metric_list))
    
    return average(metric_list)




def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    model_path = args.model_path
    BATCH_SIZE = args.batch_size


    


    if "VL" in model_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        IMAGE_FOLDER = "/scratch/azureml/cr/j/f01af20a3317416d9343927e368a55a6/exe/wd/PromptPG/data/tabmwp/tables"
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
        processor = AutoTokenizer.from_pretrained(model_path)
        IMAGE_FOLDER = None


    # test_data = read_json("/blob/v-yangyi/data/data_files/tabmwp/problems_test1k.json")
    test_data = read_json(os.path.join("data", "tabmwp", "perception_test.jsonl"))
 

    metric = batch_generate(
        test_data=test_data,
        processor=processor,
        model=model,
        image_folder=IMAGE_FOLDER,
        batch_size=BATCH_SIZE,
    )
    print(metric)


   