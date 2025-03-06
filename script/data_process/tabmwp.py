import argparse
import json
import jsonlines
import os


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--parse_type", type=str)
parser.add_argument("--data_file_dir", type=str)
args = parser.parse_args()


parse_type = args.parse_type
data_file_dir = args.data_file_dir


 

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_jsonl(path, data):
    with jsonlines.open(path, 'w') as writer:
        writer.write_all(data)

 

# input_path = "problems_train.json"

if parse_type == 'sft_describe':
    input_path = "problems_dev1k.json"
    input_path = os.path.join(data_file_dir, input_path)
    output_path = os.path.join("data", "tabmwp", "sft_describe.jsonl")

    src_data = read_json(input_path)

    save_data = []
    count = 0
    for k, item in src_data.items():
        image = "tables/" + k + ".png"
        conversations = [
            {'from': 'human', 'value': "<image>"+ "Describe this image."},
            {'from': 'gpt', 'value': item['table']}
        ]

        save_data.append({'id': count, 'image': image, 'conversations': conversations, 'cot': item['solution'], 'text_description': str(item['table'])})
        count += 1
    # check if the directory exists, if not create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    write_jsonl(output_path, save_data)

elif parse_type == "sft_reason":
    pass

elif parse_type == "rl_describe":
    input_path = "problems_train.json"
    input_path = os.path.join(data_file_dir, input_path)
    output_path = os.path.join("data", "tabmwp", "rl_describe.jsonl")

    src_data = read_json(input_path)

    save_data = []
    count = 0
    for k, item in src_data.items():
        image = "tables/" + k + ".png"
        conversations = [
            {'from': 'human', 'value': "<image>"+ "Describe this image."},
            {'from': 'gpt', 'value': item['table']}
        ]

        save_data.append({'id': count, 'image': image, 'conversations': conversations, 'cot': item['solution'], 'text_description': str(item['table'])})
        count += 1
    # check if the directory exists, if not create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    write_jsonl(output_path, save_data)


elif parse_type == "rl_reason":
    pass
else:
    raise ValueError("Invalid parse type")