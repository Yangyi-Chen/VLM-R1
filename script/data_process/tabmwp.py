import argparse
import json
import jsonlines
import os


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--parse_type", type=str)
parser.add_argument("--data_file_dir", type=str, default="/blob/v-yangyi/data/data_files/tabmwp/")
args = parser.parse_args()


parse_type = args.parse_type
data_file_dir = args.data_file_dir


def call_gpt(prompt):
    from openai import OpenAI

    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content




def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_jsonl(path, data):
    with jsonlines.open(path, 'w') as writer:
        writer.write_all(data)

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)




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
            {'from': 'gpt', 'value': "<answer> " + item['table'] + " </answer>"}
        ]

        save_data.append({'id': count, 'image': image, 'conversations': conversations, 'cot': item['solution'], 'text_description': str(item['table'])})
        count += 1
    # check if the directory exists, if not create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    write_jsonl(output_path, save_data)
elif parse_type == "sft_describe_ref":
    input_path = "problems_train.json"
    input_path = os.path.join(data_file_dir, input_path)
    output_path = os.path.join("data", "tabmwp", "sft_describe_ref.jsonl")

    src_data = read_json(input_path)

    save_data = []
    count = 0
    for k, item in src_data.items():
        image = "tables/" + k + ".png"
        conversations = [
            {'from': 'human', 'value': "<image>"+ "Describe this image."},
            {'from': 'gpt', 'value': "<answer> " + item['table'] + " </answer>"}
        ]

        save_data.append({'id': count, 'image': image, 'conversations': conversations, 'cot': item['solution'], 'text_description': str(item['table'])})
        count += 1
    # check if the directory exists, if not create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    write_jsonl(output_path, save_data)



elif parse_type == "sft_reason":
    input_path = "problems_dev1k.json"
    input_path = os.path.join(data_file_dir, input_path)
    output_path = os.path.join("data", "tabmwp", "sft_reason.jsonl")

    src_data = read_json(input_path)

    save_data = []
    count = 0
    for k, item in src_data.items():
        image = "tables/" + k + ".png"
        conversations = [
            {'from': 'human', 'value': "<image>"+ item['question']},
            {'from': 'gpt', 'value': "<visual> " + item['table'] + " </visual>\n" + item['solution'] + '\n' + "<answer> " + item['answer'] + " </answer>"}
        ]
        save_data.append({'id': count, 'image': image, 'conversations': conversations, 'cot': item['solution'], 'text_description': str(item['table'])})
        count += 1
    # check if the directory exists, if not create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    write_jsonl(output_path, save_data)


elif parse_type == "sft_textreason":
    input_path = "problems_dev1k.json"
    input_path = os.path.join(data_file_dir, input_path)
    output_path = os.path.join("data", "tabmwp", "sft_textreason.jsonl")

    src_data = read_json(input_path)

    save_data = []
    count = 0
    for k, item in src_data.items():
      
        conversations = [
            {'from': 'human', 'value': item['question'] + " <visual> " + item['table'] + " </visual>"},
            {'from': 'gpt', 'value': item['solution'] + '\n' + "<answer> " + item['answer'] + " </answer>"}
        ]
        save_data.append({'id': count, 'conversations': conversations, 'cot': item['solution'], 'text_description': str(item['table'])})
        count += 1
    
    # check if the directory exists, if not create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    write_jsonl(output_path, save_data)


elif parse_type == "sft_reason_ref":
    input_path = "problems_train.json"
    input_path = os.path.join(data_file_dir, input_path)
    output_path = os.path.join("data", "tabmwp", "sft_reason_ref.jsonl")

    src_data = read_json(input_path)

    save_data = []
    count = 0
    for k, item in src_data.items():
        image = "tables/" + k + ".png"
        conversations = [
            {'from': 'human', 'value': "<image>"+ item['question']},
            {'from': 'gpt', 'value': "<visual> " + item['table'] + " </visual>\n" + item['solution'] + '\n' + "<answer> " + item['answer'] + " </answer>"}
        ]
        save_data.append({'id': count, 'image': image, 'conversations': conversations, 'cot': item['solution'], 'text_description': str(item['table'])})
        count += 1
    # check if the directory exists, if not create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    write_jsonl(output_path, save_data)

elif parse_type == "sft_reason_direct_ref":
    input_path = "problems_train.json"
    input_path = os.path.join(data_file_dir, input_path)
    output_path = os.path.join("data", "tabmwp", "sft_reason_direct_ref.jsonl")

    src_data = read_json(input_path)

    save_data = []
    count = 0
    for k, item in src_data.items():
        image = "tables/" + k + ".png"
        conversations = [
            {'from': 'human', 'value': "<image>"+ item['question']},
            {'from': 'gpt', 'value': "<answer> " + item['answer'] + " </answer>"}
        ]
        save_data.append({'id': count, 'image': image, 'conversations': conversations, 'cot': item['solution'], 'text_description': str(item['table'])})
        count += 1
    # check if the directory exists, if not create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    write_jsonl(output_path, save_data)



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
    input_path = "problems_train.json"
    input_path = os.path.join(data_file_dir, input_path)
    output_path = os.path.join("data", "tabmwp", "rl_reason.jsonl")

    src_data = read_json(input_path)

    save_data = []
    count = 0
    for k, item in src_data.items():
        image = "tables/" + k + ".png"
        conversations = [
            {'from': 'human', 'value': "<image>"+ item['question']},
            {'from': 'gpt', 'value': item['answer']}
        ]

        save_data.append({'id': count, 'image': image, "conversations": conversations, 'solution' : item['answer']})
        count += 1
    # check if the directory exists, if not create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    write_jsonl(output_path, save_data)

elif parse_type == "rl_textreason":
    input_path = "problems_train.json"
    input_path = os.path.join(data_file_dir, input_path)
    output_path = os.path.join("data", "tabmwp", "rl_textreason.jsonl")

    src_data = read_json(input_path)

    save_data = []
    count = 0
    for k, item in src_data.items():
        conversations = [
            {'from': 'human', 'value': item['question'] + " <visual> " + item['table'] + " </visual>"},
            {'from': 'gpt', 'value': item['answer']}
        ]

        save_data.append({'id': count, "conversations": conversations, 'solution' : item['answer']})
        count += 1
    # check if the directory exists, if not create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    write_jsonl(output_path, save_data)


elif parse_type == 'generate_perception_test':
    input_path = "problems_test1k.json"
    input_path = os.path.join(data_file_dir, input_path)
    output_path = os.path.join("data", "tabmwp", "perception_test.jsonl")



    src_data = read_json(input_path)
    # import random
    # random.seed(42)
    # # shuffle the src_data dictionary
    # keys = list(src_data.keys())
    # random.shuffle(keys)

    save_data = {}
    from tqdm import tqdm

    generate_prompt = '''Here is the markdown table: \n"{}"\n Please generate one easy question that can be *directly* extracted and answered by looking at the table without any further reasoning. Return the question within <question> and </question> tag. Return the answer within <answer> and </answer> tag.'''
    for k, item in tqdm(src_data.items()):
       
        table = src_data[k]['table']
        prompt = generate_prompt.format(table)
        gpt_response = call_gpt(prompt)
        question = gpt_response.split('<question>')[1].split('</question>')[0]
        answer = gpt_response.split('<answer>')[1].split('</answer>')[0]

        save_data[k] = {
            'question': question,
            'answer': answer,
            'table': table
        }
    

    # check if the directory exists, if not create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    write_json(output_path, save_data)





elif parse_type == 'generate_perception_train':
    input_path = "problems_train.json"
    input_path = os.path.join(data_file_dir, input_path)
    output_path = os.path.join("data", "tabmwp", "perception_train.jsonl")



    src_data = read_json(input_path)
    import random
    random.seed(42)
    # shuffle the src_data dictionary
    keys = list(src_data.keys())[:6000]
    random.shuffle(keys)

    save_data = []
    from tqdm import tqdm

    generate_prompt = '''Here is the markdown table: \n"{}"\n Please generate one easy question that can be *directly* extracted and answered by looking at the table without any further reasoning. Return the question within <question> and </question> tag. Return the answer within <answer> and </answer> tag.'''
    count = 0
    for key in tqdm(keys):
        item = src_data[key]
        table = item['table']
        prompt = generate_prompt.format(table)
        gpt_response = call_gpt(prompt)
        question = gpt_response.split('<question>')[1].split('</question>')[0]
        answer = gpt_response.split('<answer>')[1].split('</answer>')[0]
        image = "tables/" + key + ".png"
        conversations = [
            {'from': 'human', 'value': "<image>"+ question},
            {'from': 'gpt', 'value':  "<visual> " + item['table'] + " </visual>\n" + "<answer> " + answer + " </answer>"}
        ]

        save_data.append({
            'id': count,
            'image': image,
            'conversations': conversations,  
        })
        count += 1
   
    # check if the directory exists, if not create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    write_jsonl(output_path, save_data)


else:
    raise ValueError("Invalid parse type")