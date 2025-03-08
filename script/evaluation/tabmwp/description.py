import argparse
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import json
from qwen_vl_utils import process_vision_info




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



    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, cache_dir="./").cuda()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    test_data = read_json("/blob/v-yangyi/data/data_files/tabmwp/problems_test1k.json")

    for k, item in test_data.items():
        image_path = os.path.join(IMAGE_FOLDER, k + ".png")
        qustion = item['question']
        table = item['table']
        answer = item['solution']


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
        print(output_text)





