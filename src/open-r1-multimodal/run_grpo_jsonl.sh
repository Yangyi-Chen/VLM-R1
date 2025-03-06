cd src/open-r1-multimodal

export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

RUN_NAME="Qwen2.5-VL-3B-GRPO-tabmwp-test1"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="7" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_jsonl.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name TabMWP \
    --data_file_paths /blob/v-yangyi/data/data_files/tabmwp/problems_train.jsonl \
    --image_folders /scratch/azureml/cr/j/f01af20a3317416d9343927e368a55a6/exe/wd/PromptPG/data/tabmwp/ \
    --max_prompt_length 1024 \
    --num_generations 7 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true

# original num generiaon = 8
# rec_jsons_processed/refcoco_train.jsonl \