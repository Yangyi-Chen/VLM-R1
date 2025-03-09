from accelerate import Accelerator
import yaml
import argparse
from tqdm import tqdm
import src.sft.data
import torch
# LR scheduler
import os
import math
from transformers import get_scheduler
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


import os
from tqdm import tqdm



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)


    max_epochs = config["training"]["max_epochs"]
    accumulate_grad_batches = config["training"]["accumulate_grad_batches"]
    output_dir = config['output_dir']
    record = config['training']['record_middle']
    resume = config.get('resume', None)

    if resume is not None:
        print(f"Resuming from {resume}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(resume)
    else:
         # load model and data, optimizer
         print("Loading model. init from Qwen2.5-VL-3B-Instruct")
         model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", cache_dir="./")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    
    accelerator = Accelerator(gradient_accumulation_steps=accumulate_grad_batches, log_with="wandb", mixed_precision='bf16')    
    

   
    data_module = getattr(
        src.sft.data, config["data"]["data_module"]
    )(config, processor)
    data_loader = data_module.train_dataloader()
    lr = config["training"]["lr"]
    # optimize those parameters that require grad
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config["training"]["weight_decay"],
    )
    
    # add lr scheduler
    num_update_steps_per_epoch = math.ceil(len(data_loader) / accelerator.gradient_accumulation_steps)
    
    # if args.max_train_steps is None:
    max_train_steps = max_epochs * num_update_steps_per_epoch
    num_warmup_steps = 0

    # New Code #
    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
    else:
        scheduler = DummyScheduler(
            optimizer, total_num_steps=max_train_steps, warmup_num_steps=num_warmup_steps
        )


    model, optimizer, data, scheduler = accelerator.prepare(model, optimizer, data_loader, scheduler)
      
    
    global_bs = config["data"]['batch_size'] * accelerator.num_processes * accumulate_grad_batches
    
    accelerator.init_trackers(project_name='SFT-INIT', init_kwargs={"wandb": {"name": config['wandb_name']}} )
    model.train()


    step = 0
    for epoch in range(max_epochs):
        for batch in tqdm(data):
            step += 1
            with accelerator.accumulate(model):
                
                outputs = model(**batch)
                loss = outputs.loss
                perplexity = torch.exp(loss)
                accelerator.log({"loss":loss.item(), "perplexity":perplexity.item()})
                
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            

            if (step % 100 == 0) and record:
                # save an intermediate model
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    os.path.join(output_dir, f"intermediate_{step}"),
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    safe_serialization=True
                )
                # Save the tokenizer
                processor.save_pretrained(
                    os.path.join(output_dir, f"intermediate_{step}"),
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    safe_serialization=True
                )

    
    unwrapped_model = accelerator.unwrap_model(model)
    tgt_path = os.path.join(output_dir, "final") if record else output_dir
        
    unwrapped_model.save_pretrained(
        tgt_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        safe_serialization=True
    )
    # Save the tokenizer
    processor.save_pretrained(
        tgt_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        safe_serialization=True
    )




if __name__ == "__main__":
    main()