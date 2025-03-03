import os
import yaml
import argparse
import src.model
import src.data
import torch
import wandb
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
from tqdm import tqdm


def main():
    # load config from args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--local-rank", type=int,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")

    args = parser.parse_args()
    local_rank = args.local_rank

    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda:{}".format(local_rank))


    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # import model and data
    model = getattr(src.model, config["model"]["model_module"])(config)

    checkpoint = config["model"].get("resume_from_checkpoint", None)
    if checkpoint is not None:
        print(f"Resuming training from {checkpoint}")
        state_dict = torch.load(checkpoint, map_location='cpu')
        msg = model.load_state_dict(state_dict, strict=False)
        print("Load state dict: ", msg)
        
    
    

    data_module = getattr(
        src.data, config["data"]["data_module"]
    )(config, model.tokenizer)
    max_steps = config["training"]["max_steps"]
    accumulate_grad_batches = config["training"]["accumulate_grad_batches"]
    checkpoint_every_n_steps = config["training"]["checkpoint_every_n_steps"]
    checkpoint_every_n_steps = checkpoint_every_n_steps * accumulate_grad_batches
    output_dir = config['output_dir'] + str(config['model']['gemma'])
    if torch.distributed.get_rank() == 0:
        wandb.login()
        run = wandb.init(
            project="filter-capfusion-train" if 'sft' not in args.config else "analysis-sft",
            name="Llava-filter-Gemma={}".format(config['model']['gemma']),
    )
        
        
        
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    data_loader = data_module.train_dataloader()
    optimizer = model.module.configure_optimizers()

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 100 if "sft" not in args.config else 0, max_steps // accumulate_grad_batches)


    model.train()
    step = 0 
    for batch in tqdm(data_loader):
        if batch is None:
            continue
        loss = model(batch, device)
        if torch.distributed.get_rank() == 0:
            wandb.log({"VL loss": loss.item()})
        loss.backward()
        
        if (step+1) % accumulate_grad_batches == 0:
            for param in model.parameters():
                if param.requires_grad:
                    if param.grad is None:
                        continue
                    param.grad.data /= accumulate_grad_batches
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


        if (step+1) % checkpoint_every_n_steps == 0:
            if torch.distributed.get_rank() == 0:
                record_step = (step+1) // accumulate_grad_batches
                if os.path.exists(output_dir):
                    pass
                else:
                    os.makedirs(output_dir)
                torch.save(model.module.state_dict(), os.path.join(output_dir, f"model_{record_step}.pt"))
            torch.distributed.barrier()
        
        
        step += 1
        if step >= max_steps:
            break


if __name__ == '__main__':
    main()