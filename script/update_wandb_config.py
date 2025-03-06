import os
import wandb

idx = wandb.util.generate_id()
os.system(f"export WANDB_RUN_ID={idx}")