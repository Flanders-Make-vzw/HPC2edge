import json
import time
from pathlib import Path
import tempfile
import os

import torch
import yaml
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import ray
from ray import tune
from ray import train
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.air import RunConfig
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.tune.tuner import Tuner, TuneConfig
from ray.train import Checkpoint

import utils
import models
import dataset


def train(cfg: utils.Config):

    ### Model
    model = getattr(models, cfg.model.name)(**cfg.model._dict)
    
    model = ray.train.torch.prepare_model(model)
    #model.to(device)

    ### Optimization
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.training.lr
    )
    
    current_epoch = 1
    if ray.train.get_checkpoint():
        loaded_checkpoint = ray.train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            checkpoint_dict = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(checkpoint_dict["model_state_dict"])
            optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

            current_epoch = checkpoint_dict["current_epoch"]
            
            scheduler = StepLR(
                optimizer, step_size=cfg.training.step_size, gamma=cfg.training.gamma, last_epoch=current_epoch-1
            )
            
    else:
            scheduler = StepLR(
                optimizer, step_size=cfg.training.step_size, gamma=cfg.training.gamma,
            )        

    ### Data
    ds = getattr(dataset, cfg.dataset.name)(**cfg.dataset._dict)

    train_size = int(cfg.training.split_pct * len(ds))
    val_size = len(ds) - train_size
    train_dataset, valid_dataset = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        cfg.training.batchsize,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        cfg.training.batchsize,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        #persistent_workers=True,
        pin_memory=True,
    )
    
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    valid_loader = ray.train.torch.prepare_data_loader(valid_loader)
    
    ### Logger
    writer = SummaryWriter()
    writer.add_text("Configuration", json.dumps(cfg._dict), global_step=0)
    ### Training loop
    #for epoch in tqdm(range(cfg.training.num_epochs), desc="Epoch"):
    running_val_loss = 0
    for epoch in range(2):
        ### Training
        model.train()  # Set the model to train mode

        for batch_idx, (
            inputs,
            targets,
        ) in tqdm(enumerate(train_loader), desc="Train", total=len(train_loader)):
            optimizer.zero_grad()

            #inputs = inputs.to(device)
            #targets = targets.to(device)
            outputs = model(inputs)
            #outputs = model(inputs.to(device)).cpu()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            utils.log_scalars(
                writer,
                {
                    "loss": loss.item(),
                },
                "Training",
                epoch * len(train_loader) + batch_idx,
            )
        scheduler.step()
        utils.log_scalars(
            writer,
            {
                "lr": scheduler.get_last_lr()[0],
            },
            "Training",
            epoch,
        )

    start_time = time.time()
    
    running_val_loss = 0
    ### Validation
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, targets) in tqdm(
            enumerate(valid_loader), desc="Valid", total=len(valid_loader)
        ):
            #outputs = model(inputs.to(device)).cpu()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_val_loss = running_val_loss + loss.item()
            utils.log_scalars(
                writer,
                {
                    "loss": loss.item(),
                },
                "Validation",
                epoch * len(valid_loader) + batch_idx,
            )
    end_time = time.time() - start_time
    
    os.makedirs("tune_model", exist_ok=True)
    torch.save({
        'current_epoch': current_epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, "tune_model/checkpoint.pt")
    
    checkpoint = Checkpoint.from_directory("tune_model")
    
    
    return running_val_loss/len(valid_loader), checkpoint


if __name__ == "__main__":
    import yaml
    import sys

    if len(sys.argv) < 2:
        print(f"No YAML configuration provided.")
        exit(1)

    with open(sys.argv[1]) as f:
        cfg = utils.Config(yaml.safe_load(f))
    train(cfg)
    print("Exiting.")
