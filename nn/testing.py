import json
from pathlib import Path
import time

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils
import models
import dataset


def test(cfg: utils.Config):
    torch.manual_seed(0)
    # run on GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing on '{device}'")

    print(cfg.test.model_fp)
    print(cfg.model._dict)
    ### Model
    # load model from pt file
    model = getattr(models, cfg.model.name)(**cfg.model._dict)
    state_dict = torch.load(cfg.test.model_fp)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    ### Optimization
    criterion = torch.nn.MSELoss()

    ### Data
    dataset_cfg = cfg.dataset._dict
    dataset_cfg["data_fp"] = cfg.test.data_fp
    dataset_cfg["start_ratio"] = cfg.test.start_ratio
    dataset_cfg["end_ratio"] = cfg.test.end_ratio
    ds = getattr(dataset, cfg.dataset.name)(deterministic=True, **dataset_cfg)

    print(dataset_cfg)


    test_loader = DataLoader(
        ds,
        cfg.test.batchsize,
        shuffle=False,
        num_workers=cfg.test.num_workers,
        drop_last=False,
    )

    ### Logger
    writer = SummaryWriter()
    writer.add_text("Configuration", json.dumps(cfg._dict), global_step=0)
    running_test_loss = 0

    ### Testing loop
    running_loss = 0.0
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, targets) in tqdm(
            enumerate(test_loader), desc="Test", total=len(test_loader)
        ):
            outputs = model(inputs.to(device)).cpu()
            loss = criterion(outputs, targets)

            running_test_loss = running_test_loss + loss.item()

            utils.log_scalars(
                writer,
                {
                    "loss": loss.item(),
                },
                "Test",
                batch_idx,
            )
            running_loss += loss.item()
    running_loss /= len(test_loader)
    print(f"Test loss: {running_loss}")
    utils.log_scalars(
        writer,
        {
            "test_loss": running_loss,
        },
        "Test",
        len(test_loader),
    )
    writer.flush()
    writer.close()

    print("Final test loss: ", running_test_loss/len(test_loader))
    print("Testing finished")


if __name__ == "__main__":
    import yaml
    import sys

    if len(sys.argv) < 2:
        print(f"No YAML configuration provided.")
        exit(1)

    with open(sys.argv[1]) as f:
        cfg = utils.Config(yaml.safe_load(f))
    test(cfg)
    print("Exiting.")
