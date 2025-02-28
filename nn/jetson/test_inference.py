import tracemalloc
from typing import Any, Dict
import torch
from tqdm import tqdm
import time
import yaml
from pathlib import Path

from .jetson_utils import JetsonMonitor

from .. import utils
from .. import models


def infer_test(
    model_name: str,
    batch_size: int,
    hyperparameters: Dict[str, Any],
    num_epochs: int = 10,
):
    ### Config
    script_dir = Path(__file__).resolve().parent
    with open(script_dir / f"../configs/{model_name}.yaml") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict["model"].update(hyperparameters)
    cfg = utils.Config(cfg_dict)

    ### GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Inferencing on '{device}'")

    ### Model
    model = getattr(models, cfg.model.name)(**cfg.model._dict)
    model.to(device)
    model.eval()

    ### Jetson Monitor
    jmonitor = JetsonMonitor()

    ### Inference loop
    inference_times = []
    with torch.no_grad():
        # discard first inference
        model(torch.rand(size=(batch_size, *cfg.inference.in_size)).to(device))
        # start monitoring
        tracemalloc.start()
        jmonitor.start()
        for _ in tqdm(range(num_epochs), desc="Epoch"):
            tic = time.time()
            model(torch.rand(size=(batch_size, *cfg.inference.in_size)).to(device))
            inference_times.append((time.time() - tic) * 1e3)

    ### Stats
    jmonitor.stop()
    jmonitor_stats = jmonitor.get_stats()
    if jmonitor_stats is None:
        jmonitor_stats = {}
        print("Warning: No jtop stats available.")
    peak_memory_usage = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    avg_inference_time = sum(inference_times) / len(inference_times)
    gpu_memory_usage = torch.cuda.memory_allocated(device)

    return {
        "peak_memory_usage": peak_memory_usage,
        "avg_inference_time": avg_inference_time,
        "gpu_memory_usage": gpu_memory_usage,
        **jmonitor_stats,
    }
