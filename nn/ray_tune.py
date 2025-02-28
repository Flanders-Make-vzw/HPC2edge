# load general modules
from typing import Dict
import argparse
import os
import time
import numpy as np
import pandas as pd
import random

# load torch and torchvision modules 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

import torchvision
from torchvision import datasets, transforms, models

# load ray modules
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.air import RunConfig
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.tune.tuner import Tuner, TuneConfig

from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pbt import PopulationBasedTrainingReplay

import nevergrad as ng
from ray.tune.search.nevergrad import NevergradSearch


def parsIni():
    parser = argparse.ArgumentParser(description='Ray Tune HPC2EDGE Example')
    parser.add_argument('--num-samples', type=int, default=24, metavar='N',
                    help='number of samples to train (default: 24)')
    parser.add_argument('--max-iterations', type=int, default=1, metavar='N',
                    help='maximum iterations to train (default: 10)')
    parser.add_argument('--par-workers', type=int, default=1, metavar='N',
                    help='parallel workers to train on a single trial (default: 1)')
    parser.add_argument('--scheduler', type=str, default='RAND',
                    help='scheduler for tuning (default: RandomSearch)')
    parser.add_argument('--data-dir', type=str, default='',
                    help='data directory for HPC2EDGE dataset')
    parser.add_argument('--seed', type=int, default=111, metavar='N',
                    help='random seed (default: 111)')

    return parser

def accuracy(output, target):
    """! function that computes the accuracy of an output and target vector 
    @param output vector that the model predicted
    @param target actual  vector
    
    @return correct number of correct predictions
    @return total number of total elements
    """
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    
    # count correct classifications
    correct = pred.eq(target.view_as(pred)).cpu().float().sum()
    
    # count total samples
    total = target.size(0)
    return correct, total

def par_mean(field):
    """! function that averages a field across all workers to a worker
    @param field field in worker that should be averaged
    
    @return mean field
    """
    
    # convert field to tensor
    res = torch.Tensor([field])
    
    # move field to GPU/worker
    res = res.cuda()
    
    # AllReduce operation
    dist.all_reduce(res,op=dist.ReduceOp.SUM,group=None,async_op=True).wait()
    
    # average of number of workers
    res/=dist.get_world_size()
    
    return res

def par_sum(field):
    """! function that sums a field across all workers to a worker
    @param field field in worker that should be summed up
    
    @return sum of all fields
    """
    # convert field to tensor
    res = torch.Tensor([field])
    
    # move field to GPU/worker
    res = res.cuda()
    
    # AllReduce operation
    dist.all_reduce(res,op=dist.ReduceOp.SUM,group=None,async_op=True).wait()
    
    return res
    

def train_hpc_edge(config):

    import training
    import utils
    
    data = {
        "model": {
            "name": "Swin3D",
            "in_chans": 1,  # Number of input image channels
            "patch_size": [config["patch_size_1"], config["patch_size_2"], config["patch_size_3"]],  # Patch size
            #"embed_dim": 96,  # Number of linear projection output channels
            "embed_dim": config["embed_dim"],
            "depths": [config["depths_1"], config["depths_2"], config["depths_3"], config["depths_4"]],  # Depths of each Swin Transformer stage
            "num_heads": [config["num_heads_1"], config["num_heads_2"], config["num_heads_3"], config["num_heads_4"]],  # Number of attention heads of each stage
            "window_size": [8, 7, 7],  # Window size
            "mlp_ratio": config["mlp_ratio"],  # Ratio of MLP hidden dim to embedding dim
            "num_classes": 400  # Penultimate hidden dim size
        },
        "dataset": {
            "name": "OneWaySP",
            "num_workers": 7,
            "data_fp": config["data_dir"],
            "num_frames": 16,
            "crop_size": 224,
            "side_size": 224,
            "repeat_frames": False
        },
        "training": {
            "split_pct": 0.8,
            "num_epochs": config["num_epochs"],
            "batchsize": config["batchsize"],
            "lr": config["lr"],
            "step_size": config["step_size"],
            "gamma": config["gamma"],
            "save_each": -1
        }
    }
    
        
    if train.get_context().get_world_rank() == 0:
        import communicator
        #communicator.connect_database()
        trial_name = train.get_context().get_trial_name()
        model_id = communicator.send_data(trial_name, config)
    
    
    network_config = utils.Config(data)
    
    for i in range(config["num_epochs"]):

        val_loss, checkpoint = training.train(network_config)

        # set to large number as default
        inference_time = 1000

        if train.get_context().get_world_rank() == 0:
            communicator.connect_database()
            inference_time = communicator.receive_data(model_id)  

        inference_time_tensor = torch.tensor([inference_time], dtype=torch.float32).cuda()

        dist.broadcast(inference_time_tensor, 0)

        inference_time = inference_time_tensor.item()

        val_loss_tensor = torch.tensor([val_loss], dtype=torch.float32).cuda()
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)

        # average val_loss over number of workers
        val_loss_report = val_loss_tensor.item()/dist.get_world_size()

        if train.get_context().get_world_rank() == 0:
            # report back the accuracy
            res_id = communicator.insert_result(
                "RAISE-LPBF-Laser training hold-out",
                model_id,
                "root_mean_squared_error",
                val_loss_report,
                dotenv_fp=".env_test",
            )

        session.report({"score": val_loss_report + inference_time, "val_loss": val_loss_report, "inference_time": inference_time}, checkpoint=checkpoint)

def main(args):
    """! main function
    @param args input arguments
    """    
    
    # initalize Ray with the correct adress and node ip adress
    ray.init(address=os.environ['ip_head'], _node_ip_address=str(os.environ["SLURMD_NODENAME"] + "i"))   
        
    random.seed(args.seed)
    
    # define the hyperparameter search space 
    config = {
        # actual hyperparameters
        "batchsize": tune.choice([8]),
        "lr": tune.loguniform(10e-5, 1),
        "step_size": tune.choice([10, 20, 40]),
        "gamma": tune.uniform(0.1, 0.9),
        
        # architectural parameters
        "patch_size_1": tune.choice([2, 4]),
        "patch_size_2": tune.choice([2, 4]),
        "patch_size_3": tune.choice([2, 4]),
        "depths_1": tune.choice([1, 2, 4]),
        "depths_2": tune.choice([1, 2, 4]),
        "depths_3": tune.choice([1, 2, 4]),
        "depths_4": tune.choice([1, 2, 4]),
        "num_heads_1": tune.choice([3]),
        "num_heads_2": tune.choice([3, 6]),
        "num_heads_3": tune.choice([3, 6, 12]),
        "num_heads_4": tune.choice([3, 6, 12, 24]),
        "mlp_ratio": tune.choice([1., 2., 3., 4.]),
        "embed_dim": tune.choice([24, 48]),
        
        #fixed arguments
        "num_epochs": tune.choice([args.max_iterations]),
        "data_dir": tune.choice([args.data_dir]),
    }
    
    mutation_config = {
        # actual hyperparameters
        "batchsize": [8],
        "lr": tune.loguniform(10e-5, 1),
        "step_size": [10, 20, 40],
        "gamma": tune.uniform(0.1, 0.9),
        
        # architectural parameters
        "patch_size_1": [2, 4],
        "patch_size_2": [2, 4],
        "patch_size_3": [2, 4],
        "depths_1": [1, 2, 4],
        "depths_2": [1, 2, 4],
        "depths_3": [1, 2, 4],
        "depths_4": [1, 2, 4],
        "num_heads_1": [3],
        "num_heads_2": [3, 6],
        "num_heads_3": [3, 6, 12],
        "num_heads_4": [3, 6, 12, 24],
        "mlp_ratio": [1., 2., 3., 4.],
        "embed_dim": [24, 48],
    }
    
    # select a hyperparameter optimization algorithm
    
    if (args.scheduler == "RAND"):
        # random scheduler
        scheduler = None
        search_alg = None
    if (args.scheduler == "NEVERGRAD"):
        
        scheduler=None

        #search_alg=NevergradSearch(optimizer=ng.optimizers.OnePlusOne)
        search_alg=NevergradSearch(optimizer=ng.optimizers.NGOpt, metric="score", mode="min",)
        
    # define a reporter/logger to specifify which metrics to print out during the optimization process    
    reporter = CLIReporter(
        metric_columns=["train_acc", "test_acc", "training_iteration", "time_this_iter_s", "time_total_s"],
        max_report_frequency=60)
    
    
    # define the general RunConfig of Ray Tune
    run_config = RunConfig(
        # name of the training run (directory name).
        name="AM_training",
        # directory to store the ray tune results in .
        storage_path="./",
        # logger
        progress_reporter=reporter,
        # stopping criterion when to end the optimization process
        stop={"training_iteration": args.max_iterations}

    )
    
    # wrapping the torch training function inside a TorchTrainer logic
    trainer = TorchTrainer(
        # torch training function
        train_loop_per_worker=train_hpc_edge,
        # setting the default resources/workers to use for the training function, including the number of CPUs and GPUs
        scaling_config=ScalingConfig(num_workers=args.par_workers, use_gpu=True, resources_per_worker={"CPU": 7, "GPU": 1}),
    )
    
    # defining the hyperparameter tuner 
    tuner = Tuner(
        # function to tune
        trainer,
        # hyperparameter search space
        param_space={"train_loop_config": config},
        # the tuning configuration
        tune_config=TuneConfig(
           # define how many trials to evaluate 
           num_samples=args.num_samples,
           # define which metric to use for measuring the performance of the trials
           metric="score",
           # if the metric should be maximized or minimized 
           mode="min",
           # define which scheduler to use 
           scheduler=scheduler,
            # define which search algorithm to use
           search_alg=search_alg,
           ),
        run_config=run_config
    )
    
    # measure the total runtime
    start_time = time.time()
    
    # start the optimization process
    result = tuner.fit()
    
    runtime = time.time() - start_time
    
    # print total runtime
    print("Total runtime: ", runtime)
    
    result_df = result.get_dataframe()
    print(result_df)
    result_df.to_csv(f'output_table_samples{args.num_samples}_seed{args.seed}.csv')

    # print metrics of the best trial
    best_result = result.get_best_result(metric="score", mode="min")    
    
    print("Best result metrics: ", best_result) 
    
    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    
    print("Best model checkpoint path: ", checkpoint_path)

    import testing
    import json
    import utils


    with open(os.path.join(best_result.path, "params.json"), 'r') as file:        
        best_config = json.load(file)["train_loop_config"]

    data = {
    "model": {
        "name": "Swin3D",
        "in_chans": 1,  # Number of input image channels
        "patch_size": [best_config["patch_size_1"], best_config["patch_size_2"], best_config["patch_size_3"]],  # Patch size
        "embed_dim": best_config["embed_dim"],
        "depths": [best_config["depths_1"], best_config["depths_2"], best_config["depths_3"], best_config["depths_4"]],  # Depths of each Swin Transformer stage
        "num_heads": [best_config["num_heads_1"], best_config["num_heads_2"], best_config["num_heads_3"], best_config["num_heads_4"]],  # Number of attention heads of each stage
        "window_size": [8, 7, 7],  # Window size
        "mlp_ratio": best_config["mlp_ratio"],  # Ratio of MLP hidden dim to embedding dim
        "num_classes": 400,  # Penultimate hidden dim size
    },
    
    "dataset": {
        "name": "OneWaySP",
        "num_workers": 6,
        "data_fp": "/RAISE_LPBF_train_sampled.hdf5",
        "num_frames": 16,
        "crop_size": 224,
        "side_size": 224,
        "repeat_frames": False
    },

    "training": {
        "split_pct": 0.8,
        "num_epochs": best_config["num_epochs"],
        "batchsize": best_config["batchsize"],
        "lr": best_config["lr"],
        "step_size": best_config["step_size"],
        "gamma": best_config["gamma"],
        "save_each": -1
    },

    "inference": {
          "in_size": [1,16,224,224],
          "num_epochs": 1
    },
    
    "test": {
      "data_fp": "/RAISE_LPBF_train_sampled.hdf5",
      "num_workers": 6,
      "model_fp": os.path.join(best_result.path, "checkpoint_000001/checkpoint.pt"),
      "batchsize" : 8
    }
    }

    best_network_config = utils.Config(data)
    testing.test(best_network_config)
    



if __name__ == "__main__":
    
    # get custom arguments from parser
    parser = parsIni()
    args = parser.parse_args()
    
    # call the main function to launch Ray
    main(args)
