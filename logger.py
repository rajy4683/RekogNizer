from torchsummary import summary
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

from argparse import ArgumentParser
from tqdm import tqdm
import os
import random

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from torch.utils.data import DataLoader
from ignite.handlers import Checkpoint, DiskSaver

import wandb


def get_wandb_dataframes(run_list=None, project=None):
    api = wandb.Api()
    delta_dataframes = []
    for run_key in run_list:
        delta_dataframes.append(api.run(run_key).history())
    return delta_dataframes

def get_wandb_dataframes_proj(project=None, count=1):
    api = wandb.Api()
    delta_dataframes = []
    if project is None:
        return
    #project_wandb = "rajy4683/"+  
    current_runs = api.runs(project, order='-created_at')
    #print(current_runs[0].history()['Train Accuracy'])
    for run_key in range(count):
        print(current_runs[run_key].name)
        delta_dataframes.append(current_runs[run_key].history())
    return delta_dataframes
