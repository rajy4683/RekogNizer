# -*- coding: utf-8 -*-
"""
   train_s9_new.py:
   Contains training using albumentation/Resnet and CIFAR10
"""

# from google.colab import drive
# drive.mount('/content/drive')
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


sys.path.append('/content/drive/My Drive/EVA4/')
sys.path.append('/content/drive/My Drive/EVA4/RekogNizer')

from RekogNizer import hyperparams
from RekogNizer import basemodelclass
from RekogNizer import fileutils
from RekogNizer import dataloader
from RekogNizer import traintest
from RekogNizer import logger
from torchsummary import summary
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import json
import torchvision.transforms as transforms
import torchvision
from albumentations import (
    HorizontalFlip, Compose, RandomCrop, Cutout,Normalize, HorizontalFlip, 
    Resize,RandomSizedCrop, MotionBlur,PadIfNeeded,Flip, IAAFliplr,
)
from albumentations.pytorch import ToTensor
import numpy as np
from RekogNizer import lrfinder
from torch.optim.lr_scheduler import StepLR, OneCycleLR, MultiStepLR, CyclicLR, ReduceLROnPlateau
import wandb


saved_model_path=None
def main():
    device = torch.device("cuda" if not hyperparams.hyperparameter_defaults['no_cuda'] else "cpu")

    
    hyperparams.hyperparameter_defaults['run_name'] = fileutils.rand_run_name()
    print("Initializing datasets and dataloaders")    
    train_path = "/content/t2/train"
    test_path="/content/t2/val"
    #model_new = basemodelclass.ResNet18(hyperparams.hyperparameter_defaults['dropout'], num_classes=200)
    trainloader, testloader = dataloader.get_imagenet_loaders(train_path, test_path, transform_train=None, transform_test=None)
    model_new = basemodelclass.S11ResNet()

    wandb_run_init = wandb.init(config=hyperparams.hyperparameter_defaults, project=hyperparams.hyperparameter_defaults['project'])
    wandb.watch_called = False
    config = wandb.config
    print(config)
    wandb.watch(model_new, log="all")

    #trainloader, testloader = dataloader.get_train_test_dataloader_cifar10()
    optimizer=optim.SGD(model_new.parameters(), lr=config.lr,momentum=config.momentum,
                         weight_decay=config.weight_decay)
    
    #optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion=nn.CrossEntropyLoss
    #scheduler = None
    cycle_momentum = True if config.cycle_momentum == "True" else False
    print("Momentum cycling set to {}".format(cycle_momentum))
    if (config.lr_policy == "clr"):
        scheduler = CyclicLR(optimizer, 
                             base_lr=config.lr*0.01, 
                             max_lr=config.lr, mode='triangular', 
                             gamma=1., 
                             cycle_momentum=True,
                             step_size_up=256)#, scale_fn='triangular',step_size_up=200)
    else:
        scheduler = OneCycleLR(optimizer, 
                                config.ocp_max_lr, 
                                epochs=config.epochs, 
                                cycle_momentum=cycle_momentum, 
                                steps_per_epoch=len(trainloader), 
                                base_momentum=config.momentum,
                                max_momentum=0.95, 
                                pct_start=config.split_pct,
                                anneal_strategy=config.anneal_strategy,
                                div_factor=config.div_factor,
                                final_div_factor=config.final_div_factor
                            )
    
    final_model_path = traintest.execute_model(model_new, 
                hyperparams.hyperparameter_defaults, 
                trainloader, testloader, 
                device, dataloader.classes,
                wandb=wandb,
                optimizer_in=optimizer,
                scheduler=scheduler,
                prev_saved_model=saved_model_path,
                criterion=criterion,
                save_best=True,
                lars_mode=False,
                batch_step=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train CIFAR')
    #parser.add_argument("-h", "--help", required=False, help="Can be used to manipulate load-balancing")
    parser.add_argument("-p", "--params", required=False, help="JSON format string of params E.g: '{\"lr\":0.01, \"momentum\": 0.9}' ")
    parser.add_argument("-r", "--saved_model_path", required=False, help="Load and resume model from this path ")

    args = parser.parse_args()
    
    if (args.saved_model_path is not None):
        saved_model_path = args.saved_model_path
        print("Model will be loaded from",saved_model_path)
    if (args.params is not None):
        #if(args.params == "params"):    
        arg_val_dict = json.loads(args.params)
        #print(hyperparams.hyperparameter_defaults['lr'])
        for key,val in arg_val_dict.items():
            print("Setting ",key," = ",val)
            hyperparams.hyperparameter_defaults[key]=val
        print("Final Hyperparameters")
        hyperparams.print_hyperparams()
        main()
    #return

        
