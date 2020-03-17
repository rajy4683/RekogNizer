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
    Resize,RandomSizedCrop, MotionBlur
)
from albumentations.pytorch import ToTensor
import numpy as np
from RekogNizer import QuizDNN



saved_model_path=None
def main():
    device = torch.device("cuda" if not hyperparams.hyperparameter_defaults['no_cuda'] else "cpu")

    fileutils.rand_run_name()


    print("Initializing datasets and dataloaders")

    torch.manual_seed(hyperparams.hyperparameter_defaults['seed'])
    patch_size=2
    transform_train = Compose([
    Cutout(num_holes=1,max_h_size=16,max_w_size=16,always_apply=True,p=1,fill_value=[0.4827*255, 0.4724*255, 0.4427*255]),
    MotionBlur(blur_limit=7, always_apply=False, p=0.5),
    HorizontalFlip(p=1),
    Normalize(
        mean=[0.4914, 0.4826, 0.44653],
        std=[0.24703, 0.24349, 0.26519],
        ),
    ToTensor()
    ])

    transform_test = Compose([
    Normalize(
        mean=[0.4914, 0.4826, 0.44653],
        std=[0.24703, 0.24349, 0.26519],
    ),
    ToTensor()
    ])

    trainset = dataloader.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = dataloader.get_dataloader(trainset, hyperparams.hyperparameter_defaults['batch_size'], shuffle=True, num_workers=2)
    
    testset = dataloader.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
    testloader = dataloader.get_dataloader(testset, hyperparams.hyperparameter_defaults['batch_size'], shuffle=False, num_workers=2)
    
    #optim.AdamW
    optimizer=optim.SGD #(model.parameters(), lr=0.001, momentum=0.9)
    criterion=nn.CrossEntropyLoss

    model_new = basemodelclass.ResNet18(hyperparams.hyperparameter_defaults['dropout'])
    final_model_path = traintest.execute_model(model_new, hyperparams.hyperparameter_defaults, 
                trainloader, testloader, 
                device, dataloader.classes,
                optimizer=optimizer,
                prev_saved_model=saved_model_path,
                criterion=criterion,save_best=True,lars_mode=False,batch_step=True)


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

        
