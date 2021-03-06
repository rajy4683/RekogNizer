# -*- coding: utf-8 -*-
"""S7EVA4_Depth_Dilate.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UozyAe8uIsdgTc4Z3ojRALQoRxIvNlMo
"""

# from google.colab import drive
# drive.mount('/content/drive')
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#!pip install git+https://github.com/rajy4683/RekogNizer.git

#!git clone https://github.com/rajy4683/RekogNizer.git /content/drive/My\ Drive/EVA4/TestImports

#!pip install -r /content/drive/My\ Drive/EVA4/RekogNizer/requirements.txt

#!wandb login a6f947d2d2f69e7a8c8ca0f69811fd554f27d204

sys.path.append('/content/drive/My Drive/EVA4/')
sys.path.append('/content/drive/My Drive/EVA4/RekogNizer')

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2
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


saved_model_path=None
def main():
    device = torch.device("cuda" if not hyperparams.hyperparameter_defaults['no_cuda'] else "cpu")

    fileutils.rand_run_name()

    # print(len(trainloader))
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # print(images.shape)

    #hyperparams.print_hyperparams()

    # fileutils.get_image_samples(trainloader, classes)

    # model_new = basemodelclass.CIFARModelDepthDilate().to(device)
    # summary(model_new,input_size=(3, 32, 32))

    # type(model_new)

    print("Initializing datasets and dataloaders")

    torch.manual_seed(hyperparams.hyperparameter_defaults['seed'])
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = dataloader.get_dataloader(trainset, hyperparams.hyperparameter_defaults['batch_size'], shuffle=True, num_workers=2)
    
    # torch.utils.data.DataLoader(trainset, batch_size=hyperparameter_defaults['batch_size'],
    #                                         shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
    testloader = dataloader.get_dataloader(testset, hyperparams.hyperparameter_defaults['batch_size'], shuffle=False, num_workers=2)
    
    # torch.utils.data.DataLoader(testset, batch_size=hyperparameter_defaults['batch_size'],
    #                                         shuffle=False, num_workers=2)

    optimizer=optim.SGD#(model.parameters(), lr=0.001, momentum=0.9)
    criterion=nn.CrossEntropyLoss
    #model = basemodelclass.CIFARModelBuilder()#.to(device)
    #model_new = basemodelclass.CIFARModelDepthDilate#.to(device)
    model_new = basemodelclass.ResNet18(hyperparams.hyperparameter_defaults['dropout'])
    #execute_model(model, hyperparameter_defaults, )
    final_model_path = traintest.execute_model(model_new, hyperparams.hyperparameter_defaults, 
                trainloader, testloader, 
                device, dataloader.classes,
                optimizer=optimizer,
                prev_saved_model=saved_model_path,
                criterion=criterion,save_best=True)

    # Commented out IPython magic to ensure Python compatibility.
    # %autoreload 2
    # from RekogNizer import logger
    # run_list = ["rajy4683/news4eva4/runs/h5cejeg2","rajy4683/news4eva4/runs/2zoz8lca", "rajy4683/news4eva4/runs/mi0limq7"]

    # runs_df = logger.get_wandb_dataframes_proj(project="rajy4683/news5",count=1)
    # new_df = pd.DataFrame().append(runs_df)

    
    # fileutils.plot_graphs([new_df],
    #             ['Train Accuracy', 'Test Accuracy'],
    #             columns=['Train Accuracy', 'Test Accuracy'],
    #             xlabel="Epochs",
    #             ylabel="Accuracy")

    # fileutils.plot_graphs([new_df],
    #             ['Train Loss', 'Test Loss'],
    #             columns=['Train Loss', 'Test Loss'],
    #             xlabel="Epochs",
    #             ylabel="Loss")

    # from RekogNizer import traintest

    #my_model = traintest.model_builder(basemodelclass.CIFARModelDepthDilate, weights_path=final_model_path)
    #"/content/drive/My Drive/EVA4/model_saves/model-2c0cd03c7a.h5")
    #class_accuracy_dict = traintest.classwise_accuracy(my_model, testloader, classes, device=torch.device("cpu"))

    #plt.bar([key for key in class_accuracy_dict.keys()],[val for val in class_accuracy_dict.values()])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train CIFAR')
    #parser.add_argument("-h", "--help", required=False, help="Can be used to manipulate load-balancing")
    parser.add_argument("-p", "--params", required=False, help="JSON format string of params E.g: '{\"lr\":0.01, \"momentum\": 0.9}' ")
    parser.add_argument("-r", "--saved_model_path", required=False, help="Load and resume model from this path ")

    args = parser.parse_args()
    
    # if(args.help is not None):
    #     print("Basic help")
    #     return
    # if (args.testloader is not None):
    #         testloader = args.testloader
    #saved_model_path=""
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

        
