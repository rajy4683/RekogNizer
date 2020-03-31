"""
    This file contains non-core utility functions 
    used in the overall project
"""

import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from RekogNizer.hyperparams import *
#from RekogNizer import 


def show_sample_images(images, labels, classes, max_count=25):
    print(images.shape, torch.mean(images,[0,2,3]),torch.std(images,[0,2,3]))
    fig = plt.figure(figsize=(10,10))
    for idx in np.arange(max_count):
        ax = fig.add_subplot(5, 5, idx+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx].cpu().numpy(), (1, 2, 0)))
        ax.set(xlabel="Actual="+classes[labels[idx].cpu().numpy()])


def rand_run_name():
    ran = random.randrange(10**80)
    myhex = "%064x" % ran
    #limit string to 64 characters
    myhex = myhex[:10]
    return myhex

def generate_model_save_path(base="/content/drive/My Drive/EVA4/model_saves",rand_string=None):
    if rand_string == None:
        rand_string=rand_run_name()
    file_name = "model-"+rand_string+".h5"
    return os.path.join(base,file_name)

# functions to show an image
def imshow_labels(img,labels):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    fig = plt.figure(figsize=(10,10))
    #plt.figsize = (10,20)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# functions to show an image
def imshow(img,labels):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #print(npimg.shape)
    fig = plt.figure(figsize=(10,10))
    #plt.figsize = (10,20)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
def get_image_samples(imageloader, classes,count=32):
    torch.manual_seed(hyperparameter_defaults['seed'])
    dataiter = iter(imageloader)
    images, labels = dataiter.next()
    # show images
    print(images.shape, torch.mean(images,[0,2,3]),torch.std(images,[0,2,3]))
    imshow(torchvision.utils.make_grid(images[:count], nrow=8),labels[:count])
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(count)))



# get some random training images
def get_image_samples(imageloader, classes,count=32):
    torch.manual_seed(hyperparameter_defaults['seed'])
    dataiter = iter(imageloader)
    images, labels = dataiter.next()
    # show images
    print(images.shape, torch.mean(images,[0,2,3]),torch.std(images,[0,2,3]))
    imshow(torchvision.utils.make_grid(images[:count], nrow=8),labels[:count])
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(count)))


def plot_graphs(df_array, legend_arr, columns=['Test Accuracy'], xlabel="Epochs", ylabel="Accuracy"):
    fig, ax = plt.subplots(figsize=(15, 6))
    for i in range(len(df_array)):
        for col in columns:
            ax.plot(range(df_array[i].shape[0]),
                    df_array[i][col])
    # ax.plot(range(40),
    #         base_metrics_dataframe['Test Accuracy'],
    #         'g',
    #         color='blue')
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(legend_arr)
    plt.show()
