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


"""
    shark_teeth_generator: generates a list of cyclic values within a given range
    range_start: start value of the range 
    range_end: ending value of the range
    steps: steps over which the cycle needs to be calculated.  
    returns: list of cyclic values within the given range
"""
def shark_teeth_generator(range_start,range_end,steps, ):
    mult_factor = (range_end - range_start)/(steps)    
    mid_point = np.floor(steps/2)
    out_arr = []
    for val in range(steps):
        if val <= mid_point:
            out_arr.append(range_start+(mult_factor*val))
        else:
            out_arr.append(range_start+(mult_factor*(2*mid_point-val)))
    return out_arr
"""
    saw_graph_generator: Plots a saw/saw-like graph for a given number of iterations and frequency
    total_steps: Total number of steps to plot the graph for 
    single_step_len: Total number of steps per cycle
    y_start: starting range of each cycle  
    y_end: ending range of each cycle  
    returns: list of cyclic values within the given range
"""
def saw_graph_generator(total_steps, single_step_len, y_start, y_end):
    fig = plt.figure(figsize=(20,5))
    start_range = 1
    total_elements = total_steps
    large_array = np.arange(1, total_elements+1, 1)
    slice_length = single_step_len
    output_array=[]
    for i in range(int(np.floor(total_elements/slice_length))):
        slice_local =large_array[slice_length*i:slice_length*(i+1)]
        out_arr = shark_teeth_generator(y_start,y_end,slice_length)
        output_array.append(out_arr)
    plt.plot(large_array[:int(slice_length * np.floor(total_elements/slice_length))],np.reshape(output_array, -1))


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

"""
    Creates a DataFrame for the DepthDataSet
    
"""
def create_csvfile_new(basedir, depth_image_filename, base_image_filename):
    base_image_zipfile = ZipFile(os.path.join(basedir,base_image_filename))
    namelist = [ (
                val,
                val.replace("image_","mask_"), 
                "depth_"+val, 
                val[17:], 
                os.path.basename(base_image_filename), 
                os.path.basename(depth_image_filename))
                for val in base_image_zipfile.namelist() if (re.search("image_",val))
                ]
    df_data_list = pd.DataFrame(namelist,columns=["ImageName", "MaskName", "Depthname","BGImageName", "BaseImageFName","DepthImageFName"])
    return df_data_list

"""
    Generates the full DataFrame from multiple depth_images.zip and base_image.zip files
"""

def create_depth_dataset_csv(base_dir, image_depth_set_map):
    df_data_list = pd.DataFrame(columns=["ImageName", "MaskName", "Depthname","BGImageName", "BaseImageFName","DepthImageFName"])
    
    for depth_zipname, image_zip_name  in image_depth_set_map:
        df_data_list = df_data_list.append(create_csvfile_new(base_dir, depth_zipname, image_zip_name))
    
    df_data_list.loc[:,'BGType'] = df_data_list.loc[:,'BGImageName'].apply(lambda x: re.split(r'_', str(x))[0] )
    df_data_list.loc[:,'BGType'] = df_data_list.loc[:,'BGType'].apply(lambda x: re.sub(r'[0-9]{1,2}\.jpg','', str(x)))
    df_data_list.loc[:,'BGType'] = df_data_list.loc[:,'BGType'].apply(lambda x: re.sub(r'meetin','meetingroom', str(x)))

    return df_data_list

"""
    Generates training/test split for the above CSV
"""
def generate_train_test(df_data_list, split_pct=0.7):
    df_train = df_data_list.sample(frac=split_pct)
    df_train = df_train.reset_index()
    df_train.drop('index',axis=1,inplace=True)
    df_test = pd.DataFrame(columns=list(df_train.columns))
    df_test=df_data_list.loc[~df_data_list['ImageName'].isin(df_train['ImageName'].values),:]
    df_test = df_test.reset_index()
    df_test.drop('index',axis=1,inplace=True)
    return df_train, df_test

"""
    Split any dataframe into training and test sets as per split_pct
"""
def split_df(df_data_list, split_pct=0.7):
    full_set = df_data_list.index.values
    train_set = df_data_list.sample(frac=split_pct).index.values
    test_set  = [ index_val for index_val in full_set if index_val not in train_set]

    df_train = df_data_list.loc[train_set].reset_index()
    df_train.drop('index',axis=1,inplace=True)

    df_test = df_data_list.loc[test_set].reset_index()
    df_test.drop('index',axis=1,inplace=True)
    return df_train, df_test
