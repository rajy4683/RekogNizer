import torch
import torchvision
import torchvision.transforms as transforms
from RekogNizer import hyperparams

from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, Cutout, MotionBlur
from albumentations import (
    HorizontalFlip, Compose, RandomCrop, Cutout,Normalize, HorizontalFlip, RandomBrightnessContrast,
    Resize,RandomSizedCrop, MotionBlur,MultiplicativeNoise,InvertImg, IAAFliplr,
	IAAPerspective,
)
from albumentations.pytorch import ToTensor
import random
import albumentations as A

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import os
import sys
import numpy as np
import pandas as pd
from torchvision import datasets
from RekogNizer import imgnetloader
from RekogNizer import basemodelclass
from RekogNizer import hyperparams
from torch.utils.data import Dataset
import shutil
from tqdm import tqdm
from zipfile import ZipFile
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import kornia

import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from PIL import Image
from tqdm import tqdm
from io import BytesIO
#from torchdata import torchdata
from torch.utils.data import DataLoader
#random.seed(42)

#cv2.imread('images/parrot.jpg')

flip = A.Compose([
    #A.IAAFliplr(p=1,always_apply=True),
    #A.ToGray(p=1),
    #A.ElasticTransform(p=1),
    #A.Solarize(p=1),
    A.CLAHE(p=1),
    A.HorizontalFlip(p=1),
    #A.Rotate(limit=30,p=1),
    #A.RandomRotate90(p=1),
    A.ChannelShuffle(p=1),
    #A.RandomBrightnessContrast(p=1),    
    #A.RandomGamma(p=1),    
    ## A.CLAHE(p=1),    
], p=1)

resize_bg = A.Compose([
    A.Resize(250,250,always_apply=True,p=1),
    #A.CLAHE(p=1),
    #A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1),
], p=1)

handle_depth = A.Compose([
    A.Resize(250,250,always_apply=True,p=1),
    A.ToGray(always_apply=True,p=1)
    #A.CLAHE(p=1),
    #A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1),
], p=1)

strong = A.Compose([
    A.ChannelShuffle(p=1),
], p=1)

base_tensor = A.Compose([
    A.pytorch.ToTensor(),
], p=1)

def resize_bg_nonNorm(h,w):
    return A.Compose([
    A.Resize(h,w,always_apply=True,p=1),
    A.pytorch.ToTensor(),
    #A.ToGray(always_apply=True,p=1)
    #A.CLAHE(p=1),
    #A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1),
], p=1)

def resize_bg(h,w,mean,std):
    return A.Compose([
    A.Resize(h,w,always_apply=True,p=1),
    A.Normalize(
        mean=mean, std=std
        ),
    A.pytorch.ToTensor(),
    #A.ToGray(always_apply=True,p=1)
    #A.CLAHE(p=1),
    #A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1),
], p=1)


def get_train_test_loader(train_chunk_start, 
                          train_chunk_end, 
                          train_batch_size,
                          test_chunk_start, 
                          test_chunk_end,
                          test_batch_size,
                          image_size,
                          unzip_orig_files=True):
      
    csv_file ='/content/drive/My Drive/EVA4/tsai/S15EVA4/FinalDataSet/DepthMapDataSetTrain.csv'
    test_csv_file ='/content/drive/My Drive/EVA4/tsai/S15EVA4/FinalDataSet/DepthMapDataSetTest.csv'
    size_hw = image_size
    root_dir = '/content/drive/My Drive/EVA4/tsai/S15EVA4/'
    #size_hw = (224,224)
    train_dest_dir="/content/train"
    test_dest_dir="/content/test"
    transform_base=resize_bg(size_hw[0],size_hw[1],[0.56632738, 0.51567622, 0.45670792], [0.1076622, 0.10650349, 0.12808967] )
    transform_bg=resize_bg(size_hw[0],size_hw[1],[0.57469445, 0.52241555, 0.45992244], [0.11322354, 0.11195428, 0.13441683])
    transform_mask = resize_bg_nonNorm(size_hw[0],size_hw[1])
    transform_depth = resize_bg_nonNorm(size_hw[0],size_hw[1])


    train_loader = retrieve_dataloader(csv_file, 
                                        root_dir,
                                        train_chunk_start,
                                        train_chunk_end,
                                        train_batch_size,
                                        transform_base,
                                        transform_bg,
                                        transform_mask,
                                        transform_depth,
                                        unzip_orig_files=unzip_orig_files, 
                                        dest_dir=train_dest_dir)

    test_loader = retrieve_dataloader(test_csv_file, 
                                      root_dir,
                                      test_chunk_start,
                                      test_chunk_end,
                                      test_batch_size,
                                      transform_base,
                                      transform_bg,
                                      transform_mask,
                                      transform_depth,
                                      unzip_orig_files=unzip_orig_files, 
                                      dest_dir=test_dest_dir)
    return train_loader, test_loader

"""
Utility function to retrieve dataloader.
If  
  Inputs
    csv_file: Train/Test CSV file
    root_dir: Root dir containing the zip ,
    chunk_start: Starting slice for the dataset
    chunk_end: Ending Slice for the dataset
    batch_size: Size of each batch
    transform_base: Transform to apply to the fg_bg images
    transform_bg: Transform to apply to the bg images
    transform_mask: Transform to apply to the mask images
    transform_depth: Transform to apply to the depth images
    transform_opt: Actual Spatial/Color transformations (This will be applied in combination to {fg_bg, mask} and {bg, depth})
    unzip_orig_files(bool): Whether to unzip original datafiles
    dest_dir: ""
"""
def retrieve_dataloader(csv_file,
                        root_dir,
                        chunk_start,
                        chunk_end,
                        batch_size,
                        transform_base,
                        transform_bg,
                        transform_mask,
                        transform_depth,
                        transform_opt=None,
                        unzip_orig_files=True, 
                        dest_dir="/content/train"):

    if unzip_orig_files == True:
      extract_data_files(csv_file, root_dir, chunk_start,chunk_end, dest_dir=dest_dir)
    
    depth_dataset = DepthMaskDataSet(csv_file, dest_dir, chunk_start,chunk_end, read_zip=False,
                                 transform_base= transform_base,  #mmddataloader.resize_bg(size_hw[0],size_hw[1],[0.56632738, 0.51567622, 0.45670792], [0.1076622, 0.10650349, 0.12808967] ),
                                 transform_bg= transform_bg, #mmddataloader.resize_bg(size_hw[0],size_hw[1],[0.57469445, 0.52241555, 0.45992244], [0.11322354, 0.11195428, 0.13441683]),
                                 transform_mask = transform_mask, #mmddataloader.resize_bg_nonNorm(size_hw[0],size_hw[1]),
                                 transform_depth = transform_depth, #mmddataloader.resize_bg_nonNorm(size_hw[0],size_hw[1]),
                                 transform_opt= transform_opt, #mmddataloader.flip, 
                                 )
    kwargs= { 'num_workers':8,'pin_memory': True}
    data_loader = DataLoader(depth_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return data_loader

def extract_data_files_old(csv_file, root_dir, start,end, dest_dir="/content/S15EVA4"):
    depthmask_csv = pd.read_csv(csv_file).loc[start:end,:]
    image_file_zip_dict = {val:ZipFile(os.path.join(root_dir,val)) 
                                for val in depthmask_csv['BaseImageFName'].unique()}
    depth_zip_dict = {val:ZipFile(os.path.join(root_dir,val)) 
                                for val in depthmask_csv['DepthImageFName'].unique()}
    bg_image_zip_dict = ZipFile("/content/drive/My Drive/EVA4/tsai/S15EVA4/FinalDataSet/bg_images.zip")
    pbar = tqdm(range(len(depthmask_csv)))

    #for idx in range(len(depthmask_csv)):
    for batch_idx,idx in enumerate(pbar):
        base_img_name = depthmask_csv.iloc[idx, 0]
        mask_img_name = depthmask_csv.iloc[idx, 1]
        depth_img_name = depthmask_csv.iloc[idx, 2]
        bg_image_name = depthmask_csv.iloc[idx, 3]
        base_img_zip = image_file_zip_dict[depthmask_csv.iloc[idx, 4]]
        depth_img_zip = depth_zip_dict[depthmask_csv.iloc[idx, 5]]
        #self.depthmask_csv = pd.read_csv(csv_file)
        
        base_img = base_img_zip.extract(base_img_name, dest_dir)
        bg_img = bg_image_zip_dict.extract(bg_image_name, dest_dir)

        ### GT labels 
        mask_img = base_img_zip.extract(mask_img_name, dest_dir)
        depth_img = depth_img_zip.extract(depth_img_name, dest_dir)
    print("Total file count:{}".format(len(glob.glob(dest_dir+"/*jpg"))))


def extract_data_files(csv_file, root_dir, start,end, dest_dir="/content/S15EVA4"):
    depthmask_csv = pd.read_csv(csv_file).loc[start:end,:]
    image_file_zip_dict = {val:ZipFile(os.path.join(root_dir,val)) 
                                for val in depthmask_csv['BaseImageFName'].unique()}
    depth_zip_dict = {val:ZipFile(os.path.join(root_dir,val)) 
                                for val in depthmask_csv['DepthImageFName'].unique()}
    baseimage_groups = depthmask_csv.groupby('BaseImageFName')
    depthimage_groups = depthmask_csv.groupby('DepthImageFName')
    bg_image_zip_dict = ZipFile("/content/drive/My Drive/EVA4/tsai/S15EVA4/FinalDataSet/bg_images.zip")
    
    #pbar = tqdm(range(len(depthmask_csv)))
    print("Extracting image and mask files")
    for zip_name in list(baseimage_groups.groups.keys()):
        zip_obj = ZipFile(os.path.join(root_dir,zip_name))
        img_mask_arr = np.hstack(depthmask_csv.loc[baseimage_groups.groups[zip_name],['ImageName','MaskName']].values)
        pbar = tqdm(img_mask_arr)
        pbar.set_description(zip_name)
        for batch_idx, image_name in enumerate(pbar):
            zip_obj.extract(image_name,dest_dir)
            pbar.set_description(zip_name + " "+image_name)
    
    print("Extracting depth files")
    for zip_name in list(depthimage_groups.groups.keys()):
        zip_obj = ZipFile(os.path.join(root_dir,zip_name))
        img_mask_arr = np.hstack(depthmask_csv.loc[depthimage_groups.groups[zip_name],['Depthname']].values)
        pbar = tqdm(img_mask_arr)
        pbar.set_description(zip_name)
        for batch_idx, image_name in enumerate(pbar):
            zip_obj.extract(image_name,dest_dir)
            pbar.set_description(zip_name + " "+image_name)

    print("Extracting bg files")
    for depth_image_name in depthmask_csv['BGImageName'].unique():
        bg_image_zip_dict.extract(depth_image_name,dest_dir)

    print("Total file count:{} ".format(len(glob.glob(dest_dir+"/*jpg"))))

from multiprocessing import Pool
#from multiprocessing.pool import ThreadPool

def extract_individual_zipfile(file_name_list, root_dir, zip_name, dest_dir):
    zip_obj = ZipFile(os.path.join(root_dir,zip_name))
    print("Zip Name:{} Total Files:{}".format(zip_name, len(file_name_list)))
    #pbar = tqdm(file_name_list)
    #pbar.set_description(zip_name)
    for batch_idx, image_name in enumerate(file_name_list):
        zip_obj.extract(image_name,dest_dir)
        #pbar.update(1)

def extract_data_files_mt(csv_file, root_dir, start,end, dest_dir="/content/S15EVA4"):
    depthmask_csv = pd.read_csv(csv_file).loc[start:end,:]
    image_file_zip_dict = depthmask_csv['BaseImageFName'].unique()
    depth_zip_dict = depthmask_csv['DepthImageFName'].unique()
    baseimage_groups = depthmask_csv.groupby('BaseImageFName')
    depthimage_groups = depthmask_csv.groupby('DepthImageFName')
    bg_image_zip_dict = ZipFile("/content/drive/My Drive/EVA4/tsai/S15EVA4/FinalDataSet/bg_images.zip")
    
    #pbar = tqdm(range(len(depthmask_csv)))
    total_zip_file_count = len(image_file_zip_dict) + len(depth_zip_dict) + 1
    #pool = ThreadPool( total_zip_file_count )
    pool = Pool( total_zip_file_count )
    print("Extracting image and mask files Total:{}".format(total_zip_file_count))
    offset = 0

    #pbar= tqdm(total=total_zip_file_count)#, position=pos, desc=zip_name)
    for idx, zip_name in enumerate(list(baseimage_groups.groups.keys())):
        img_mask_arr = np.hstack(depthmask_csv.loc[baseimage_groups.groups[zip_name],['ImageName','MaskName']].values)
        pool.apply_async(extract_individual_zipfile, args=(img_mask_arr, root_dir, zip_name, dest_dir))

    offset = len(list(image_file_zip_dict)) - 1
    for idx,zip_name in enumerate(list(depthimage_groups.groups.keys())):
        zip_obj = ZipFile(os.path.join(root_dir,zip_name))
        img_mask_arr = np.hstack(depthmask_csv.loc[depthimage_groups.groups[zip_name],['Depthname']].values)
        pool.apply_async(extract_individual_zipfile, args=(img_mask_arr, root_dir, zip_name, dest_dir))

    pool.close()
    pool.join()
    
    print("Extracting bg files")
    for depth_image_name in depthmask_csv['BGImageName'].unique():
        bg_image_zip_dict.extract(depth_image_name,dest_dir)

    print("Total file count:{} ".format(len(glob.glob(dest_dir+"/*jpg"))))


def update_mean_variance(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2

    return (count, mean, M2)

# Retrieve the mean, variance and sample variance from an aggregate
def finalize_mean_variance(existingAggregate):
    (count, mean, M2) = existingAggregate
    if count < 2:
        return float('nan')
    else:
       (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
       return (mean, variance, sampleVariance)



class DepthMaskDataSet(Dataset):
    """Depth and Mask prediction Dataset.
    """


    def __init__(self, csv_file, root_dir, start,end, read_zip=False,
                 transform_base=None,
                 transform_bg=None,
                 transform_mask=None, 
                 transform_depth=None,
                 transform_opt=None,
                 transform_input_only=None):
        """
        Args:
            csv_file (string): Path to the csv file with depth_images, base_images, mask_images and respective zip files.
            root_dir (string): Base dir containing main zip files for depth and image/mask zip files.
            transform (tuple of callable, optional): Optional transform to be applied on depth and mask.
        """
        self.depthmask_csv = pd.read_csv(csv_file).loc[start:end,:]
        #self.depthmask_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform_base = transform_base
        self.transform_bg = transform_bg
        self.transform_mask = transform_mask
        self.transform_depth = transform_depth
        self.transform_input_only = transform_input_only
        self.transform_opt = transform_opt
        self.read_zip = read_zip
        if (self.read_zip == True):
            self.image_file_zip_dict = {val:ZipFile(os.path.join(self.root_dir,val)) 
                                       for val in self.depthmask_csv['BaseImageFName'].unique()}
            self.depth_zip_dict = {val:ZipFile(os.path.join(self.root_dir,val)) 
                                       for val in self.depthmask_csv['DepthImageFName'].unique()}
        
        self.bg_image_zip_dict = ZipFile("/content/drive/My Drive/EVA4/tsai/S15EVA4/FinalDataSet/bg_images.zip")
        # if transform:
        #     self.transform_base = transform
        #     #self.transform_mask = transform[1]
        # if target_transform
        #     self.transform_base = None
        #     #self.transform_mask = None

    def __len__(self):
        return len(self.depthmask_csv)
    """
        Returns {
            inputs:[overlayed_image, background_image], 
            targets:[gt_depth_map, gt_mask]
            }
        Index(['ImageName', 'MaskName', 'Depthname', 'BGImageName', 'BaseImageFName',
       'DepthImageFName', 'BGType']
    """
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        base_img_name = self.depthmask_csv.iloc[idx, 0]
        mask_img_name = self.depthmask_csv.iloc[idx, 1]
        depth_img_name = self.depthmask_csv.iloc[idx, 2]
        bg_image_name = self.depthmask_csv.iloc[idx, 3]
        
        #ZipFile(os.path.join(self.root_dir,
        #                        self.depthmask_csv.iloc[idx, 4]))
        
        # ZipFile(os.path.join(self.root_dir,
        #                         self.depthmask_csv.iloc[idx, 5]))
        #print(base_img_name,mask_img_name,depth_img_name,bg_image_name )
        #print(base_img_zip,depth_img_zip)
        ### GT original inputs
        try:
            if (self.read_zip == True):
                base_img_zip = self.image_file_zip_dict[self.depthmask_csv.iloc[idx, 4]]
                depth_img_zip = self.depth_zip_dict[self.depthmask_csv.iloc[idx, 5]]
                base_img = np.array(Image.open(BytesIO((base_img_zip.read(base_img_name)))))
                bg_img = np.array(Image.open(BytesIO(self.bg_image_zip_dict.read(bg_image_name))))
                
                ### GT labels 
                mask_img = np.array(Image.open(BytesIO((base_img_zip.read(mask_img_name)))))
                depth_img = np.array(Image.open(BytesIO((depth_img_zip.read(depth_img_name)))))
            else:            
                base_img = np.array(Image.open(os.path.join(self.root_dir,base_img_name)))
                bg_img = np.array(Image.open(os.path.join(self.root_dir,bg_image_name)))
                
                ### GT labels 
                mask_img = np.array(Image.open(os.path.join(self.root_dir,mask_img_name)))
                depth_img = np.array(Image.open(os.path.join(self.root_dir,depth_img_name)))

            #return sample
        except KeyError as key_err:
            print(key_err)
            return None
        if self.transform_input_only:
          output_base = self.transform_input_only(image=base_img, mask=bg_img)
          base_img = output_base['image']
          bg_img = output_base['mask']
         
        if self.transform_opt:
          output_base = self.transform_opt(image=base_img, mask=mask_img)
          base_img = output_base['image']
          mask_img = output_base['mask']
          output_bg = self.transform_opt(image=bg_img, mask=depth_img)
          bg_img = output_bg['image']
          depth_img = output_bg['mask']
        
        #if self.transform_base:
        base_img = self.transform_base(image=base_img)['image']
        bg_img = self.transform_bg(image=bg_img)['image']
        #if self.transform_target:
        mask_img = self.transform_mask(image=mask_img)['image']
        depth_img = self.transform_depth(image=depth_img)['image']



        if isinstance(base_img, torch.Tensor):
            final_input = torch.cat((base_img,bg_img),axis=0)
            # final_input = final_input.reshape(final_input.shape[2],
            #                                   final_input.shape[0],
            #                                   final_input.shape[1],
            #                                   )
        elif isinstance(base_img, np.ndarray):
            final_input = np.concatenate((base_img,bg_img),axis=2)
        #     final_input = final_input.reshape(final_input.shape[2],
        #                                       final_input.shape[0],
        #                                       final_input.shape[1],
        #                                       )
        mask_img = mask_img.reshape(1,
                                    mask_img.shape[0],
                                    mask_img.shape[1],
                                    )
        
        depth_img = depth_img.reshape(1,
                                    depth_img.shape[0],
                                    depth_img.shape[1],
                                    )

        #sample = {'input':final_input, 'output':list([mask_img, depth_img]) }
        #sample = {'input':{base_img_name:base_img, bg_image_name:bg_img}, 
        #          'output':{mask_img_name:mask_img, depth_img_name:depth_img} }
        # sample = {'input':{base_img_name:base_img}, 
        #           'output':{mask_img_name:mask_img, depth_img_name:depth_img} }
        return final_input, mask_img, depth_img
