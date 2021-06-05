import torch
import torchvision
import torchvision.transforms as transforms
from RekogNizer import hyperparams

from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, Cutout, MotionBlur
from albumentations import (
    HorizontalFlip, Compose, RandomCrop, Cutout,Normalize, HorizontalFlip, RandomBrightnessContrast,
    Resize,RandomSizedCrop, MotionBlur,InvertImg, IAAFliplr,
	IAAPerspective,
)
from albumentations.pytorch import ToTensor
import random

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import os
import sys
import numpy as np
import pandas as pd
from torchvision import datasets
from RekogNizer import imgnetloader
from torch.utils.data import Dataset
from PIL import Image

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle




"""
    Dataset class for the DepthDataSet.
    It consists of following images:
    1. Images with Only Background E.g: Malls, classrooms, college_outdoors, lobbies etc (bg_image) Resolution: 250x250
    2. Images with an object/person overlayed randomly on a Background (fg_bg_images). Resolution: 250x250
    3. Ground truth of Masked Images of foregroud object/person (mask_images) Resolution: 250x250
    4. Ground truth of Depth Map generated from fg_bg_images. (depth_images) Resolution: 320x240 
    
    The CSV file for the DataSet (DepthMapDataSet.csv) contains following columns:
    "ImageName"      : fg_bg_image    
    "MaskName"       : mask_image
    "Depthname",     : depth_image
    "BGImageName"    : bg_image
    "BaseImageFName" : Zip file containing fg_bg_images and mask_images
    "DepthImageFName": Zip file containing depth_images
    "BGType"         : Class to which the bg_image belongs
    "BGImageFName"   : Zip file containing bg_images

    Total Count of entries in DepthMapDataSet.csv = 483820

    Following additional CSVs are provided:Randomized Training and Test CSVs are provided:
    1. DepthMapDataSetTest.csv: Randomized Test Samples (30%)
    2. DepthMapDataSetTrain.csv: Randomized Training Samples (70%)
    3. DepthMapDataSetSample.csv: Randomized 500 samples

    ImageType,      Count,      Dimension,  Channel Space,  ChannelWise Mean,                    ChannelWise StdDev
    fg_bg_images,   484320,     250x250x3,  RGB,           [0.56632738, 0.51567622, 0.45670792]  [0.1076622,  0.10650349, 0.12808967]
    bg_images,      484320,     250x250x3,  RGB,           [0.57469445, 0.52241555, 0.45992244]  [0.11322354, 0.11195428, 0.13441683]
    mask_images,    484320,     250x250x3,  RGB,           [0.05795104, 0.05795104, 0.05795104]  [0.02640032, 0.02640032, 0.02640032]
    depth_images,   484320,     320x240x3,  RGB,           [0.61635181, 0.21432114, 0.50569604]  [0.09193359, 0.07619106, 0.04919082]


"""

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class DepthMaskDataSet(Dataset):
    """Depth and Mask prediction Dataset.
    """
    def __init__(self, csv_file, 
                root_dir, 
                bg_image_path="/content/drive/My Drive/EVA4/tsai/S15EVA4/bg_images.zip",
                transform=None, target_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with depth_images, base_images, mask_images and respective zip files.
            root_dir (string): Base dir containing main zip files for depth and image/mask zip files.
            bg_image_path
            transform : Optional transform to be applied on fg_bg_image and bg_image.
            transform_target : Optional transform to be applied on mask_image and depth_image.
        """
        self.depthmask_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform_base = transform
        self.transform_target = target_transform
        self.image_file_zip_dict = {val:ZipFile(os.path.join(self.root_dir,val)) 
                                    for val in self.depthmask_csv['BaseImageFName'].unique()}
        self.depth_zip_dict = {val:ZipFile(os.path.join(self.root_dir,val)) 
                                    for val in self.depthmask_csv['DepthImageFName'].unique()}
        
        self.bg_image_zip_dict = ZipFile("/content/drive/My Drive/EVA4/tsai/S15EVA4/bg_images.zip")
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
        base_img_zip = self.image_file_zip_dict[self.depthmask_csv.iloc[idx, 4]]
        #ZipFile(os.path.join(self.root_dir,
        #                        self.depthmask_csv.iloc[idx, 4]))
        depth_img_zip = self.depth_zip_dict[self.depthmask_csv.iloc[idx, 5]]
        # ZipFile(os.path.join(self.root_dir,
        #                         self.depthmask_csv.iloc[idx, 5]))
        #print(base_img_name,mask_img_name,depth_img_name,bg_image_name )
        #print(base_img_zip,depth_img_zip)
        ### GT original inputs
        try:
            base_img = np.array(Image.open(BytesIO((base_img_zip.read(base_img_name)))))
            bg_img = np.array(Image.open(BytesIO(self.bg_image_zip_dict.read(bg_image_name))))
            
            ### GT labels 
            mask_img = np.array(Image.open(BytesIO((base_img_zip.read(mask_img_name)))))
            depth_img = np.array(Image.open(BytesIO((depth_img_zip.read(depth_img_name)))))
            
            #return sample
        except KeyError as key_err:
            print(key_err)
            return None

        if self.transform_base:
           base_img = self.transform_base(image=base_img)['image']
           bg_img = self.transform_base(image=bg_img)['image']
        if self.transform_target:
           mask_img = self.transform_target(image=mask_img)['image']
           depth_img = self.transform_base(image=depth_img)['image']

           #mask_img =  self.transform_mask(mask_img)
        #if self.transform_mask:
        #   
        sample = {'input':list([base_img, bg_img]), 'output':list([mask_img, depth_img]) }
        return sample


class MyCIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(MyCIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


## Dummy wrapper around 
def get_dataloader(dataset, batch_size, shuffle=True, num_workers=2):
    return torch.utils.data.DataLoader(dataset, batch_size,
                                         shuffle=shuffle, num_workers=num_workers)

def get_default_transforms_cifar10():
    torch.manual_seed(hyperparams.hyperparameter_defaults['seed'])
    patch_size=2
    transform_train = Compose([
    #Cutout(num_holes=1,max_h_size=16,max_w_size=16,always_apply=True,p=1,fill_value=[0.5268*255, 0.5267*255, 0.5328*255]),
    Cutout(num_holes=1,max_h_size=16,max_w_size=16,always_apply=True,p=1,fill_value=[0.4819*255, 0.4713*255, 0.4409*255]),
    IAAFliplr(p=0.5),
    #MotionBlur(blur_limit=7, always_apply=True, p=1),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, always_apply=True, p=1),
    #MultiplicativeNoise(multiplier=1.5, p=1),
    #InvertImg(p=0.5),
    #HorizontalFlip(p=1),
    Normalize(
       mean=[0.4914, 0.4826, 0.44653],
       std=[0.24703, 0.24349, 0.26519],
       ),
    # Normalize(
    #    mean=[0.5268, 0.5267, 0.5328],
    #    std=[0.3485, 0.3444, 0.3447],
    #    ),
    
    ToTensor()
    ])

    transform_test = Compose([
    Normalize(
        mean=[0.4914, 0.4826, 0.44653],
        std=[0.24703, 0.24349, 0.26519],
    ),
    ToTensor()
    ])
    return transform_train, transform_test

"""
    EVA4 S2 Dataset consists of 4 classes:
        1. Large QuadCopters
        2. Small QuadCopters
        3. Winged Drones
        4. Flying Birds
"""

class QDFDataSet(Dataset):
    """Depth and Mask prediction Dataset.
    """

    def __init__(self, csv_file, root_dir="/content",
                 transform=None):
        """
        Args:
            csv_file (string): CSV file containing training/test data.
            root_dir (string): Base dir containing ZipFiles for each class.
            transform (tuple of callable, optional): Optional transform to be applied.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    """
        Returns {
            inputs:[overlayed_image, background_image], 
            targets:[gt_depth_map, gt_mask]
            }
        Index(['FileName', 'DirName', 'Extn', 'Size', 'ClassName', 'Width', 'Height',
       'Orientation'])
    """
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        base_img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 4], self.dataframe.iloc[idx, 0])
        target_val = self.dataframe.iloc[idx, 12]
           

        
        try:
            base_img = np.array(Image.open(os.path.join(self.root_dir,base_img_name)).convert("RGB") )
            ## base_img = np.array(Image.open(BytesIO((base_img_zip.read(base_img_name))))) ## For reading from ZipFile
        except KeyError as key_err:
            print(key_err)
            return None
        if self.transform:
          output_base = self.transform(image=base_img)
          base_img = output_base['image']
  
        return base_img, target_val




def get_normalizer_transform(dataset_type,transforms_type):
    if dataset_type == "imagenet":
        return transforms_type.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif dataset_type == "cifar10":
        return transforms_type.Normalize(mean=[0.4914, 0.4826, 0.44653], std=[0.24703, 0.24349, 0.26519])

def get_minimal_transforms(dataset_type, transforms_type):
    torch.manual_seed(hyperparams.hyperparameter_defaults['seed'])
    norm_transform = get_normalizer_transform(dataset_type,transforms_type)
    transform_train = transforms_type.Compose([
        norm_transform,
        transforms_type.ToTensor()
    ])

    transform_test = transforms_type.Compose([
        norm_transform,
        transforms_type.ToTensor()
    ])
    return transform_train, transform_test


def get_train_test_dataloader_cifar10(transform_train=None, transform_test=None):
    torch.manual_seed(hyperparams.hyperparameter_defaults['seed'])
    
    transform_train_def, transform_test_def = get_default_transforms_cifar10()
    if (transform_train is None):
        transform_train = transform_train_def
    if (transform_test is None):
        transform_test = transform_test_def
    
    trainset = MyCIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = get_dataloader(trainset, hyperparams.hyperparameter_defaults['batch_size'], shuffle=True, num_workers=2)
    testset = MyCIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = get_dataloader(testset, hyperparams.hyperparameter_defaults['batch_size'], shuffle=False, num_workers=2)
    return trainloader, testloader


def get_imagenet_loaders(train_path, test_path, transform_train=None, transform_test=None):
    torch.manual_seed(hyperparams.hyperparameter_defaults['seed'])
    
    train_transform = Compose([
        
        #Resize(32, 32, interpolation=1, always_apply=True, p=1),
        Cutout(num_holes=1,max_h_size=8,max_w_size=8,always_apply=True,p=1,fill_value=[0.485*255, 0.456*255, 0.406*255]),
        IAAFliplr(p=0.5),
        #MotionBlur(blur_limit=7, always_apply=True, p=1),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, always_apply=True, p=1),
		#IAAPerspective(scale=(0.05, 0.1), keep_size=True, always_apply=False, p=0.5),
        Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        ToTensor()
    ])

    test_transform = Compose([
        #MotionBlur(blur_limit=7, always_apply=True, p=1),
        #Resize(32, 32, interpolation=1, always_apply=True, p=1),
        Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        ToTensor()
    ])
    train_data, test_data = imgnetloader.generate_timgnet_train_test_data("/content/t2/", 0.7, train_transform, test_transform )
    print(train_data.transform, test_data.transform)

    kwargs = {'num_workers': 2, 'pin_memory': True}
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=hyperparams.hyperparameter_defaults['batch_size'],shuffle=True,**kwargs)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=hyperparams.hyperparameter_defaults['batch_size'],shuffle=True,**kwargs)
    
    return trainloader, testloader
    
