"""
    Custom DataSet class and utility methods based on standard DataSetFolder
    but adapted to load and handle tiny-imagenet
    Procedure to invoke/use this dataset:
        imgnetloader.generate_timgnet_train_test_data(base_path, 
                                                    split_pct=0.7, 
                                                    train_transform=None, 
                                                    test_transform=None )
    which will return a shuffled dataset(shuffled across 'val' and 'train' folders)  
"""


from torchvision.datasets.vision import VisionDataset

from PIL import Image

import os
import os.path
import numpy as np
from torch import randperm

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def get_test_dataset(test_dir, class_to_idx, instances, extensions=None, is_valid_file=None):
    #instances = []
    #directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for line in open( test_dir + '/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        img_name = "images/"+img_name
        path = os.path.join(test_dir, img_name)
        #print(path, class_id)
        if is_valid_file(path):
            item = path, class_to_idx[class_id]
            instances.append(item)
            #print(item)
    return instances

def make_dataset(directory, test_dir, class_to_idx, extensions=None, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        #print(directory, target_class)
        target_dir = os.path.join(directory, target_class)
        
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    if (test_dir != None):
        instances = get_test_dataset(test_dir, class_to_idx, instances, extensions=extensions, is_valid_file=None)

    return instances

class MyDatasetFolder(VisionDataset):
    def __init__(self, root, samples, class_to_idx, loader,  extensions=None, test_root=None,transform=None,
                 target_transform=None, is_valid_file=None):
        super(MyDatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        #classes, class_to_idx = self._find_classes(self.root)
        #self.test_root = test_root
        #samples = make_dataset(self.root, self.test_root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = list(class_to_idx.keys())
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            image_np = np.array(sample)
            sample = self.transform(image=image_np)['image']
            #self.transform(image=img)['image']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_id_dictionary(path):
    id_dict = {}
    for i, line in enumerate(open( path + '/wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i
    return id_dict

def get_class_to_id_dict(path):
    id_dict = get_id_dictionary(path)
    all_classes = {}
    result = {}
    for i, line in enumerate(open( path + '/words.txt', 'r')):
        n_id, word = line.split('\t')[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (key, all_classes[key])  
    return result


def generate_timgnet_train_test_data(base_path, split_pct=0.7, train_transform=None, test_transform=None ):
    train_path = os.path.join(base_path+"train/")
    test_path = os.path.join(base_path+"val/")
    id_dict = get_id_dictionary(base_path)#get_class_to_id_dict(base_path)
    #print((id_dict))
    sample_list = make_dataset(train_path, test_path, id_dict, IMG_EXTENSIONS, None)
    #print(len(sample_list))
    sample_len = len(sample_list)
    #split_pct = 0.7
    lengths = [int(sample_len*split_pct), int(sample_len*(1-split_pct))]
    indices = randperm(sum(lengths)).tolist()
    sample_train  =[sample_list[idx ] for idx in indices[:int(sample_len*split_pct)]]
    sample_test  =[sample_list[idx ] for idx in indices[int(sample_len*split_pct):]]

    train_data=MyDatasetFolder(train_path, sample_train, id_dict,
                pil_loader,
                IMG_EXTENSIONS,
                test_root=test_path,
                transform=train_transform,
                target_transform=None,
                is_valid_file=None)
    
    test_data=MyDatasetFolder(train_path, sample_test, id_dict,
                pil_loader,
                IMG_EXTENSIONS,
                test_root=test_path,
                transform=test_transform,
                target_transform=None,
                is_valid_file=None)
    
    return train_data,test_data



class MyImageFolder(MyDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, test_root=None,transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(MyImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          test_root=test_root,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples