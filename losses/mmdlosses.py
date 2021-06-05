import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, OneCycleLR, MultiStepLR, CyclicLR
from RekogNizer import fileutils
import wandb
from tqdm import tqdm
from torchsummary import summary
#from torchlars import LARS
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torchvision
from RekogNizer import lrfinder
from torch.optim.optimizer import Optimizer
from torch._six import inf
from kornia.losses import ssim

mask_mean = 0.0579508
mask_std = 0.001662
depth_mean = 0.3679109, 
depth_std = 0.03551773

def simple_unnormalize(input, mean, std):
    input = (input * std)+mean

def simple_normalize(input, mean, std):
    input = (input - mean)*std

def construct_criterion(input_target_tuple, loss_func_list, normalize_target=False, mean=0., std=1.):
    if normalize_target == True:
        simple_normalize(input_target_tuple[1], mean, std)

    total_loss = 0.
    for loss_func in loss_func_list:
        total_loss += loss_func(input_target_tuple[0], input_target_tuple[1])
    return total_loss

def LocalPixelLossL1(input, target):
       
    out = nn.L1Loss(reduction='none')(input, target)
    out = out * target.expand_as(out)
    
    # expand_as because weights are prob not defined for mini-batch
    loss = out.mean() # or sum over whatever dimensions
    return loss

def LocalFocalLoss(input, target):
    return FocalLoss()(input, target)

def LocalPixelLoss(input, target):
    #print(torch.mean(input), torch.mean(target))#, torch.mean(out))
    out = nn.MSELoss(reduction='none')(input, target)
    out = out * target.expand_as(out)
    
    # expand_as because weights are prob not defined for mini-batch
    loss = out.mean() # or sum over whatever dimensions
    return loss

def LocalL1Loss(input, target, reduction='mean'):
    return nn.L1Loss(reduction=reduction)(input, target)

def LocalRMSELoss(input, target, reduction='mean'):
    return torch.sqrt(nn.MSELoss(reduction=reduction)(input, target))

def LocalBCELoss(input, target):
    return nn.BCEWithLogitsLoss()(input, target)

def LocalSSIMLoss(input, target,reduction='mean'):
    return ssim(input, target, 11, reduction=reduction)


def relative_dice_loss(input, target):
    return (dice_loss(input, target)/dice_loss(target, target)) - 1
    #final_loss = (input_loss/target_loss)
    #return final_loss

def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.

    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma=1):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.mean()




def bce_with_bce(mask_tuple, depth_tuple):
    mask_loss = LocalBCELoss(mask_tuple[0], mask_tuple[1])
    depth_loss = nn.BCEWithLogitsLoss()(depth_tuple[0], torch.sigmoid(depth_tuple[1]))
    #(torch.sqrt(nn.MSELoss()(depth_tuple[0], depth_tuple[1]))+ssim(depth_tuple[0], depth_tuple[1], 11))/2
    #torch.sqrt(nn.MSELoss()(depth_tuple[0], depth_tuple[1]))
    #ssim(depth_tuple[0], depth_tuple[1], 11)
    return mask_loss,depth_loss


def bce_dice_with_rmse_ssim(mask_tuple, depth_tuple):
    mask_loss = (dice_loss(mask_tuple[0], mask_tuple[1]) + LocalBCELoss(mask_tuple[0], mask_tuple[1]))/2
    depth_loss = (torch.sqrt(nn.MSELoss()(depth_tuple[0], depth_tuple[1]))+ssim(depth_tuple[0], depth_tuple[1], 11))/2
    #torch.sqrt(nn.MSELoss()(depth_tuple[0], depth_tuple[1]))
    #ssim(depth_tuple[0], depth_tuple[1], 11)
    return mask_loss,depth_loss

def bce_dice_with_rmse(mask_tuple, depth_tuple):
    mask_loss = (dice_loss(mask_tuple[0], mask_tuple[1]) + 
                LocalBCELoss(mask_tuple[0], mask_tuple[1]))/2
    
    depth_loss = torch.sqrt(nn.MSELoss()(depth_tuple[0], depth_tuple[1]))
    #torch.sqrt(nn.MSELoss()(depth_tuple[0], depth_tuple[1]))
    #ssim(depth_tuple[0], depth_tuple[1], 11)
    return mask_loss,depth_loss

def bce_dice_with_ssim(mask_tuple, depth_tuple):
    mask_loss = (dice_loss(mask_tuple[0], mask_tuple[1]) + 
                LocalBCELoss(mask_tuple[0], mask_tuple[1]))/2
    depth_loss = ssim(depth_tuple[0], depth_tuple[1], 11, reduction='mean')
    #torch.sqrt(nn.MSELoss()(depth_tuple[0], depth_tuple[1]))
    #ssim(depth_tuple[0], depth_tuple[1], 11)
    return mask_loss,depth_loss


def dice_with_ssim(mask_tuple, depth_tuple):
    mask_loss = dice_loss(mask_tuple[0], mask_tuple[1])
    depth_loss = ssim(depth_tuple[0], depth_tuple[1], 11, reduction='mean')
    #torch.sqrt(nn.MSELoss()(depth_tuple[0], depth_tuple[1]))
    #ssim(depth_tuple[0], depth_tuple[1], 11)
    return mask_loss,depth_loss

def dice_with_rmse(mask_tuple, depth_tuple):
    mask_loss = dice_loss(mask_tuple[0], mask_tuple[1])
    depth_loss = torch.sqrt(nn.MSELoss()(depth_tuple[0], depth_tuple[1]))
    #ssim(depth_tuple[0], depth_tuple[1], 11, reduction='mean')
    #torch.sqrt(nn.MSELoss()(depth_tuple[0], depth_tuple[1]))
    #ssim(depth_tuple[0], depth_tuple[1], 11)
    return mask_loss,depth_loss

def bce_with_rmse_ssim(mask_tuple, depth_tuple):
    mask_loss = LocalBCELoss(mask_tuple[0], mask_tuple[1])
    depth_loss = (1-ssim(depth_tuple[0], depth_tuple[1], 11, reduction='mean'))+torch.sqrt(nn.MSELoss()(depth_tuple[0], depth_tuple[1]))
    #torch.sqrt(nn.MSELoss()(depth_tuple[0], depth_tuple[1]))
    #ssim(depth_tuple[0], depth_tuple[1], 11)
    return mask_loss,depth_loss



def bce_with_ssim(mask_tuple, depth_tuple):
    mask_loss = LocalBCELoss(mask_tuple[0], mask_tuple[1])
    depth_loss = ssim(depth_tuple[0], depth_tuple[1], 11, reduction='mean')
    #torch.sqrt(nn.MSELoss()(depth_tuple[0], depth_tuple[1]))
    #ssim(depth_tuple[0], depth_tuple[1], 11)
    return mask_loss,depth_loss


def bce_with_rmse(mask_tuple, depth_tuple):
    mask_loss = LocalBCELoss(mask_tuple[0], mask_tuple[1])
    depth_loss = torch.sqrt(nn.MSELoss()(depth_tuple[0], depth_tuple[1]))
    #ssim(depth_tuple[0], depth_tuple[1], 11)
    return mask_loss,depth_loss


def joint_ssim_loss(mask_tuple, depth_tuple):
    mask_ssim_loss = ssim(mask_tuple[0], mask_tuple[1], 11)
    depth_ssim_loss = ssim(depth_tuple[0], depth_tuple[1], 11)
    return (mask_ssim_loss+depth_ssim_loss)/2


def rmse_loss(mask_tuple, depth_tuple):
    return torch.sqrt(nn.MSELoss()(mask_tuple[0], mask_tuple[1])), torch.sqrt(nn.MSELoss()(depth_tuple[0], depth_tuple[1]))


def gradient_loss(gen_frames, gt_frames, alpha=1):
    def gradient(x):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)

        h_x = x.size()[-2]
        w_x = x.size()[-1]
        # gradient step=1
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx, dy = right - left, bottom - top 
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy

    # gradient
    gen_dx, gen_dy = gradient(gen_frames)
    gt_dx, gt_dy = gradient(gt_frames)
    #
    grad_diff_x = torch.abs(gt_dx - gen_dx)
    grad_diff_y = torch.abs(gt_dy - gen_dy)

    # condense into one tensor and avg
    return torch.mean(grad_diff_x ** alpha + grad_diff_y ** alpha)

# def get_loss(loss):
#     if loss[0] == 'dice':
#         print('dice')
#         return dice_loss
#     elif loss[0] == 'focal':
#         print('focal')
#         return w(FocalLoss(loss[1]))
#     else:
#         print('bce')
#         return w(nn.BCEWithLogitsLoss())