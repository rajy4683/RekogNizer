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
from RekogNizer import mmdlrfinder
from RekogNizer import mmdlosses
import kornia
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
import time

from RekogNizer.mmdlosses import joint_ssim_loss, rmse_loss, dice_loss
FLAGS = {}
FLAGS['data_dir'] = "/tmp/cifar"
FLAGS['batch_size'] = 8
FLAGS['num_workers'] = 4
FLAGS['learning_rate'] = 0.02
FLAGS['momentum'] = 0.9
FLAGS['num_epochs'] = 20
FLAGS['num_cores'] = 8
FLAGS['log_steps'] = 5
FLAGS['metrics_debug'] = False

def display_samples(model_new, data_loader, device, sample_count=8, loop_count=2, unnormalize=False):
    img = iter(data_loader).next()
    for batch_idx, img in enumerate(data_loader):
        if(batch_idx > loop_count):
            break
        with torch.no_grad():
            output_val = model_new(img['input'].to(device)).to("cpu")
        output_pred = torch.split(output_val,1,dim=1)
        if ( unnormalize == False):
            mask_pred = [kornia.tensor_to_image((output_pred[0][val].to("cpu")*255).byte()) for val in range(sample_count) ]
            depth_pred = [kornia.tensor_to_image((output_pred[1][val].to("cpu")*255).byte()) for val in range(sample_count) ]
            img_shape = img['output'][0][1].shape[1]
            mask_gt = [img['output'][0][val].reshape(img_shape,img_shape) for val in range(sample_count) ]
            depth_gt = [img['output'][1][val].reshape(img_shape,img_shape) for val in range(sample_count) ]
        else:
            mask_pred = [kornia.tensor_to_image(((output_pred[0][val].to("cpu")*0.0016620444341229432+0.057950844077600344)*255).byte()) for val in range(sample_count) ]
            depth_pred = [kornia.tensor_to_image(((output_pred[1][val].to("cpu")*0.03551773442719045+0.3679109312239146)*255).byte()) for val in range(sample_count) ]
            img_shape = img['output'][0][1].shape[1]
            mask_gt = [(img['output'][0][val].reshape(img_shape,img_shape)*0.0016620444341229432+0.057950844077600344)*255 for val in range(sample_count) ]
            depth_gt = [(img['output'][1][val].reshape(img_shape,img_shape)*0.03551773442719045+0.3679109312239146)*255 for val in range(sample_count) ]




        #Image.fromarray(kornia.tensor_to_image((mask_pred*255).byte())),
        plt.figure(figsize=(10,10)) 
        plt.imshow(np.vstack([np.hstack(mask_pred),np.hstack(depth_pred)]),cmap='gray')
        plt.figure(figsize=(10,10)) 
        plt.imshow(np.vstack([np.hstack(mask_gt),np.hstack(depth_gt)]),cmap='gray')


def load_model_for_infer(model_class, model_file, device):
    #model_new = mmdmodels.UNet(n_channels=6, n_classes=2)
    checkpoint = torch.load(model_file)
    model_class.load_state_dict(checkpoint['model_state_dict'])
    return model_class.to(device)

def find_lr_type1(model, optimizer, criterion, imageloader, testloader=None, seed=1, start_lr=0.0001, end_lr=100, step_mode="exp",num_iter=500):
    torch.manual_seed(1)
    lr_finder = mmdlrfinder.LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(imageloader, val_loader=None, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter, step_mode="exp")
    min_loss = np.min(lr_finder.history['loss'])
    # min_lr = lrfinder.history['lr'][np.argmin(lr_finder.history['loss'])]
    # max_lr = np.max(lr_finder.history['lr'])
    
    # print("Min loss:{} Min LR:{} Max LR:{}".format(min_loss, min_lr, max_lr))
    # lr_finder.plot()
    # lr_finder.reset()

    return lr_finder

def train_batch(model,device,
               train_loader, 
               optimizer, 
               criterion,# = nn.MSELoss(), 
               scheduler = None,
               batch_step = False,
               stop_train=200000):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    train_mask_loss = 0
    train_depth_loss = 0
    train_accuracy = 0
    for batch_idx, dataset in enumerate(pbar):
        if(batch_idx >= stop_train):
            break
        data = dataset['input'].to(device)
        gt_mask = dataset['output'][0].to(device)
        gt_depth = dataset['output'][1].to(device)
        optimizer.zero_grad()
        pred_mask, pred_depth = torch.split(model(data), 1,dim=1)

        # get the index of the max log-probability
        #train_accuracy += pred.eq(target.view_as(pred)).sum().item()

        #loss = (torch.sqrt(criterion(gt_mask, pred_mask)) + torch.sqrt(criterion(gt_depth, pred_depth)))/2
        mask_loss = mmdlosses.construct_criterion((pred_mask, gt_mask), criterion['mask'])
        depth_loss = mmdlosses.construct_criterion((pred_depth, gt_depth), criterion['depth'])
        #mask_loss,depth_loss = criterion((pred_mask, gt_mask),(pred_depth, gt_depth))
        loss = mask_loss+depth_loss
        loss.backward()
        optimizer.step()
        train_loss += torch.mean(loss).item()
        train_mask_loss += torch.mean(mask_loss).item()
        train_depth_loss += torch.mean(depth_loss).item()
        pbar.set_description(desc= f'loss={loss.item():.6f} m_loss={mask_loss.item():.6f} d_loss={depth_loss.item():.6f} batch_id={batch_idx}')

        ### Specifically for CyclicLR. TODO: Add isinstance check for scheduler as well.
        if (batch_step == True and scheduler is not None):
            scheduler.step()
        #pbar.set_description(desc= f'loss={loss.item():0.3f} batch_id={batch_idx}')
        #train_loss += loss.item()
    pbar.close()
    dataset_len = len(train_loader)
    return train_loss/dataset_len,  train_mask_loss/dataset_len, train_depth_loss/dataset_len,

def test_batch(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.
    test_mask_loss = 0.
    test_depth_loss = 0.
    correct = 0
    pbar = tqdm(test_loader)
    #example_images = []
    with torch.no_grad():
        for batch_idx, dataset in enumerate(pbar):
            data = dataset['input'].to(device)
            gt_mask = dataset['output'][0].to(device)
            gt_depth = dataset['output'][1].to(device)
            pred_mask, pred_depth = torch.split(model(data), 1,dim=1)

            mask_loss = mmdlosses.construct_criterion((pred_mask, gt_mask), criterion['mask'])
            depth_loss = mmdlosses.construct_criterion((pred_depth, gt_depth), criterion['depth'])

            loss = mask_loss+depth_loss

            #pbar.set_description(desc= f'loss={loss} batch_id={batch_idx}')
            test_loss += torch.mean(loss).item()
            test_mask_loss += torch.mean(mask_loss).item()
            test_depth_loss += torch.mean(depth_loss).item()
            pbar.set_description(desc= f'loss={loss.item():.6f} m_loss={mask_loss.item():.6f} d_loss={depth_loss.item():.6f} batch_id={batch_idx}')
            #.item() #F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            #pred_mask, pred_depth = torch.split(model(data), 1,dim=1)
            
        #example_images.append(wandb.Image(
        #        data[0], caption="Pred: {} Truth: {}".format(classes[pred[0].item()], classes[target[0]])))

    test_loss /= len(test_loader)
    test_mask_loss /= len(test_loader)
    test_depth_loss /= len(test_loader)
    #test_accuracy = (100. * correct) / len(test_loader.dataset)
    pbar.close()
    return test_loss,test_mask_loss,test_depth_loss


def train_batch_tpu(model,device,
               train_loader, 
               optimizer, 
               criterion,# = nn.MSELoss(), 
               scheduler = None,
               batch_step = False,
               stop_train=200000):
    #print(f'[xla:{xm.get_ordinal()}] started training')
    model.train()
    pbar = tqdm(train_loader)

    tracker = xm.RateTracker()
    train_loss = 0
    train_mask_loss = 0
    train_depth_loss = 0
    train_accuracy = 0
    total_samples = 0
    for batch_idx, dataset in enumerate(pbar):
        if(batch_idx >= stop_train):
            break
        data = dataset['input']#.to(device)
        gt_mask = dataset['output'][0]#.to(device)
        gt_depth = dataset['output'][1]#.to(device)
        optimizer.zero_grad()
        pred_mask, pred_depth = torch.split(model(data), 1,dim=1)
        #total_samples += data.shape[0]
        # get the index of the max log-probability
        #train_accuracy += pred.eq(target.view_as(pred)).sum().item()

        #loss = (torch.sqrt(criterion(gt_mask, pred_mask)) + torch.sqrt(criterion(gt_depth, pred_depth)))/2
        mask_loss = mmdlosses.construct_criterion((pred_mask, gt_mask), criterion['mask'])
        depth_loss = mmdlosses.construct_criterion((pred_depth, gt_depth), criterion['depth'])
        #mask_loss,depth_loss = criterion((pred_mask, gt_mask),(pred_depth, gt_depth))
        loss = mask_loss+depth_loss
        loss.backward()
        #optimizer.step()
        xm.optimizer_step(optimizer)
        train_loss += torch.mean(loss).item()
        train_mask_loss += torch.mean(mask_loss).item()
        train_depth_loss += torch.mean(depth_loss).item()
        pbar.set_description(desc= f'[xla:{xm.get_ordinal()}]loss={loss.item():.6f} m_loss={mask_loss.item():.6f} d_loss={depth_loss.item():.6f} batch_id={batch_idx}')

        ### Specifically for CyclicLR. TODO: Add isinstance check for scheduler as well.
        if (batch_step == True and scheduler is not None):
            scheduler.step()
        #pbar.set_description(desc= f'loss={loss.item():0.3f} batch_id={batch_idx}')
        #train_loss += loss.item()
        tracker.add(FLAGS['batch_size'])
        #if batch_idx % FLAGS['log_steps'] == 0:
        #    print('[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
        #        xm.get_ordinal(), batch_idx, loss.item(), tracker.rate(),
        #        tracker.global_rate(), time.asctime()), flush=True)
    pbar.close()
    dataset_len = batch_idx#len(train_loader)
    return train_loss/dataset_len,  train_mask_loss/dataset_len, train_depth_loss/dataset_len,


def execute_model_tpu(model_class, hyperparams, 
                  train_loader, 
                  test_loader, 
                  device,
                  criterion=rmse_loss,
                  optimizer_in=optim.SGD, 
                  wandb = None,                  
                  scheduler=None,
                  prev_saved_model=None,
                  save_best=False, 
                  batch_step=False, 
                  lars_mode=False,
                  **kwargs):
    
    if wandb is None:
        hyperparams['run_name'] = fileutils.rand_run_name()
        wandb.init(config=hyperparams, project=hyperparams['project'])
    
    config = wandb.config
    model_path = fileutils.generate_model_save_path(rand_string=config.run_name)
    print("Model saved to: ",model_path)
    best_loss = 10000.0
    
    
    
    # Set random seeds and deterministic pytorch for reproducibility
    # random.seed(config.seed)       # python random seed
    torch.manual_seed(config.seed) # pytorch random seed
    # numpy.random.seed(config.seed) # numpy random seed
    #torch.backends.cudnn.deterministic = True

    # Initialize our model, recursively go over all modules and convert their parameters and buffers to CUDA tensors (if device is set to cuda)
    if(prev_saved_model != None):
        print("Not Implemented for TPUs")
    else:
        #model = model_class(config.dropout).to(device)
        model = model_class.to(device)

    if not xm.is_master_ordinal():
        xm.rendezvous('load_model')
    
    if xm.is_master_ordinal():
        xm.rendezvous('load_model')
    
    #xm.master_print(f'model_loaded, start train', flush=True)
    xm.master_print(f'model_loaded, start train')
    
    #summary(model.to(device),input_size=(3, 64, 64))
    optimizer = optimizer_in#(model.parameters(), lr=config.lr,momentum=config.momentum,
                           #weight_decay=config.weight_decay) #
    
    for epoch in range(1, config.epochs + 1):
        para_loader = pl.ParallelLoader(train_loader, [device])
        #print("Invoking training fn")
        #epoch_train_acc,epoch_train_loss = train(config, model, device, train_loader, optimizer,criterion(), epoch)
        epoch_train_loss,epoch_train_mask_loss,epoch_train_depth_loss = train_batch_tpu(model, device, 
               para_loader.per_device_loader(device), 
               optimizer, 
               criterion,# = nn.MSELoss(), 
               scheduler = None,
               batch_step = batch_step,
               stop_train=200000)
        #para_loader = pl.ParallelLoader(test_loader, [device])
        #epoch_test_loss,epoch_test_mask_loss,epoch_test_depth_loss = test_batch_tpu(model, device, test_loader, criterion)

        last_lr = 0#scheduler._last_lr
        xm.master_print('\nEpoch: {:.0f} Train set: Average loss: {:.6f}, Mask loss:{:.6f}, Depth loss:{:.6f}, lr:{} Time={}'.format(epoch, epoch_train_loss, 
                                                                epoch_train_mask_loss, epoch_train_depth_loss, last_lr, time.asctime()))
        # print('Epoch: {:.0f} Test set: Average loss: {:.6f}, Mask loss:{:.6f}, Depth loss:{:.6f}'.format(epoch, 
        #                                                         epoch_test_loss,epoch_test_mask_loss,epoch_test_depth_loss))
       
        # if(save_best == True and epoch_test_loss < best_loss):
        #     print("Model saved as Test loss reduced from ", best_loss, " to ", epoch_test_loss)
        #     torch.save({
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'test_loss':epoch_test_loss,
        #         'epoch':epoch
        #         }, model_path)
        #     best_loss = epoch_test_loss

        if (scheduler != None and 
            epoch > config.start_lr and 
            batch_step == False):
            print("Non CyclicLR Case")
            #scheduler.step(epoch_test_loss)
            scheduler.step(epoch_train_loss)
        
    print("Final model save path:",model_path," best loss:",best_loss)
    #wandb.save('model.h5')
    return model_path





def execute_model(model_class, hyperparams, 
                  train_loader, 
                  test_loader, 
                  device,
                  criterion=rmse_loss,
                  optimizer_in=optim.SGD, 
                  wandb = None,                  
                  scheduler=None,
                  prev_saved_model=None,
                  save_best=False, 
                  batch_step=False, 
                  lars_mode=False,
                  **kwargs):
    
    if wandb is None:
        hyperparams['run_name'] = fileutils.rand_run_name()
        wandb.init(config=hyperparams, project=hyperparams['project'])
    
    #wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release
    config = wandb.config
    model_path = fileutils.generate_model_save_path(rand_string=config.run_name)
    print("Model saved to: ",model_path)
    #print("Hyper Params:")
    #print(config)
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    best_loss = 10000.0
    #device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    
    # Set random seeds and deterministic pytorch for reproducibility
    # random.seed(config.seed)       # python random seed
    torch.manual_seed(config.seed) # pytorch random seed
    # numpy.random.seed(config.seed) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # Initialize our model, recursively go over all modules and convert their parameters and buffers to CUDA tensors (if device is set to cuda)
    if(prev_saved_model != None):
        # model = model_builder(model_class, 
        #                       weights_path=prev_saved_model,    
        #                       local_device=device)
        model,best_acc = model_builder2(model_class, 
                      weights_path=prev_saved_model,    
                      local_device=device)
        print("Model loaded from ", prev_saved_model, " with previous accuracy:",best_acc)
    else:
        #model = model_class(config.dropout).to(device)
        model = model_class.to(device)
    
    #summary(model.to(device),input_size=(3, 64, 64))
    optimizer = optimizer_in#(model.parameters(), lr=config.lr,momentum=config.momentum,
                           #weight_decay=config.weight_decay) #
    
    ### We will skip LR-scheduler when using LARS because of unknown interactions
    #if(lars_mode == True):
        #optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
    #optimizer = LARS(optimizer=optimizer, eps=0.6, trust_coef=0.001)
    #else:
    #scheduler = None
    #scheduler = CyclicLR(optimizer, base_lr=config.lr*0.01, max_lr=config.lr, mode='triangular', gamma=1.)#, cycle_momentum=False)#,step_size_up=1000)#, scale_fn='triangular',step_size_up=200)    

    #scheduler = StepLR(optimizer, step_size=config.sched_lr_step, gamma=config.sched_lr_gamma)
    

    #scheduler = MultiStepLR(optimizer, milestones=[10,20], gamma=config.sched_lr_gamma)
    # WandB â€“ wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    #wandb.watch(model, log="all")

    for epoch in range(1, config.epochs + 1):
        #epoch_train_acc,epoch_train_loss = train(config, model, device, train_loader, optimizer,criterion(), epoch)
        epoch_train_loss,epoch_train_mask_loss,epoch_train_depth_loss = train_batch(model, device, 
               train_loader, 
               optimizer, 
               criterion,# = nn.MSELoss(), 
               scheduler = None,
               batch_step = batch_step,
               stop_train=200000)

        epoch_test_loss,epoch_test_mask_loss,epoch_test_depth_loss = test_batch(model, device, test_loader, criterion)

        last_lr = 0#;scheduler.get_last_lr()
        print('\nEpoch: {:.0f} Train set: Average loss: {:.6f}, Mask loss:{:.6f}, Depth loss:{:.6f}, lr:{}'.format(epoch, epoch_train_loss, 
                                                                epoch_train_mask_loss, epoch_train_depth_loss, last_lr))
        print('Epoch: {:.0f} Test set: Average loss: {:.6f}, Mask loss:{:.6f}, Depth loss:{:.6f}'.format(epoch, 
                                                                epoch_test_loss,epoch_test_mask_loss,epoch_test_depth_loss))
        #myoptim = optimizer.state_dict()['param_groups'][0]
        #print('Epoch: {:.0f} Optimizer values: LR: {:.10f}, LastLR:{:.10f}, Momentum: {:.10f}, Weight Decay: {:.10f}'.format(
        #epoch, scheduler.get_lr()[0],scheduler.get_last_lr()[0],myoptim['momentum'],myoptim['weight_decay']))

        #print('Epoch: {:.0f} Optimizer values: LastLR:{:.10f}, Momentum: {:.10f}, Weight Decay: {:.10f}'.format(
        #epoch, scheduler.get_last_lr()[0],myoptim['momentum'],myoptim['weight_decay']))

        #stats_logger(global_stats_array, 1,0.1,99.0,0.1,98.0,0.001,0.78,0.00001)
        
        # wandb.log({ "Train Accuracy": epoch_train_acc, 
        #            "Train Loss": epoch_train_loss, 
        #            "Test Accuracy":epoch_test_acc, 
        #            "Test Loss": epoch_test_loss,
        #            #"Learning Rate": config.lr})
        #            "Learning Rate": last_lr})
        
        if(save_best == True and epoch_test_loss < best_loss):
            print("Model saved as Test loss reduced from ", best_loss, " to ", epoch_test_loss)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss':epoch_test_loss,
                'epoch':epoch
                }, model_path)
            best_loss = epoch_test_loss

        if (scheduler != None and 
            epoch > config.start_lr and 
            batch_step == False):
            print("Non CyclicLR Case")
            scheduler.step(epoch_test_loss)
        
    print("Final model save path:",model_path," best loss:",best_loss)
    #wandb.save('model.h5')
    return model_path

