"""
     All model class definitions for MNIST
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models


class MNISTDigitBuilder(nn.Module):
    def __init__(self, dropout=0.1):
        super(MNISTDigitBuilder, self).__init__()
        self.dropout_val = dropout
        self.bias = False
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(8, 8, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # nn.Conv2d(8, 8, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(8),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1,stride=1, bias=self.bias),            
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(8, 16, 3, padding=1, bias=self.bias),            
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val),
            # nn.Conv2d(16, 16, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            #nn.Dropout(self.dropout_val)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3,bias=self.bias),            
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(16, 16, 3,bias=self.bias),            
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val)
        )
        
        self.gap_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(16, 10, 1, bias=self.bias)
        )
                
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        #x = x.view(x.size(0), -1)
        x = self.gap_linear(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)
        return x

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class BareCIFAR(nn.Module):
    def __init__(self):
        super(BareCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CIFARModelDepthDilate(nn.Module):
    def __init__(self,dropout=0.1):
        super(CIFARModelDepthDilate, self).__init__()
        self.layer1_channels = 32
        self.dropout_val = dropout
        self.bias = False

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.layer1_channels, 3, padding=1, stride=1,bias=self.bias),
            #nn.Conv2d(3,self.layer1_channels,1,1,0,1,1,bias=bias),
            nn.BatchNorm2d(self.layer1_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels, self.layer1_channels, 3, padding=1, stride=1,bias=self.bias,groups=self.layer1_channels),
            nn.Conv2d(self.layer1_channels,self.layer1_channels,1,1,0,1,1,bias=self.bias),      
            nn.BatchNorm2d(self.layer1_channels),
            nn.ReLU(),
            # nn.Conv2d(self.layer1_channels, self.layer1_channels, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.layer1_channels, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias), #groups=self.layer1_channels),
            #nn.Conv2d(self.layer1_channels,self.layer1_channels*2,1,1,0,1,1,bias=self.bias),
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias, groups=self.layer1_channels*2),
            nn.Conv2d(self.layer1_channels*2,self.layer1_channels*2,1,1,0,1,1,bias=self.bias),      
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*2),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*4, 3, padding=2, stride=1,bias=self.bias,dilation=2), #groups=self.layer1_channels*2),
            #nn.Conv2d(self.layer1_channels*2,self.layer1_channels*4,1,1,0,1,1,bias=self.bias),       
            nn.BatchNorm2d(self.layer1_channels*4),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=2, stride=1,bias=self.bias, groups=self.layer1_channels*4, dilation=2),
            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4,1,1,0,1,1,bias=self.bias),
            nn.BatchNorm2d(self.layer1_channels*4),
            nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*4),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))

        self.conv4 = nn.Sequential(
            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*8, 3, padding=1, stride=1,bias=self.bias),#, groups=self.layer1_channels),            
            nn.BatchNorm2d(self.layer1_channels*8),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels*8, self.layer1_channels*8, 3, padding=1, stride=1,bias=self.bias, groups=self.layer1_channels),
            nn.Conv2d(self.layer1_channels*8, self.layer1_channels*8,1,1,0,1,1,bias=self.bias),
            nn.BatchNorm2d(self.layer1_channels*8),
            nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*8),
            # nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))
        
        self.gap_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(self.layer1_channels*8, 10, 1, bias=self.bias)
        )

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #x = x.view(x.size(0), -1)
        x = self.gap_linear(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)

        return x

class CIFARModel(nn.Module):
    def __init__(self):
        super(CIFARModel, self).__init__()
        self.layer1_channels = 32
        self.dropout_val = 0.1
        self.bias = False

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.layer1_channels, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(self.layer1_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels, self.layer1_channels, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(self.layer1_channels),
            nn.ReLU(),
            # nn.Conv2d(self.layer1_channels, self.layer1_channels, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.layer1_channels, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(self.layer1_channels*2),
            nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*2, self.layer1_channels*2, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*2),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.layer1_channels*2, self.layer1_channels*4, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(self.layer1_channels*4),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(self.layer1_channels*4),
            nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*4),
            # nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))

        self.conv4 = nn.Sequential(
            nn.Conv2d(self.layer1_channels*4, self.layer1_channels*8, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(self.layer1_channels*8),
            nn.ReLU(),
            nn.Dropout(self.dropout_val),
            nn.Conv2d(self.layer1_channels*8, self.layer1_channels*8, 3, padding=1, stride=1,bias=self.bias),            
            nn.BatchNorm2d(self.layer1_channels*8),
            nn.ReLU(),
            # nn.Conv2d(self.layer1_channels*4, self.layer1_channels*4, 3, padding=1, bias=self.bias),            
            # nn.BatchNorm2d(self.layer1_channels*8),
            # nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            nn.Dropout(self.dropout_val))
        
        self.gap_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(self.layer1_channels*8, 10, 1, bias=self.bias)
        )

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        #x = x.view(x.size(0), -1)
        x = self.gap_linear(x)
        x = x.view(-1, 10)
        x = F.log_softmax(x, dim=1)
        return x

'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F






class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,dropout=0.0 ):
        super(BasicBlock, self).__init__()
        self.dropout_val=dropout
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                #nn.Dropout(self.dropout_val)
            )
        self.dropout_layer = nn.Dropout(self.dropout_val)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))#self.dropout_layer(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))#self.dropout_layer(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MonoMaskDepthResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(MonoMaskDepthResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3*2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.convFinal = nn.Conv2d(512*block.expansion, 128, kernel_size=1, bias=False)
        self.convFinalBn = nn.BatchNorm2d(128)

        #self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.convFinalBn(self.convFinal(out)))
        #out = F.avg_pool2d(out, 4)
        #out = out.view(out.size(0), -1)
        out = out.view(out.size(0), 2, out.size(2)*8, out.size(3)*8)
        #out = self.linear(out)
        return out





class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,dropout=0.0,scale_input=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.dropout = dropout
        self.scale_input = scale_input

        self.conv1 = nn.Conv2d(3*self.scale_input, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,dropout=self.dropout))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(dropout=0.0, num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], dropout=dropout, num_classes=num_classes)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

def MonoMaskDepthResNetFunc():
    #return MonoMaskDepthResNet(Bottleneck, [3,4,6,3])
    return MonoMaskDepthResNet(Bottleneck, [3,3,3,3])

# def MonoMaskDepthResNet18():
#     #return MonoMaskDepthResNet(Bottleneck, [3,4,6,3])
#     return ResNet18


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

"""
    ModifiedResBlock: Class for creating Modified ResNet block. Based on S11:
        X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU
        R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) 
        Add(X, R1)

"""

class ModifiedResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ModifiedResBlock, self).__init__()
        self.layerconv = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
        )
        ### This layer applies after the first conv and we intend to keep the channel size same
        self.resconv = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
            )
        #self.shortcut = nn.Sequential() 

    def forward(self, x):
        out = self.layerconv(x)
        res = self.resconv(out)
        #out = res
        #out = F.relu(out)
        return out+res

"""
    S11: Custom resnet block based model
    It used the ModifiedResBlock which doesnt have multiple layers.
    PrepLayer:
        Conv 3x3 s1, p1) >> BN >> RELU [64]
    Layer1:
        ModifiedResBlock(128)
    Layer 2:
        Conv 3x3 [256]
        MaxPooling2D
        BN
        ReLU
    Layer 3:
        ModifiedResBlock(512)
    MaxPooling:(with Kernel Size 4) 
    FC Layer 
    SoftMax
"""

class S11ResNet(nn.Module):
    def __init__(self, num_classes=10,dropout=0.0):
        super(S11ResNet, self).__init__()
        self.in_planes = 64
        self.resize_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.prep_layer = nn.Sequential(
            nn.Conv2d(64, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU()
        )
        self.layer1 = ModifiedResBlock(self.in_planes, self.in_planes*2, 1)
            
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.in_planes*2, self.in_planes*4, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=False),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(self.in_planes*4),
            nn.ReLU()
        )
        self.layer3 = ModifiedResBlock(self.in_planes*4, self.in_planes*8, 1)
        self.layer4_bigmax = nn.MaxPool2d(4,4)
        #self.fc_layer = nn.Linear(512, 10)
        self.fc_layer = nn.Linear(512, 200)

    def forward(self, x):
        out = self.resize_layer(x)
        out = self.prep_layer(out)
        #out = self.prep_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4_bigmax(out)
        out = out.view(-1, 512)
        #out = out.view(-1, 10)
        out = self.fc_layer(out)
        #
        out = F.log_softmax(out, dim=1)
        return out


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())

#        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),            
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(6, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class MobileNetV2Mod(nn.Module):
    def __init__(self, base_model, n_class, dropout_val=0.2):
        super().__init__()
        #self.base_model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        #self.base_layers = list(self.base_model.children())[:-1]
        self.core_layers = nn.Sequential(*list(base_model.children())[:-1])
        self.classes = n_class
        self.dropout_val = dropout_val
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
                                         nn.Linear(in_features=1280, out_features=n_class))
    def forward(self, x):
        x = self.core_layers(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)



        #x = x.view(x.size(0), -1)
        #print(x.size(0))
        #x = self.final_layer(x)
        #x = x.view(-1, self.classes)
        x = F.log_softmax(x, dim=1)
        return x

class MobileNetV2New(nn.Module):
    def freeze_layer(self, my_layer):
        for param in my_layer:
            #print("Setting False")
            param.requires_grad = False

    def get_core_layer_count(self):
        return len(self.core_layer_params)

    def get_core_layer_params(self):
        return sum(self.core_layer_params,[])

    def unfreeze_core_layer(self, n_lower_layer):
        if(n_lower_layer > self.get_core_layer_count()):
            print("Request to unfreeze {} layers, core layer has {} layers".format(n_lower_layer, self.get_core_layer_count))
            return None
        param_list = sum(self.core_layer_params[-n_lower_layer:],[])
        for param in param_list:
            param.requires_grad = True
        return sum(self.core_layer_params[-n_lower_layer:],[param for param in self.classifier.parameters()])

    def __init__(self, base_model, n_class, dropout_val=0.2, base_freeze=True):
        super().__init__()
        #self.base_model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        #self.base_layers = list(self.base_model.children())[:-1]
        self.core_layer = nn.Sequential(*list(base_model.features))
        
        if(base_freeze == True):
            self.freeze_layer(self.core_layer.parameters())
        self.core_layer_params = [[param[1] for param in child.named_parameters()] for child in self.core_layer.children()]
        self.classes = n_class
        self.dropout_val = dropout_val
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_features=1280, out_features=n_class)
       
        
    def forward(self, x):
        x = self.core_layer(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        #x = x.view(x.size(0), -1)
        #print(x.size(0))
        #x = self.final_layer(x)
        #x = x.view(-1, self.classes)
        x = F.log_softmax(x, dim=1)
        return x


class MobileNetV24C(nn.Module):
    def freeze_layer(self, my_layer):
        for param in my_layer:
            #print("Setting False")
            param.requires_grad = False

    def get_core_layer_count(self):
        return len(self.core_layer_params)

    def get_core_layer_params(self):
        return sum(self.core_layer_params,[])

    def unfreeze_core_layer(self, n_lower_layer):
        if(n_lower_layer > self.get_core_layer_count()):
            print("Request to unfreeze {} layers, core layer has {} layers".format(n_lower_layer, self.get_core_layer_count))
            return None
        param_list = sum(self.core_layer_params[-n_lower_layer:],[])
        for param in param_list:
            param.requires_grad = True
        return sum(self.core_layer_params[-n_lower_layer:],[param for param in self.classifier.parameters()])

    def __init__(self, base_model, n_class, dropout_val=0.2, base_freeze=True):
        super().__init__()
        #self.base_model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        #self.base_layers = list(self.base_model.children())[:-1]
        self.core_layer = nn.Sequential(*list(base_model.features))
        
        if(base_freeze == True):
            self.freeze_layer(self.core_layer.parameters())
        self.core_layer_params = [[param[1] for param in child.named_parameters()] for child in self.core_layer.children()]
        self.classes = n_class
        self.dropout_val = dropout_val
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Linear(in_features=1280, out_features=1000),
                                         nn.Linear(in_features=1000, out_features=n_class))
       
        
    def forward(self, x):
        x = self.core_layer(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        #x = x.view(x.size(0), -1)
        #print(x.size(0))
        #x = self.final_layer(x)
        #x = x.view(-1, self.classes)
        x = F.log_softmax(x, dim=1)
        return x


# test()
