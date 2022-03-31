# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:56:41 2022

@author: Jasper Havenhand
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random

BATCH_SIZE = 20
MEMORY_SIZE = 10000
memory = deque([],maxlen=MEMORY_SIZE)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DeepQLNetwork(nn.Module):
    
    def _init_(self,inputH,inputW,outputDim):
        super(DeepQLNetwork, self).__init__()
    
        # Input is of size (3, 210, 160)
        # Greyscale conversion
        
        # down-sampling
        
        # cropping
        
        # Images should now be (84x84x4)
        # convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        def conv2d_size_out(size, kernel_size = 3, stride = 1, padding = 1):
            return (size - (kernel_size + 2*padding)) // stride  + 1
        convW = conv2d_size_out(inputW)
        convH = conv2d_size_out(inputH)
        self.lin1 = nn.Linear(convW*convH*32,256)
        self.lin2 = nn.Linear(256,outputDim)
        
    def forward(self,input):
        output = F.conv1(input)
        output = F.conv2(output)
        output = F.lin1(output)
        output = F.lin2(output)
        return output
    
    def memorySample():
        return random.sample(memory, BATCH_SIZE)
    
    def memoryAdd(transition):
        memory.append(transition)
    
    