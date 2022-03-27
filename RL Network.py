# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:56:41 2022

@author: Student
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class DeepRLNetwork(nn.Module):
    def _init_(self):
        super(DeepRLNetwork, self).__init__()
    
        # Input is of size (3, 210, 160)
        # Greyscale conversion
        
        # down-sampling
        
        # cropping
        
        # Images should now be (84x84x4)
        # convolutional layers
        self.conv1 = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)