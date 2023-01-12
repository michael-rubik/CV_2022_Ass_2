# TUWIEN - WS2022 CV: Task4 - Mask Classification using CNN
# *********+++++++++*******++++INSERT GROUP NO. HERE
from typing import List
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F

# A simple CNN for mask classification


class MaskClassifier(nn.Module):

    def __init__(self, name, img_size=64, dropout: float = 0, batch_norm: bool = False):
        """
        Initializes the network architecture, creates a simple cnn of convolutional and max pooling
        layers. If batch_norm is set to true, batchnorm is applied to the convolutional layers.
        If dropout>0, dropout is applied to the linear layers.
        HINT: nn.Conv2d(...), nn.MaxPool2d(...), nn.BatchNorm2d(...), nn.Linear(...), nn.Dropout(...)
        dropout: dropout rate between 0 and 1
        batch_norm: if batch normalization should be applied
        """

        # student code start
        super().__init__()

        self.name = name
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.linear = nn.Linear(in_features=6272, out_features=1)

        # student code end

    def forward(self, x: Tensor):
        """
        Applies the predefined layers of the network to x
        x: input tensor to be classified [batch_size x channels x height x width] - Tensor
        """

        # student code start
        
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = self.maxpool2(x)     
        x = torch.nn.functional.relu(x)  
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        print(min(x))
        print(max(x))

        # student code end

        return x
