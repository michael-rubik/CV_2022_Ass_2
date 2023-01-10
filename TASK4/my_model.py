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
        pass
        # student code end

    def forward(self, x: Tensor):
        """
        Applies the predefined layers of the network to x
        x: input tensor to be classified [batch_size x channels x height x width] - Tensor
        """

        # student code start
        raise NotImplementedError("TO DO in my_model.py")
        # student code end

        return x
