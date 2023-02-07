from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class MyModel(nn.Module):
    """
    A class for a NN to train and make predictions on MNIST data.

    ...

    Attributes
    ----------
    name : str
        first name of the person
    surname : str
        family name of the person
    age : int
        age of the person

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """

    def __init__(self):
        """Constructs all the layers of the NN."""

        super().__init__()
        self.in_fc = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out_fc = nn.Linear(64, 10)

        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Does a forward pass on the NN

        Parameters
        ----------
        x : torch.Tensor
            batch of data to perform a forward pass upon

        Returns
        -------
        features, x : torch.Tensor, torch.Tensor
            the 2d features before the output layer & the output of the NN
        """
        # self.pshape(x)
        x = F.relu(self.in_fc(x))
        x = self.dropout(x)
        # self.pshape(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # self.pshape(x)
        features = F.relu(self.fc3(x))
        x = self.dropout(features)
        # self.pshape(x)
        x = F.log_softmax(self.out_fc(x), dim=0)
        # self.pshape(x)

        return features, x

    def pshape(self, x: torch.Tensor) -> None:
        """
        Prints the shape of the input tensor

        Parameters
        ----------
        x : torch.Tensor
            input tensor
        """
        print("Shape: ", x.shape)
