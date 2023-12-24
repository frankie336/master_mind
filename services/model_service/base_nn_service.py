# Filename: base_nn.py
import torch
import torch.nn as nn


class BaseNN(nn.Module):
    """
    A base neural network class for creating various neural network architectures.

    This class is intended to be subclassed to create specific types of neural networks.
    It defines a basic structure and leaves the implementation of the forward pass and
    hyperparameter retrieval to be defined in subclasses.

    Attributes:
        input_size (int): The size of the input layer.
        num_classes (int): The number of output classes for classification tasks.
    """

    def __init__(self, input_size, num_classes=1):
        """
        Initializes the BaseNN with input size and number of classes.

        Args:
            input_size (int): The size of the input layer.
            num_classes (int): The number of output classes. Default is 1 for binary classification.
        """
        super(BaseNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

    def forward(self, x):
        """
        Defines the forward pass of the network. This method should be implemented by subclasses.

        Args:
            x (torch.Tensor): The input data tensor.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_hyperparameters_str(self):
        """
        Retrieves a string representation of the model's hyperparameters. This method should be implemented by subclasses.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method.")
