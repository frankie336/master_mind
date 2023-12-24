# services/model_service/simple_classifier_service.py
import torch
import torch.nn.functional as F
from services.model_service.base_nn_service import BaseNN
import torch.nn as nn


class SimpleBinaryClassifier(BaseNN):
    """
    A simple binary classification neural network, extending the BaseNN class.

    This network consists of two fully connected layers with a ReLU activation function
    after the first layer and a sigmoid activation function after the second layer,
    making it suitable for binary classification tasks.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer, outputting the class probability.
    """

    def __init__(self, input_size, num_classes=1):
        """
        Initializes the SimpleBinaryClassifier with input size and number of classes.

        Args:
            input_size (int): The size of the input layer.
            num_classes (int): The number of output classes. Default is 1 for binary classification.
        """
        super(SimpleBinaryClassifier, self).__init__(input_size, num_classes)
        self.fc1 = nn.Linear(self.input_size, 10)
        self.fc2 = nn.Linear(10, self.num_classes)

    def forward(self, x):
        """
        Implements the forward pass of the network.

        Args:
            x (torch.Tensor): The input data tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Use sigmoid for binary classification
        return x

    def get_hyperparameters_str(self):
        return f"Input Size: {self.input_size}, Num Classes: {self.num_classes}, Layers: [fc1: {self.fc1}, fc2: {self.fc2}]"
