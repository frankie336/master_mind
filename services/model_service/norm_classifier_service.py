# services/model_service/norm_classifier_service.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from services.model_service.base_nn_service import BaseNN
import torch.nn as nn


class BatchNormOnlyDeepNN(BaseNN):
    """
    A deep neural network for binary classification with batch normalization layers.

    This network extends the BaseNN class and includes multiple fully connected layers
    with batch normalization applied after each layer except the output layer. It uses ReLU
    activation functions in hidden layers and a sigmoid activation function in the output layer,
    making it suitable for binary classification tasks.

    Attributes:
        fc1 to fc4 (nn.Linear): Fully connected layers with varying output sizes.
        bn1 to bn3 (nn.BatchNorm1d): Batch normalization layers corresponding to fc1 to fc3.
    """

    def __init__(self, input_size, num_classes=1):
        """
        Initializes the BatchNormOnlyDeepNN with input size and number of classes.

        Args:
            input_size (int): The size of the input layer.
            num_classes (int): The number of output classes. Default is 1 for binary classification.
        """
        super(BatchNormOnlyDeepNN, self).__init__(input_size, num_classes)
        self.fc1 = nn.Linear(self.input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, self.num_classes)  # Output layer for binary classification

    def forward(self, x):
        """
        Implements the forward pass of the network.

        Args:
            x (torch.Tensor): The input data tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))  # Use sigmoid for binary classification
        return x

    def get_hyperparameters_str(self):
        """
        Retrieves a string representation of the model's hyperparameters.

        Returns:
            str: A string containing hyperparameter information of the model.
        """
        return (f"Input Size: {self.input_size}, Num Classes: {self.num_classes}, "
                f"Layers: [fc1: {self.fc1}, bn1: {self.bn1}, "
                f"fc2: {self.fc2}, bn2: {self.bn2}, "
                f"fc3: {self.fc3}, bn3: {self.bn3}, "
                f"fc4: {self.fc4}]")
