# configs/model_config.py

model_config = {
    "epochs": 10,  # Number of training epochs.
    "learning_rate": 0.001,  # Learning rate for the optimizer.
    "checkpoint_interval": 5,  # Interval for saving model checkpoints.
    "loss_function": "CrossEntropyLoss",  # Loss function. Change as per model needs.
    # Add additional model-specific parameters as needed.
}

