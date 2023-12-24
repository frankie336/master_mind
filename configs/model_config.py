config = {
    'epochs': 10,  # Number of epochs to train the model
    'learning_rate': 0.001,  # Learning rate for the optimizer
    'batch_size': 32,  # Batch size for the training data loader
    'checkpoint_interval': 5,  # Interval at which to save model checkpoints
    'loss': 'BCE',  # Loss function type ('BCE' for binary, 'CE' for CrossEntropy)
    'optimizer': 'Adam',  # Type of optimizer to use
    'optimizer_params': {  # Additional parameters for the optimizer, if any
        'weight_decay': 1e-4,
        # Other optimizer-specific parameters can be added here
    },
    'scheduler': 'ReduceLROnPlateau',  # Learning rate scheduler type
    'scheduler_params': {  # Parameters for the learning rate scheduler
        'mode': 'min',
        'factor': 0.1,
        'patience': 10,
        'verbose': True,
        # Other scheduler-specific parameters can be added here
    },
    'early_stopping_patience': 10,  # Patience for early stopping
    'save_best_only': True,  # Flag to save only the best model checkpoint
    'model_save_path': 'models/',  # Directory to save model checkpoints
    # Google Colab specific configurations
    'in_google_colab': False,  # Flag to indicate if running in Google Colab
    'google_drive_model_path': '/content/drive/My Drive/Colab Notebooks/models/',  # Path to save models in Google Drive
    'google_drive_reg_path': '/content/drive/My Drive/Colab Notebooks/regularization/',  # Path for saving regularization files
    # Add any additional configurations that might be needed
}
