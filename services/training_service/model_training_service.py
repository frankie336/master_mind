# services/training_service/model_training_service.py
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import logging
from datetime import datetime
from tqdm import tqdm


class ModelTrainingService:
    def __init__(self, model, train_loader, model_config, config,
                 model_version_service, google_model_path=False,
                 google_reg_path=False,in_google_colab=False):

        self.model = model
        self.train_loader = train_loader
        self.model_config = model_config
        self.config = config
        self.model_version_service = model_version_service
        self.in_google_colab = in_google_colab
        self.google_model_path = google_model_path
        self.config = google_reg_path

        # Model loss function
        self.criterion = nn.BCELoss() if self.model_config.get('loss',
                                                               'default_loss') == 'BCE' else nn.CrossEntropyLoss()

        self.logger = logging.getLogger(__name__)

        # Optimizer setup
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.model_config.get('learning_rate', 0.001)
        )


    def train_model(self):
        start_time = datetime.now()

        # Retrieve model training parameters from model_config
        epochs = self.model_config.get('epochs', 2)
        checkpoint_interval = self.model_config.get('checkpoint_interval', 5)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.logger.info(f"Model is on: {device}")

        scheduler = ReduceLROnPlateau(self.optimizer, **self.model_config.get('scheduler_params', {}))

        best_loss = float('inf')
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            self.model.train()  # Set model to training mode
            epoch_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                outputs, labels = self.adjust_shapes(outputs, labels)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)

            avg_epoch_loss = epoch_loss / len(self.train_loader.dataset)
            scheduler.step(avg_epoch_loss)
            self.logger.info(f"Epoch {epoch + 1}, Average Loss: {avg_epoch_loss}")

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                if epoch % checkpoint_interval == 0:
                    self.save_checkpoint(epoch, best_loss)

            # Early stopping condition based on scheduler's bad epochs count
            if getattr(scheduler, 'num_bad_epochs', 0) == self.model_config.get('early_stopping_patience', 10):
                self.logger.info("Early stopping triggered.")
                break

        training_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Training completed in {training_time:.2f} seconds.")

        return self.model, best_loss, len(self.train_loader.dataset), training_time

    def adjust_shapes(self, outputs, labels):
        """Adjust the labels and outputs dimensions if necessary."""
        if self.model.num_classes == 1 and outputs.dim() > 1:
            outputs = outputs.squeeze(1)
        labels = labels.view_as(outputs)  # Ensure label shapes match output shapes
        return outputs, labels

    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint."""
        try:

            model_save_path = self.config.get('model_save_path', './models/')

            os.makedirs(model_save_path, exist_ok=True)  # Ensure directory exists

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_filename = f"checkpoint_epoch_{epoch}_loss_{loss:.4f}_{timestamp}.pth"
            path = os.path.join(model_save_path, checkpoint_filename)

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            }, path)
            self.logger.info(f"Checkpoint saved: {path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint at {path}: {str(e)}")
