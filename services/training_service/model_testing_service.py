import torch
import torch.nn as nn
import logging
import time
from tqdm import tqdm


class ModelTestingService:
    def __init__(self, model, testing_loader, path_builder, in_google_colab=False):
        self.model = model
        self.testing_loader = testing_loader  # DataLoader instance
        self.path_builder = path_builder
        self.in_google_colab = in_google_colab
        self.criterion = nn.CrossEntropyLoss() if config['loss'] == 'CE' else nn.BCELoss()

        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for the testing service."""
        if not logging.getLogger().hasHandlers():
            # Configure logging only if it hasn't been configured elsewhere
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def test_model(self):
        start_time = time.time()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()  # Set the model to evaluation mode

        total_correct = 0
        total_samples = 0
        test_loss = 0.0

        try:
            with torch.no_grad():  # Disable gradient calculation
                for inputs, labels in tqdm(self.testing_loader, desc="Testing Batches", unit="batch"):
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = self.model(inputs)
                    outputs, labels = self.adjust_shapes(outputs, labels)

                    loss = self.criterion(outputs, labels)
                    test_loss += loss.item() * inputs.size(0)

                    predicted = self.get_predictions(outputs)
                    total_samples += labels.size(0)
                    total_correct += (predicted == labels).sum().item()

            avg_test_loss = test_loss / total_samples
            test_accuracy = total_correct / total_samples
            self.logger.info(f'Average Test Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')
        except Exception as e:
            self.logger.exception("An error occurred during testing.")
            return self.model, None, None

        training_time = time.time() - start_time
        self.logger.info(f"Testing completed in {training_time:.2f} seconds.")
        self.model.train()  # Set the model back to training mode

        return self.model, avg_test_loss, test_accuracy

    def adjust_shapes(self, outputs, labels):
        """Adjust the labels and outputs dimensions if necessary."""
        if self.model.num_classes == 1 and outputs.dim() > 1:
            outputs = outputs.squeeze(1)
        labels = labels.view_as(outputs)  # Ensure label shapes match output shapes
        return outputs, labels

    def get_predictions(self, outputs):
        """Get predictions from the model outputs."""
        if outputs.dim() == 1:  # Binary classification
            predicted = (outputs > 0.5).float()  # Using 0.5 as the threshold
        else:  # Multi-class classification
            _, predicted = torch.max(outputs, 1)
        return predicted
