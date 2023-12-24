import time
import torch
from configs.config import ForexMastermindConfig
from services.common.tools import FFDataLoader, ModelVersionService
from services.model_service.simple_classifier_service import SimpleBinaryClassifier
from services.training_service.data_preprocessor_service import DataPreprocessor
from services.training_service.model_training_service import ModelTrainingService
from services.training_service.model_testing_service import ModelTestingService
from services.training_service.model_evaluation_service import ModelEvaluationService


class ModelOrchestrator:
    def __init__(self, sample_size, batch_size, config, model_config=None, file_path=None, version=None,
                 in_google_colab=False, google_model_path=None, google_reg_path=None,
                 model_version_service=None):
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.file_path = file_path
        self.version = version
        self.in_google_colab = in_google_colab
        self.google_model_path = google_model_path
        self.google_reg_path = google_reg_path

        # Configuration settings
        self.config = config if config is not None else ForexMastermindConfig()
        self.model_config = model_config  # Separate configuration for model-related settings
        self.model_version_service = model_version_service if model_version_service is not None else ModelVersionService(
            config=self.config, in_google_colab=in_google_colab)

        self.data_preprocessor = DataPreprocessor(config=self.config, batch_size=self.batch_size, in_google_colab=in_google_colab)

    def run(self, epochs=2, learning_rate=0.001, checkpoint_interval=5):
        # Assuming FFDataLoader and other services are appropriately defined
        ff_data_loader = FFDataLoader()
        df = ff_data_loader.load_data(self.file_path)

        # Data loading, sampling, and preprocessing
        sampled_df = df.sample(frac=self.sample_size) if self.sample_size < 1 else df
        train_loader, val_loader, test_loader, features = self.data_preprocessor.preprocess_data(
            unprocessed_training_data_df=sampled_df, model_version=self.model_version_service.get_current_model_version())

        # Model initialization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleBinaryClassifier(input_size=len(features)).to(device)

        # Execute Training Service with model_config
        model_training_service = ModelTrainingService(model=model, train_loader=train_loader, model_config=self.model_config,
                                                      config=self.config, model_version_service=self.model_version_service,
                                                      google_model_path=self.google_model_path, google_reg_path=self.google_reg_path)
        trained_model, best_loss, total_samples_trained, training_time = model_training_service.train_model()

        # Execute Testing Service
        model_testing_service = ModelTestingService(model=trained_model, batch_size=self.batch_size,
                                                    testing_loader=test_loader, config=self.config, in_google_colab=self.in_google_colab)
        model, avg_test_loss, test_accuracy = model_testing_service.test_model()

        # Execute Evaluation Service
        model_evaluation_service = ModelEvaluationService(model=trained_model, batch_size=self.batch_size,
                                                          get_hyperparameters_str=trained_model.get_hyperparameters_str(), training_time=training_time,
                                                          eval_loader=val_loader, features=features, config=self.config,
                                                          in_google_colab=self.in_google_colab, google_model_path=self.google_model_path,
                                                          google_reg_path=self.google_reg_path)
        model_evaluation_service.evaluate(test_avg_test_loss=avg_test_loss, test_accuracy=test_accuracy)

# Usage Example (Uncomment to test)
if __name__ == '__main__':
     config = ForexMastermindConfig()
     model_config = {"epochs": 10, "learning_rate": 0.001, "checkpoint_interval": 5}
     file_path = config.get_next_data_path(data_type='ForexData', training_data_version='20')
     orchestrator = ModelOrchestrator(sample_size=0.01, batch_size=32, config=config, model_config=model_config, file_path=file_path,
                                      in_google_colab=False, google_model_path=None, google_reg_path=None)
     #orchestrator.run()
