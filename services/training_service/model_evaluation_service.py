import torch
import torch.nn as nn
import logging
import time
import os
import csv
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import shap  # Assuming SHAP is installed and available

from services.common.tools import DataHashService


class ModelEvaluationService:

    def __init__(self, model,  batch_size, get_hyperparameters_str, training_time, eval_loader, features, config,
                 model_version_service=None, in_google_colab=False, google_model_path=None, google_reg_path=None):

        self.model = model
        self.batch_size = batch_size
        self.get_hyperparameters_str = get_hyperparameters_str
        self.training_time = training_time
        self.eval_loader = eval_loader
        self.features = features
        #self.total_samples_trained = total_samples_trained
        self.config = config
        self.model_version_service = model_version_service
        self.in_google_colab = in_google_colab
        self.google_model_path = google_model_path
        self.google_reg_path = google_reg_path
        self.y_true = None
        self.y_pred =None

    def collect_metadata(self, test_avg_test_loss, test_accuracy,
                         eval_accuracy, classification_report_str, data_hash):

        metadata = {
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_version": self.model_version_service.get_next_model_version(),
            "data_hash": data_hash,
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "total_samples_trained": 'TBC',
            "model_type": type(self.model).__name__,
            "training_time": self.training_time,
            "accuracy_in_testing": test_accuracy,
            "accuracy_in_evaluation": eval_accuracy,
            "notes": 'TBC',
            "classification_report": classification_report_str,
            "model_hyperparameters": self.get_hyperparameters_str,
            "feature_list": self.features,
            "trained_with": 'v11',
        }

        return metadata

    def evaluate(self, test_avg_test_loss, test_accuracy):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()  # Set the model to evaluation mode

        y_pred, y_true, y_scores, X_batches = [], [], [], []
        total_correct = 0
        total_samples = 0

        data_hash = DataHashService.generate_data_hash(self.model)

        with torch.no_grad():
            for inputs, labels in self.eval_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)

                # For binary classification with single output neuron
                predicted = (outputs > 0.5).float()  # Using 0.5 as threshold
                y_pred.extend(predicted.cpu().numpy().flatten())
                y_true.extend(labels.cpu().numpy().flatten())
                y_scores.extend(outputs.cpu().numpy().flatten())

                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                X_batches.append(inputs.cpu().numpy())

        eval_accuracy = total_correct / total_samples
        print(f"Eval Accuracy: {eval_accuracy:.2f}%")

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        X_test = np.concatenate(X_batches, axis=0)

        # Adjust target_names as per your class labels
        target_names = ['Class 0', 'Class 1']
        report = classification_report(y_true, y_pred, target_names=target_names)
        print("Classification Report:\n", report)

        # Generate evaluation plots and metadata
        self._generate_evaluation_plots(y_true, y_pred, y_scores, X_test)

        metadata = self.collect_metadata(test_avg_test_loss=test_avg_test_loss,
                                         test_accuracy=test_accuracy, eval_accuracy=eval_accuracy,
                                         classification_report_str=report, data_hash=data_hash)

        # Conditional model saving based on evaluation accuracy threshold
        accuracy = metadata['accuracy_in_evaluation']
        if eval_accuracy >= 0.60:
            self._save_model(accuracy=accuracy)

        self._write_metadata_to_registry(metadata=metadata)

    def _save_model(self, accuracy):

        model_version = self.model_version_service.get_next_model_version()
        model_directory = self.config.get_model_directory(in_google_colab=self.in_google_colab)
        model_filename = f"IntraDayForexPredictor_v{model_version}.pt"
        model_path = os.path.join(model_directory, model_filename)
        torch.save(self.model.state_dict(), model_path)
        print(f"Deep NN model saved as {model_path}")

    def _write_metadata_to_registry(self, metadata):

        model_registry_path = os.path.join(
            self.config.get_model_directory(in_google_colab=self.in_google_colab), 'nn_model_registry_.csv')

        # Check if the registry file exists, if not, create it with headers
        if not os.path.isfile(model_registry_path):
            with open(model_registry_path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=metadata.keys())
                writer.writeheader()

        # Append the metadata to the registry file
        with open(model_registry_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=metadata.keys())
            writer.writerow(metadata)

        print(f"Model metadata saved for model_version {metadata['model_version']}")

    def _generate_evaluation_plots(self, y_true, y_pred, y_scores, X_test):
        """
        Generate and display evaluation plots.
        """
        fig, axs = plt.subplots(3, 2, figsize=(20, 18))

        # Plots
        self._plot_confusion_matrix(y_true, y_pred, axs[0, 0])
        self._plot_roc_curve(y_true, y_scores, axs[0, 1])
        self._plot_precision_recall_curve(y_true, y_scores, axs[1, 0])
        self._plot_lift_curve(y_true, y_scores, axs[1, 1])
        self._plot_correlation_heatmap(axs[2, 0], X_test)
        #self._plot_feature_interactions(axs[2, 1], X_test, [0, 1])  # Replace [0, 1] with actual feature indices
        #self._plot_feature_interactions(axs[2, 1], self.X_test, [0, 1])
        plt.tight_layout()
        self._save_evaluation_plots()
        plt.show()

    @staticmethod
    def _plot_confusion_matrix(y_true, y_pred, ax):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Confusion Matrix")

    @staticmethod
    def _plot_roc_curve(y_true, y_score, ax):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')

    @staticmethod
    def _plot_precision_recall_curve(y_true, y_score, ax):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        average_precision = average_precision_score(y_true, y_score)
        ax.step(recall, precision, color='blue', where='post', label=f'AP={average_precision:.2f}')
        ax.fill_between(recall, precision, step='post', alpha=0.2, color='blue')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])

    @staticmethod
    def _plot_lift_curve(y_true, y_scores, ax):
        # Ensure y_true and y_scores are numpy arrays
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        # Sort scores and corresponding truth values
        indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[indices]

        # Calculate cumulative gains
        cumulated_gain = np.cumsum(y_true_sorted)
        cumulated_gain_percentage = cumulated_gain / cumulated_gain[-1]

        # Calculate lift
        lift = cumulated_gain_percentage / (np.arange(len(y_true)) + 1) / np.mean(y_true)

        # Plot the lift curve
        ax.plot(lift, label='Lift Curve')
        ax.plot([0, len(lift)], [1, 1], 'k--', label='Random')
        ax.set_xlabel('Number of Cases')
        ax.set_ylabel('Lift')
        ax.set_title('Lift Curve')
        ax.legend(loc='upper right')

    @staticmethod
    def _plot_correlation_heatmap(ax, X, threshold=0.8):
        # Convert NumPy array to DataFrame
        X_df = pd.DataFrame(X)

        # Compute the correlation matrix
        corr_matrix = X_df.corr()

        # Optionally, filter the correlation matrix
        if threshold is not None:
            corr_matrix = corr_matrix[abs(corr_matrix) > threshold]

        # Plot the heatmap
        sns.heatmap(corr_matrix, ax=ax, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
        ax.set_title('Correlation Heatmap')
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='y', rotation=0)

    @staticmethod
    def _plot_shap_summary(self, X):
        explainer = shap.KernelExplainer(self.model.predict, X)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, show=False)

        plot_filename = f"IntraDayForexPredictor_v{self.config.get_current_model_version()}_shap.png"
        plot_path = os.path.join(self.config.get_plot_directory(), plot_filename)
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    @staticmethod
    def _plot_feature_interactions(self, ax, X, features):
        top_features = [X.columns[i] for i in features]
        ax.scatter(X[top_features[0]], X[top_features[1]], c=y_true)
        ax.set_xlabel(str(top_features[0]))
        ax.set_ylabel(str(top_features[1]))
        ax.set_title('Feature Interactions')
    @staticmethod
    def get_shap_values(self, X):
        # Create a KernelExplainer

        self.model.eval()
        explainer = shap.KernelExplainer(self.model, shap.sample(X, 100))
        shap_values = explainer.shap_values(X, nsamples=100)

        return shap_values[1] if isinstance(shap_values, list) else shap_values

    def _save_evaluation_plots(self):
        """
        Save the evaluation plots to a specified directory.
        """
        plot_directory = self.config.get_plot_directory(in_google_colab=self.in_google_colab)
        model_version = self.model_version_service.get_current_model_version()
        plot_filename = f"IntraDayForexPredictor_v{model_version}_evaluation.png"
        plot_path = os.path.join(plot_directory, plot_filename)
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f'Saved plot to: {plot_path}')
        plt.close()

