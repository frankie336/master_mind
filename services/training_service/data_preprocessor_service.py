import os
import joblib
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from services.common.tools import Balance, DataFrameCleanerService


class DataPreprocessor:
    def __init__(self, batch_size, config, in_google_colab=False):
        self.batch_size = batch_size
        self.config = config
        self.in_google_colab = in_google_colab

    def _balance_dataset(self, data):
        balance = Balance(data=data, column='Target')
        return balance.balance_samples()

    def _drop_nan_values(self, data):
        return data.dropna()

    def drop_column_from_dataframe(self, df, column_name):
        if column_name in df.columns:
            return df.drop(columns=[column_name])
        else:
            raise ValueError(f"The column '{column_name}' does not exist in the DataFrame.")

    def _validate_features(self, X):
        if X.isnull().any().any():
            raise ValueError("NaN values found in dataset.")
        if not all(X[col].dtype.kind in 'fi' for col in X.columns):
            non_numeric_cols = [col for col in X.columns if X[col].dtype.kind not in 'fi']
            raise ValueError(f"Non-numeric data types found in features: {non_numeric_cols}")

    def _pre_stage_data(self, unprocessed_training_data_df, model_version):
        cleaner_service = DataFrameCleanerService()
        training_data_df = cleaner_service.find_string_columns(df=unprocessed_training_data_df)

        training_data_df = self._drop_nan_values(data=training_data_df)
        training_data_df = self._balance_dataset(data=training_data_df)

        scaler = StandardScaler()
        feature_columns = [col for col in training_data_df.columns if col != 'Target' and np.issubdtype(training_data_df[col].dtype, np.number)]
        training_data_df[feature_columns] = scaler.fit_transform(training_data_df[feature_columns])

        scaler_path = os.path.join(self.config.get_scaler_directory(in_google_colab=self.in_google_colab), f"IntraDayForexPredictor_v{model_version}_scaler.sav")
        joblib.dump(scaler, scaler_path)

        return training_data_df

    def preprocess_data(self, unprocessed_training_data_df, model_version):
        training_data_scaled = self._pre_stage_data(unprocessed_training_data_df=unprocessed_training_data_df, model_version=model_version)
        print("Pre-staged data:", training_data_scaled)

        features = [col for col in training_data_scaled.columns if col != 'Target']
        print("Features:", features)

        train_loader, val_loader, test_loader = self.create_data_loaders_from_dataframe(df=training_data_scaled, target_column='Target')
        return train_loader, val_loader, test_loader, features

    def create_data_loaders_from_dataframe(self, df, target_column):
        X = df.drop(target_column, axis=1).values
        y = df[target_column].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
        train_size = int(train_ratio * len(X_scaled))
        val_size = int(val_ratio * len(X_scaled))
        test_size = len(X_scaled) - train_size - val_size

        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=(val_ratio + test_ratio), random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), random_state=42)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
