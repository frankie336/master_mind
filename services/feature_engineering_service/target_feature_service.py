# services/feature_engineering_service/target_feature_service.py
import pandas as pd


class TargetFeatureService:
    def __init__(self, data):
        self.data = data

    def add_binary_target(self):
        """
        Adds a binary target column to the DataFrame.
        The target is 1 if the next period's open is higher than the current period's close, else 0.

        Returns:
            pd.DataFrame: DataFrame with the target column added.
        """
        self.data.dropna(inplace=True)
        target_col = (self.data['open'].shift(-1) > self.data['close']).astype(int)
        self.data = pd.concat([self.data, target_col.rename('Target')], axis=1)
        self.data = self.data.iloc[:-1]

        return self.data
