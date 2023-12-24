# services/feature_engineering_service/feature_engineering_service.py
import pandas as pd
from scipy.stats import linregress


class PriceFeatureService:
    """
    Provides methods to calculate and add various price-based features to financial market data.

    Attributes:
        data (pd.DataFrame): The financial market data to which the price features will be added.
    """

    def __init__(self, data):
        """
        Initializes the PriceFeatureService with the given financial market data.

        Args:
            data (pd.DataFrame): The financial market data to which the price features will be added.
        """
        self.data = data

    def add_price_features(self):
        """
        Calculates and adds various price-based features to the data, including mid-price,
        price dynamics, linear regression slope of the mid-price, and lagged features for
        price-related columns.

        Returns:
            pd.DataFrame: The financial market data with the new price-related features added.
        """
        # Computes the mid-price using the average of 'high' and 'low' columns
        self.data['Mid-price'] = (self.data['high'].astype(float) + self.data['low'].astype(float)) / 2

        # Computes the price dynamics using 'close' and 'open' columns
        self.data['Price Dynamics'] = self.data['close'].astype(float) - self.data['open'].astype(float)

        # Adds the slope of the linear regression line of the 'Mid-price' over a rolling window
        self.data['LR_Slope'] = self.data['Mid-price'].rolling(window=5).apply(lambda x: linregress(range(len(x)), x).slope, raw=False)

        # Creates new columns in the DataFrame by shifting certain price-related columns by one period
        self.data = pd.concat([self.data, pd.DataFrame({
            'prev_Mid-price': self.data['Mid-price'].shift(1),
            'prev_low': self.data['low'].shift(1),
            'prev_high': self.data['high'].shift(1),
            'prev_close': self.data['close'].shift(1),
            'prev_volume': self.data['volume'].shift(1),
            'prev_open': self.data['open'].shift(1),
        })], axis=1)

        return self.data
