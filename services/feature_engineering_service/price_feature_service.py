import pandas as pd
import numpy as np
from scipy.stats import linregress


class PriceFeatureService:
    def __init__(self, data):
        self.data = data

    def add_price_features(self):

        self._compute_mid_price()
        self._compute_price_dynamics()
        self._add_linear_regression_slope(window_size=5)
        self._create_previous_period_features()

        return self.data

    def _compute_mid_price(self):
        """Computes the mid-price using the average of 'high' and 'low' columns."""
        self.data['Mid-price'] = (self.data['high'].astype(float) + self.data['low'].astype(float)) / 2

    def _compute_price_dynamics(self):
        """Computes the price dynamics using 'close' and 'open' columns."""
        self.data['Price Dynamics'] = self.data['close'].astype(float) - self.data['open'].astype(float)

    def _add_linear_regression_slope(self, window_size=5):
        """
        Adds the slope of the linear regression line of the 'Mid-price' over a rolling window.
        """
        def get_slope(data):
            """Helper function to apply linear regression and return the slope."""
            regression = linregress(np.arange(len(data)), data.values)
            return regression.slope

        # Use rolling window and apply the helper function
        self.data['LR_Slope'] = self.data['Mid-price'].rolling(window=window_size).apply(get_slope, raw=False)

    def _create_previous_period_features(self):
        """
        Creates new columns in the DataFrame by shifting certain price-related columns by one period.

        This method generates lagged (previous period) features for specific price-related columns such as 'Mid-price',
        'low', 'high', 'close', 'volume', and 'open'. These lagged features provide insight into the previous period's
        values, which can be useful for time-series forecasting and analysis.

        The method appends these new lagged feature columns to the existing DataFrame, enhancing the dataset with
        historical context that may be relevant for predictive modeling.

        The resulting DataFrame includes the following additional columns:
        - 'prev_Mid-price': The mid-price from the previous period.
        - 'prev_low': The low price from the previous period.
        - 'prev_high': The high price from the previous period.
        - 'prev_close': The closing price from the previous period.
        - 'prev_volume': The trading volume from the previous period.
        - 'prev_open': The opening price from the previous period.
        """
        shifted_data = pd.DataFrame({
            'prev_Mid-price': self.data['Mid-price'].shift(1),
            'prev_low': self.data['low'].shift(1),
            'prev_high': self.data['high'].shift(1),
            'prev_close': self.data['close'].shift(1),
            'prev_volume': self.data['volume'].shift(1),
            'prev_open': self.data['open'].shift(1),
        })

        self.data = pd.concat([self.data, shifted_data], axis=1)
