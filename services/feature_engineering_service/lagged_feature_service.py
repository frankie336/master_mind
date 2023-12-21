


class LaggedFeatureService:
    def __init__(self, data):
        self.data = data

    def create_lagged_features(self, n_lags=60):
        """
        Creates lagged features for specific columns.

        Args:
            n_lags (int): Number of lagged features to create.

        Returns:
            pd.DataFrame: DataFrame with lagged features added.
        """
        columns_to_lag = ['Mid-price', 'volume']
        original_number_of_columns = self.data.shape[1]
        lagged_frames = [self.data]

        for col in columns_to_lag:
            for lag in range(1, n_lags + 1):
                lagged_col_name = f'{col}-lag-{lag}'
                lagged_frame = self.data[col].shift(lag).to_frame(lagged_col_name)
                lagged_frames.append(lagged_frame)

        self.data = pd.concat(lagged_frames, axis=1)

        # Validate the number of columns in the resulting DataFrame
        expected_columns = original_number_of_columns + len(columns_to_lag) * n_lags
        assert self.data.shape[1] == expected_columns, "Mismatch in expected number of columns."

        return self.data
