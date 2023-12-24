# services/feature_engineering_service/feature_engineering_service.py
from services.common.tools import DateTimeFeatureService
from services.feature_engineering_service.technicalIn_dicator_service import TechnicalIndicatorService
from services.feature_engineering_service.target_feature_service import TargetFeatureService
from services.feature_engineering_service.price_feature_service import PriceFeatureService
from configs.path_builder import PathBuilder


class FeatureEngineeringService:
    def __init__(self, live_data=False):
        self.live_data = live_data
        self.config = PathBuilder()

    def create_lagged_features(self, data, n_lags=60):
        """
        Creates lagged features for specific columns.
        """
        columns_to_lag = ['Mid-price', 'volume']
        for col in columns_to_lag:
            for lag in range(1, n_lags + 1):
                lagged_col_name = f'{col}-lag-{lag}'
                data[lagged_col_name] = data[col].shift(lag)

        return data

    def apply_feature_engineering(self, data):
        # Applying datetime features
        datetime_service = DateTimeFeatureService(data)
        data = datetime_service.add_datetime_features()

        price_service = PriceFeatureService(data)
        data = price_service.add_price_features()

        # Creating lagged features
        lagged_feature_service = self.create_lagged_features(data=data)
        data = lagged_feature_service.create_lagged_features()

        # Adding technical indicators
        technical_service = TechnicalIndicatorService(data)
        data = technical_service.add_technical_indicators()

        # Adding target if not live data
        if not self.live_data:
            target_service = TargetFeatureService(data=data)
            data = target_service.add_binary_target()

        return data




