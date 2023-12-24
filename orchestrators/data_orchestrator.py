# orchestrators/data_orchestrator.py
from infrastructure.builders import DataOrchestratorBuilder  # Adjust the import path as necessary
from services.common.constants import API_KEY, START_DATE, END_DATE
from configs.path_builder import ForexMastermindConfig


import logging
import pandas as pd
from services.common.constants import CURRENCY_PAIRS


class DataOrchestrator:
    def __init__(self, price_data_fetcher, calendar_data_fetcher,
                 news_fetcher, econ_indicator_fetcher, feature_engineer, data_saver):
        self.price_data_fetcher = price_data_fetcher
        self.calendar_data_fetcher = calendar_data_fetcher
        self.news_fetcher = news_fetcher
        self.econ_indicator_fetcher = econ_indicator_fetcher
        self.feature_engineer = feature_engineer
        self.data_saver = data_saver
        self.logger = logging.getLogger(__name__)

    def fetch_price_data(self):

        currency_pairs = CURRENCY_PAIRS
        combined_data = pd.DataFrame()

        for pair in currency_pairs:
            pair_data = self.price_data_fetcher.fetch_data_in_chunks(pair)
            combined_data = pd.concat([combined_data, pair_data], ignore_index=True)

        return combined_data

    def fetch_and_save_data(self):
        # Fetching price data and applying feature engineering
        price_data_df = self.fetch_price_data()
        price_data_processed = self.feature_engineer.add_price_features(price_data_df)
        self.data_saver.save_data(price_data_processed, 'price_data_processed.csv')

        # Fetching and processing economic calendar data
        calendar_data_df = self.calendar_data_fetcher.fetch_calendar_data()
        calendar_data_processed = self.feature_engineer.add_calendar_features(calendar_data_df)
        self.data_saver.save_data(calendar_data_processed, 'calendar_data_processed.csv')

        # Fetching and saving news data
        news_data_df = self.news_fetcher.fetch_news_data()
        self.data_saver.save_data(news_data_df, 'news_data.csv')

        # Fetching and saving economic indicator data
        econ_indicator_data = self.econ_indicator_fetcher.fetch_all_indicators()
        self.data_saver.save_data(econ_indicator_data, 'econ_indicator_data.csv')


if __name__ == '__main__':
    # Initialize the builder with necessary parameters
    builder = DataOrchestratorBuilder(
        api_key=API_KEY,
        start_date=START_DATE,
        end_date=END_DATE,
        ec_start_date='2023-01-01',
        ec_end_date='2023-12-31',
        config=ForexMastermindConfig()
    )
    # Build the DataOrchestrator with all the services configured
    orchestrator = builder.build()
    orchestrator.fetch_and_save_data()