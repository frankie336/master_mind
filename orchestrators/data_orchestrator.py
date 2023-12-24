# orchestrators/data_orchestrator.py
from services.common.constants import API_KEY, START_DATE, END_DATE, CURRENCY_PAIRS
from infrastructure.infrastructure import DataOrchestratorBuilder
from configs.config import ForexMastermindConfig


class DataOrchestrator:
    def __init__(self, price_data_fetcher, calendar_data_fetcher, news_fetcher, econ_indicator_fetcher, feature_engineer, data_saver):
        self.price_data_fetcher = price_data_fetcher
        self.calendar_data_fetcher = calendar_data_fetcher
        self.news_fetcher = news_fetcher
        self.econ_indicator_fetcher = econ_indicator_fetcher
        self.feature_engineer = feature_engineer
        self.data_saver = data_saver

    def fetch_and_save_data(self):
        # Fetching price data and applying feature engineering
        price_data_df = self.price_data_fetcher.fetch_data()
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
    builder = DataOrchestratorBuilder(api_key=API_KEY, config=ForexMastermindConfig())

    # Build the DataOrchestrator with all the services configured
    orchestrator = builder.build()

    # Now you can use orchestrator to run the data pipeline
    orchestrator.fetch_and_save_data()



