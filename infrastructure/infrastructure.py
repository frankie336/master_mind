# infrastructure/infrastructure.py
from services.common.constants import API_KEY, CURRENCY_PAIRS
from services.fetch_service.forex_price_data_chunk_fetcher import ForexPriceDataChunkFetcher
from services.common.constants import START_DATE, END_DATE
from services.fetch_service.news_fetcher_service import NewsFetcherService
from services.fetch_service.economic_calendar_service import EconomicCalendarFetcherService
from services.fetch_service.Indicators_service import EconomicIndicatorsFetcherService
from services.common.tools import SentimentAnalyzerService
from services.feature_engineering_service.feature_engineering_service import FeatureEngineeringService
from services.common.tools import DataFrameSaver


class DataOrchestratorBuilder:
    def __init__(self, api_key, start_date, end_date, ec_start_date, ec_end_date, config):
        self.api_key = api_key
        self.start_date = start_date
        self.end_date = end_date
        self.ec_start_date = ec_end_date
        self.ec_end_date = ec_end_date
        self.config = config

    def get_dates(self):
        # Implement logic to determine start and end dates
        # ...
        return START_DATE, END_DATE

    def build_fetcher_services(self, start_date, end_date):
        # Instantiate and configure fetcher services
        forex_data_fetcher_service = ForexPriceDataChunkFetcher(
            start_date=START_DATE, end_date=END_DATE, api_key=API_KEY, currency_pairs=CURRENCY_PAIRS)

        calendar_data_fetcher_service = EconomicCalendarFetcherService(
            api_key=self.api_key, start_date=self.start_date, end_date=self.end_date)

        sentiment_analyzer_service = SentimentAnalyzerService()

        EconomicIndicatorsFetcherService(
            api_key=self.api_key, start_date=self.start_date, end_date=self.end_date,
            ec_start_date=self.ec_start_date, ec_end_date=self.ec_end_date)

        news_fetcher_service = NewsFetcherService(
            sentiment_analyzer_service=sentiment_analyzer_service,
            api_key=self.api_key, start_date=start_date,
            end_date=end_date, ec_start_date=self.ec_start_date, ec_end_date=self.ec_end_date, is_live=False)

        economic_indicators_fetcher_service = EconomicIndicatorsFetcherService(
            api_key=self.api_key, start_date=self.start_date, end_date=self.end_date,
            ec_start_date=self.ec_start_date, ec_end_date=self.ec_end_date, is_live=False)

        return forex_data_fetcher_service, calendar_data_fetcher_service, news_fetcher_service, economic_indicators_fetcher_service

    def build_feature_engineering_service(self):
        # Instantiate and configure feature engineering service

        feature_engineering_service = FeatureEngineeringService()

        return feature_engineering_service

    def build_data_saver_service(self):
        # Instantiate and configure data saver service

        data_saver_service = DataFrameSaver()

        return data_saver_service

    def build(self):
        # Fetch and configure all services
        start_date, end_date = self.get_dates()
        fetchers = self.build_fetcher_services(start_date, end_date)
        feature_engineer = self.build_feature_engineering_service()
        data_saver = self.build_data_saver_service()

        # Create and return a DataOrchestrator instance with all services
        return DataOrchestrator(
            price_data_fetcher=fetchers[0],
            calendar_data_fetcher=fetchers[1],
            news_fetcher=fetchers[2],
            econ_indicator_fetcher=fetchers[3],
            feature_engineer=feature_engineer,
            data_saver=data_saver
        )
