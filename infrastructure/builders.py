# infrastructure/builders.py
from services.fetch_service.price_data_chunk_fetcher import ForexPriceDataChunkFetcher
from services.fetch_service.economic_calendar_service import EconomicCalendarFetcherService
from services.fetch_service.Indicators_service import EconomicIndicatorsFetcherService
from services.fetch_service.news_fetcher_service import NewsFetcherService
from services.common.tools import SentimentAnalyzerService
from services.feature_engineering_service.feature_engineering_service import FeatureEngineeringService
from services.common.tools import DataFrameSaver

from services.common.constants import CURRENCY_PAIRS


class DataOrchestratorBuilder:
    def __init__(self, api_key, start_date, end_date, ec_start_date, ec_end_date, config):
        self.api_key = api_key
        self.start_date = start_date
        self.end_date = end_date
        self.ec_start_date = ec_start_date
        self.ec_end_date = ec_end_date
        self.config = config

    def build_price_data_fetcher(self):
        # Construct and return the ForexPriceDataChunkFetcher
        forex_data_fetcher_service = ForexPriceDataChunkFetcher(
            start_date=self.start_date, end_date=self.end_date,
            api_key=self.api_key, currency_pairs=CURRENCY_PAIRS)
        return forex_data_fetcher_service

    def build_calendar_data_fetcher(self):
        # Construct and return the EconomicCalendarFetcherService
        calendar_data_fetcher_service = EconomicCalendarFetcherService(
            api_key=self.api_key,
            start_date=self.start_date,
            end_date=self.end_date,
            ec_start_date=self.ec_start_date,
            ec_end_date=self.ec_end_date,
            is_live=False)  # Set to True if fetching live data
        return calendar_data_fetcher_service

    def build_news_fetcher(self):
        # Construct and return the NewsFetcherService
        # Assuming you have a sentiment analyzer service for news sentiment
        sentiment_analyzer_service = SentimentAnalyzerService()
        news_fetcher_service = NewsFetcherService(
            sentiment_analyzer_service=sentiment_analyzer_service,
            api_key=self.api_key,
            start_date=self.start_date,
            end_date=self.end_date,
            ec_start_date=self.ec_start_date,
            ec_end_date=self.ec_end_date,
            is_live=False)  # Set to True if fetching live news
        return news_fetcher_service

    def build_econ_indicator_fetcher(self):
        # Construct and return the EconomicIndicatorsFetcherService
        econ_indicator_fetcher_service = EconomicIndicatorsFetcherService(
            api_key=self.api_key,
            start_date=self.start_date,
            end_date=self.end_date,
            ec_start_date=self.ec_start_date,
            ec_end_date=self.ec_end_date,
            is_live=False)  # Set to True if fetching live indicators
        return econ_indicator_fetcher_service

    def build_feature_engineer(self):
        # Construct and return the FeatureEngineeringService
        feature_engineering_service = FeatureEngineeringService()
        return feature_engineering_service

    def build_data_saver(self):
        # Construct and return the DataFrameSaver
        data_saver_service = DataFrameSaver()
        return data_saver_service

    def build(self):
        # Use the individual build methods to construct each component
        price_data_fetcher = self.build_price_data_fetcher()
        calendar_data_fetcher = self.build_calendar_data_fetcher()
        news_fetcher = self.build_news_fetcher()
        econ_indicator_fetcher = self.build_econ_indicator_fetcher()
        feature_engineer = self.build_feature_engineer()
        data_saver = self.build_data_saver()

        from orchestrators.data_orchestrator import DataOrchestrator

        # Create and return a DataOrchestrator instance with all services
        return DataOrchestrator(
            price_data_fetcher=price_data_fetcher,
            calendar_data_fetcher=calendar_data_fetcher,
            news_fetcher=news_fetcher,
            econ_indicator_fetcher=econ_indicator_fetcher,
            feature_engineer=feature_engineer,
            data_saver=data_saver
        )
