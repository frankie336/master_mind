import unittest
from unittest.mock import patch, Mock
from services.fetch_service.news_fetcher_service import NewsFetcherService
from services.common.tools import SentimentAnalyzerService
from services.common.constants import API_KEY
import pandas as pd


class TestNewsFetcherService(unittest.TestCase):

    def setUp(self):
        # Setup your known inputs and outputs
        self.api_key = API_KEY
        self.start_date = '2023-01-01'
        self.end_date = '2023-01-31'
        self.sentiment_analyzer_service = SentimentAnalyzerService()
        self.fetcher = NewsFetcherService(
            self.sentiment_analyzer_service, self.api_key, self.start_date, self.end_date, self.start_date, self.end_date)

    def test_initialization(self):
        # Test the initialization
        self.assertEqual(self.fetcher.api_key, self.api_key)
        self.assertEqual(self.fetcher.start_date, self.start_date)
        self.assertEqual(self.fetcher.end_date, self.end_date)

    @patch('services.fetch_service.news_fetcher_service.requests.get')
    def test_fetch_news_data_success(self, mock_get):
        # Simulate a successful response from the API
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                'publishedDate': '2023-01-15T00:00:00Z',
                'text': 'This is a test news item.',
                'title': 'Test News Title'
            }
        ]

        news_data_df = self.fetcher.fetch_news_data()
        self.assertIsInstance(news_data_df, pd.DataFrame)
        self.assertFalse(news_data_df.empty)
        self.assertTrue('date' in news_data_df.columns)
        self.assertTrue('sentiment_score' in news_data_df.columns)
        self.assertTrue('news_headline' in news_data_df.columns)

    @patch('services.fetch_service.news_fetcher_service.requests.get')
    def test_fetch_news_data_failure(self, mock_get):
        # Simulate a failed response from the API
        mock_response = mock_get.return_value
        mock_response.status_code = 500
        mock_response.json.return_value = []

        news_data_df = self.fetcher.fetch_news_data()
        self.assertIsInstance(news_data_df, pd.DataFrame)
        self.assertTrue(news_data_df.empty)

    @patch('services.fetch_service.news_fetcher_service.requests.get')
    def test_fetch_news_data_loop_exit(self, mock_get):
        # Simulate responses from the API
        # First call returns data, second call returns an empty list to stop the loop
        mock_get.side_effect = [
            Mock(status_code=200, json=lambda: [{
                'publishedDate': '2023-01-15T00:00:00Z',
                'text': 'This is a test news item.',
                'title': 'Test News Title'
            }]),
            Mock(status_code=200, json=lambda: [])
        ]

        # Run the fetch_news_data method which should now exit the loop after the second API call
        news_data_df = self.fetcher.fetch_news_data()
        self.assertIsInstance(news_data_df, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
