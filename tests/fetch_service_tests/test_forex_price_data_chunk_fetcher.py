import unittest
from unittest.mock import patch
from services.fetch_service.price_data_chunk_fetcher import ForexPriceDataChunkFetcher
import pandas as pd
from io import StringIO


class TestForexPriceDataChunkFetcher(unittest.TestCase):

    def setUp(self):
        # Setup your known inputs and outputs
        self.api_key = 'test_api_key'
        self.start_date = '2023-01-01'
        self.end_date = '2023-01-31'
        self.pair = 'EURUSD'
        self.fetcher = ForexPriceDataChunkFetcher(None, self.start_date, self.end_date, self.api_key)

    def test_initialization(self):
        # Test the initialization
        self.assertEqual(self.fetcher.api_key, self.api_key)
        self.assertEqual(self.fetcher.start_date, self.start_date)
        self.assertEqual(self.fetcher.end_date, self.end_date)

    @patch('services.fetch_service.forex_price_data_chunk_fetcher.requests.get')
    def test_fetch_data_in_chunks_success(self, mock_get):
        # Simulate a successful response from the API
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = [{"timestamp": "123456789", "open": "1.12345", "close": "1.12346"}]

        data_df = self.fetcher.fetch_data_in_chunks(self.pair)
        self.assertIsInstance(data_df, pd.DataFrame)
        self.assertFalse(data_df.empty)
        self.assertTrue('Pair' in data_df.columns)

    @patch('services.fetch_service.forex_price_data_chunk_fetcher.requests.get')
    def test_fetch_data_in_chunks_failure(self, mock_get):
        # Simulate a failed response from the API
        mock_response = mock_get.return_value
        mock_response.status_code = 500
        mock_response.json.return_value = []

        data_df = self.fetcher.fetch_data_in_chunks(self.pair)
        self.assertIsInstance(data_df, pd.DataFrame)
        self.assertTrue(data_df.empty)


if __name__ == '__main__':
    unittest.main()