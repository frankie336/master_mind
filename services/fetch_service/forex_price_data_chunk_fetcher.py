# services/fetch_service/forex_price_data_chunk_fetcher.py
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import requests
import pandas as pd
from services.common.tools import DateRangeGenerator


class ForexPriceDataChunkFetcher:
    def __init__(self, start_date, end_date, api_key, currency_pairs, live_data=False):
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.currency_pairs = currency_pairs  # List of currency pairs
        self.live_data = live_data

    def fetch_price_data(self):
        combined_data = pd.DataFrame()
        for pair in tqdm(self.currency_pairs, desc="Fetching data for all pairs"):
            pair_data = self.fetch_data_in_chunks(pair)
            combined_data = pd.concat([combined_data, pair_data], ignore_index=True)
        return combined_data

    def fetch_data_in_chunks(self, pair):
        all_data = []
        date_range_generator = DateRangeGenerator()
        for start, end in tqdm(date_range_generator.generate_date_ranges(self.start_date, self.end_date, timedelta(days=30)), desc=f"Fetching {pair}"):
            retry_count = 0
            max_retries = 3
            successful = False
            while retry_count < max_retries and not successful:
                endpoint = f"https://financialmodelingprep.com/api/v3/historical-chart/1min/{pair}?from={start.strftime('%Y-%m-%d')}&to={end.strftime('%Y-%m-%d')}&apikey={self.api_key}"
                try:
                    response = requests.get(endpoint)
                    if response.status_code == 200:
                        forex_data = response.json()
                        if forex_data:
                            df = pd.DataFrame(forex_data)
                            df['Pair'] = pair
                            all_data.append(df)
                            successful = True
                    else:
                        retry_count += 1
                        time.sleep(5)
                except Exception as e:
                    retry_count += 1
                    time.sleep(5)
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()