import pandas as pd
from services.common.constants import INDICATORS
import requests


class EconomicIndicatorsFetcherService:
    def __init__(self, api_key, start_date, end_date, ec_start_date=None, ec_end_date=None, is_live=False):
        self.api_key = api_key
        self.start_date = start_date or 'default_start_date'  # Use a default or configuration
        self.end_date = end_date or 'default_end_date'  # Use a default or configuration
        self.ec_start_date = ec_start_date or self.start_date
        self.ec_end_date = ec_end_date or self.end_date
        self.is_live = is_live

    def fetch_indicator(self, name):
        start_date, end_date = (self.ec_start_date, self.ec_end_date) if self.is_live else (
        self.start_date, self.end_date)
        endpoint = f"https://financialmodelingprep.com/api/v4/economic?name={name}&from={start_date}&to={end_date}&apikey={self.api_key}"

        try:
            response = requests.get(endpoint)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return pd.DataFrame(response.json())
        except requests.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except Exception as err:
            print(f"An error occurred: {err}")

        return pd.DataFrame()  # Return an empty DataFrame in case of an error

    def fetch_all_indicators(self):
        all_data = pd.DataFrame()

        for indicator in INDICATORS:  # Ensure INDICATORS is defined or passed as an argument
            try:
                data = self.fetch_indicator(indicator)
                if not data.empty:
                    all_data = pd.concat([all_data, data.assign(Indicator=indicator)], ignore_index=True)
            except Exception as e:
                print(f"Failed to fetch data for {indicator}: {e}")
                continue

        if 'date' not in all_data.columns or all_data.empty:
            return None

        pivot_data_df = all_data.pivot(index='date', columns='Indicator', values='value').reset_index()
        return pivot_data_df
