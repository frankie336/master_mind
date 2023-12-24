# services/fetch_service/news_fetcher_service.py
import requests
import pandas as pd


class EconomicCalendarFetcherService:
    """
    Fetches economic calendar data within a specified date range.

    Attributes:
        start_date (str): The start date for the data range in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data range in 'YYYY-MM-DD' format.
        api_key (str): The API key for accessing the data source.
        is_live (bool): A flag indicating whether to fetch live data or historical data.
    """
    def __init__(self, api_key, start_date, end_date, ec_start_date=None, ec_end_date=None, is_live=False):
        self.start_date = start_date
        self.end_date = end_date
        self.ec_start_date = ec_start_date or start_date  # Fallback to start_date if not provided
        self.ec_end_date = ec_end_date or end_date  # Fallback to end_date if not provided
        self.api_key = api_key
        self.is_live = is_live

    def fetch_calendar_data(self):
        # Determine the actual start and end dates based on whether the data is live
        start_date = self.ec_start_date if self.is_live else self.start_date
        end_date = self.ec_end_date if self.is_live else self.end_date

        # Construct the endpoint URL
        endpoint = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={start_date}&to={end_date}&apikey={self.api_key}"

        # Make the request and return the data as a DataFrame
        try:
            response = requests.get(endpoint)
            response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code

            # If the response is successful, return the data as a DataFrame
            return pd.DataFrame(response.json())
        except requests.RequestException as e:
            print(f"Failed to fetch economic calendar data: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of an error
