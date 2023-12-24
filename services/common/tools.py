# services/common/tools.py
import ast
import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta

import logging
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import torch
from textblob import TextBlob

from configs.path_builder import PathBuilder
from services.common.constants import API_KEY

logger = logging.getLogger(__name__)


class DateTimeFeatureService:
    def __init__(self, data):
        self.data = data

    def add_datetime_features(self):

        self.convert_to_datetime()
        self._add_time_of_day()

        return self.data

    def convert_to_datetime(self):
        """Converts the 'date' column to datetime format."""
        self.data['date'] = pd.to_datetime(self.data['date'])

        return self.data

    def _add_time_of_day(self):
        """Adds a column with the time of day encoded as a fraction."""
        self.data['time_of_day'] = (self.data['date'].dt.hour * 3600 +
                                    self.data['date'].dt.minute * 60 +
                                    self.data['date'].dt.second) / (24 * 3600)


class SentimentAnalyzerService:
    """
    Performs sentiment analysis on text data.
    """

    @staticmethod
    def analyze_sentiment(text):
        """
        Analyzes the sentiment of the given text.

        Args:
            text (str): The text to analyze.

        Returns:
            float: The sentiment polarity score (-1.0 to 1.0).
        """
        blob = TextBlob(text)

        return blob.sentiment.polarity


class DateRangeGenerator:
    """
    This class is responsible for generating date ranges.

    Methods:
    generate_date_ranges(start, end, delta): Generates date ranges between the start and end dates with a specified interval.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_date_ranges(self, start, end, delta):
        """
        Generates a sequence of date ranges from a specified start date to an end date with a given interval.

        Parameters:
        start (str): The start date in 'YYYY-MM-DD' format.
        end (str): The end date in 'YYYY-MM-DD' format.
        delta (timedelta): The time interval for each date range.

        Yields:
        tuple: A tuple of two datetime objects representing the start and end of each interval within the specified date range.

        Example:
        """
        self.logger.info(f"Generating date ranges from {start} to {end} with delta {delta}")
        current = datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.strptime(end, '%Y-%m-%d')
        while current < end_date:
            yield current, min(current + delta, end_date)
            current += delta
        self.logger.info("Date ranges generated successfully")


class FinancialModelingPrepCurrencyConverter:
    """
    This class provides methods to convert currency using the Financial Modeling Prep API.

    Attributes:
        api_key (str): The API key for authenticating with the Financial Modeling Prep API.

    Example usage:
        api_key = 'your_api_key_here'
        converter = FinancialModelingPrepCurrencyConverter(api_key)
        converted_amount = converter.convert_currency(100, 'EUR', 'USD')
        print(f"Converted Amount: {converted_amount}")
    """

    def __init__(self, api_key=API_KEY):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def get_latest_close_price(self, currency_pair):
        url = f"{self.base_url}/historical-chart/1min/{currency_pair}?apikey={self.api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Get the latest 'close' price from the first data point
            latest_data = data[0] if data else {}
            return latest_data.get('close')
        else:
            raise Exception(f"Failed to fetch data: HTTP {response.status_code}")

    def convert_currency(self, amount, from_currency, to_currency):
        currency_pair = f"{from_currency}{to_currency}"
        latest_close_price = self.get_latest_close_price(currency_pair)
        if latest_close_price is not None:
            return amount * latest_close_price
        else:
            raise Exception("Could not find the latest close price for the currency pair")


class ListComparatorService:
    """
    Service class to compare two lists for differences in elements and their index positions.
    This class provides methods to identify missing elements between two lists and compares
    the index positions of matching elements. It is useful for ensuring consistency and order
    between lists that are supposed to be similar or identical.
    """

    def compare_lists(self, list1, list2):
        """
        Compares two lists to find elements that are missing in each list and checks for discrepancies in index positions.

        Args:
            list1 (list): The first list to be compared.
            list2 (list): The second list to be compared.

        Returns:
            tuple of (set, set, list of tuples):
                - The first set contains elements that are present in list2 but missing in list1.
                - The second set contains elements that are present in list1 but missing in list2.
                - The list of tuples contains the elements and their respective index positions in both lists,
                  or a message indicating the element is not present in the second list.
        """
        # Finding missing elements
        missing_in_list1 = set(list2) - set(list1)
        missing_in_list2 = set(list1) - set(list2)

        # Comparing index positions
        index_comparison = []
        for item in list1:
            if item in list2:
                list1_index = list1.index(item)
                list2_index = list2.index(item)
                index_comparison.append((item, list1_index, list2_index))
            else:
                index_comparison.append((item, list1.index(item), 'Not in list2'))

        return missing_in_list1, missing_in_list2, index_comparison


class DataFrameColumnOrderService:
    def __init__(self, column_order):
        """
        Initialize the service with the desired column order.

        Args:
        column_order (list): The desired order of columns.
        """
        self.column_order = column_order

    def enforce_column_order(self, df):
        """
        Enforce the ordering of DataFrame's columns to match the predefined list.

        Args:
        df (pd.DataFrame): The DataFrame whose columns need to be ordered.

        Returns:
        pd.DataFrame: DataFrame with columns ordered as specified.
        """
        # Check if DataFrame columns match the desired column order
        if self.column_order != list(df.columns):
            print("Reordering DataFrame columns to match the specified order.")
            # Reorder the DataFrame columns and handle any missing columns
            reordered_df = df[self.column_order].copy()
            print("Column order has been updated.")
            return reordered_df
        else:
            print("DataFrame columns are already in the correct order.")
            return df


class DataFrameCleanerService:
    """
    Provides utilities for cleaning a pandas DataFrame by removing non-numeric columns.
    """

    @staticmethod
    def find_string_columns(df):
        """
        Removes non-numeric columns from the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be cleaned.

        Returns:
            pd.DataFrame: A DataFrame with non-numeric columns removed.
        """
        # Identify non-numeric columns
        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
        # Drop non-numeric columns and return the cleaned DataFrame
        return df.drop(columns=non_numeric_columns)


class ModelVersionService:
    """
    Service for managing and retrieving model version numbers based on the model files saved.
    It can determine the next version number to use and identify the current version in use.
    """

    def __init__(self, config, in_google_colab):
        """
        Initializes the service with configuration settings and Google Colab context.

        Args:
            config (PathBuilder): Configuration object containing paths and settings.
            in_google_colab (bool): Flag indicating execution within Google Colab environment.
        """
        self.config = config
        self.in_google_colab = in_google_colab

    def get_existing_versions(self):
        """
        Scans the model directory for existing model files and extracts version numbers.

        Returns:
            List of tuples: Each tuple contains version numbers (MAJOR, MINOR, PATCH) for each model.
        """
        model_dir = self.config.get_model_directory(in_google_colab=self.in_google_colab)
        pattern = re.compile(r'IntraDayForexPredictor_v(\d+)\.(\d+)\.(\d+)\.(sav|pt)')
        versions = []

        for filename in os.listdir(model_dir):
            match = pattern.match(filename)
            if match:
                major, minor, patch, _ = match.groups()
                versions.append((int(major), int(minor), int(patch)))

        return versions

    def get_next_model_version(self):
        """
        Determines the next model version incrementing the patch number by one.

        Returns:
            String: The next model version in the format 'MAJOR.MINOR.(PATCH+1)'.
        """
        existing_versions = self.get_existing_versions()
        if not existing_versions:
            return '1.0.0'

        latest_version = max(existing_versions)
        major, minor, patch = latest_version
        next_version = f"{major}.{minor}.{patch + 1}"
        return next_version

    def get_current_model_version(self):
        """
        Retrieves the highest model version from the existing models.

        Returns:
            String: The current model version in the format 'MAJOR.MINOR.PATCH'.
        """
        existing_versions = self.get_existing_versions()
        if not existing_versions:
            return '1.0.0'

        latest_version = max(existing_versions)
        major, minor, patch = latest_version
        current_version = f"{major}.{minor}.{patch}"
        return current_version


class DataHashService:
    """
    Provides services for generating a hash for a given model's state dictionary.
    This can be used to verify the integrity or uniqueness of the model's state.
    """

    @staticmethod
    def generate_data_hash(model):
        """
        Generates a SHA-256 hash for the state dictionary of a given PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model to generate a hash for.

        Returns:
            str: A SHA-256 hash representing the state of the model.
        """
        # Convert the model state dict to a byte stream
        byte_stream = bytearray()
        state_dict = model.state_dict()
        for key in sorted(state_dict):
            byte_stream.extend(state_dict[key].cpu().numpy().tobytes())

        # Create a hash from the byte stream
        data_hash = hashlib.sha256(byte_stream).hexdigest()
        return data_hash


class TimeOfDayEncoder:
    """
    Service class for adding a time-of-day feature to a DataFrame as a fractional representation.
    """

    def __init__(self, data):
        """
        Initializes the encoder with the provided data.

        Args:
            data (pd.DataFrame): The DataFrame to which the time-of-day encoding will be added.
        """
        self.data = data

    def add_time_of_day_fraction(self):
        """
        Adds a new column to the DataFrame representing the time of day as a fraction of a day.

        The 'date' column of the DataFrame is expected to contain datetime objects. The method
        calculates the fraction of the day that has passed at each timestamp and adds it as a new
        column 'time_of_day'.

        Returns:
            pd.DataFrame: The DataFrame with the added 'time_of_day' column.
        """
        # Convert date column to datetime if it's not already
        self.data['date'] = pd.to_datetime(self.data['date'])

        # Calculate time of day as a fraction
        self.data['time_of_day'] = ((self.data['date'].dt.hour * 3600 +
                                     self.data['date'].dt.minute * 60 +
                                     self.data['date'].dt.second) / (24 * 3600))

        return self.data


class RelativeDateGenerator:
    """
    Provides utility methods for calculating date ranges based on the current date.
    """

    @staticmethod
    def get_yesterday_today_date():
        """
        Retrieves the dates for yesterday and today in 'YYYY-MM-DD' format.

        This method calculates the date for the previous day (yesterday) and the current date (today),
        returning both in a standard format. It's useful for generating date ranges that cover the
        last 24 hours.

        Returns:
            tuple of str: A tuple containing two strings, the first being yesterday's date and the
                          second being today's date, both in 'YYYY-MM-DD' format.
        """
        current_date = datetime.now()
        previous_date = current_date - timedelta(days=1)
        return previous_date.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d')


class DeduplicationService:
    """
    Service class for removing duplicate records from a pandas DataFrame.
    """

    def __init__(self, dataframe):
        """
        Initializes the DeduplicationService with the provided DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame to process for duplicate removal.
        """
        self.dataframe = dataframe

    def remove_duplicates(self):
        """
        Removes duplicate rows from the DataFrame and resets the index.

        Returns:
            pd.DataFrame: A DataFrame with duplicates removed and index reset.
        """
        self.dataframe = self.dataframe.drop_duplicates().reset_index(drop=True)
        return self.dataframe

    def remove_duplicates_based_on_columns(self, subset):
        """
        Removes duplicate rows based on specified columns and resets the index.

        Args:
            subset (list): A list of column names to consider for identifying duplicates.

        Returns:
            pd.DataFrame: A DataFrame with duplicates removed (based on the specified columns) and index reset.
        """
        self.dataframe = self.dataframe.drop_duplicates(subset=subset).reset_index(drop=True)
        return self.dataframe


class DataFrameColumnToListService:
    def __init__(self, dataframe, column_name):
        """
        Initialize the service with a pandas DataFrame and the name of the column.

        :param dataframe: pandas DataFrame
        :param column_name: string, the name of the column to convert
        """
        self.dataframe = dataframe
        self.column_name = column_name

    def column_to_list(self):
        """
        Converts the specified column of the DataFrame to a list with no duplicates.

        :return: list, containing unique values from the specified column
        """
        if self.column_name not in self.dataframe.columns:
            raise ValueError(f"Column '{self.column_name}' not found in DataFrame.")

        return list(self.dataframe[self.column_name].drop_duplicates())


class ListDataFrameColumns:
    """
    Service class for extracting and listing the column names of a pandas DataFrame.
    """

    def __init__(self, dataframe):
        """
        Initializes the ListDataFrameColumns service with a specified DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame from which to list column names.
        """
        self.dataframe = dataframe

    def df_column_names_to_list(self):
        """
        Converts the column names of the DataFrame to a list.

        Returns:
            list: A list containing the names of the columns in the DataFrame.
        """
        return list(self.dataframe.columns)


class FFDataLoader:
    """
    A utility class for loading data from CSV files with error handling.
    """

    def load_data(self, data_path):
        """
        Loads data from a CSV file, skipping any lines that cause parsing errors.

        Args:
            data_path (str): The path to the CSV file to load.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the loaded data.
        """
        try:
            data = pd.read_csv(data_path, low_memory=False, on_bad_lines='skip')
        except pd.errors.ParserError as e:
            print(f"Parser Error: {e}")
            data = pd.DataFrame()

        return data


class DataFrameSaver:
    """
    Provides functionality to save pandas DataFrames to CSV files.
    """

    @staticmethod
    def save_df(df, path, append=False):
        """
        Saves a DataFrame to a CSV file, with options to either append or overwrite.

        This method writes the DataFrame to a CSV file at the specified path. If the file exists
        and 'append' is set to True, it appends the DataFrame to the file. Otherwise, it overwrites
        the existing file or creates a new one.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            path (str): The file path to save the DataFrame to.
            append (bool): If True, append data to the file. If False, overwrite the file.

        Logs:
            An informational message indicating the file name and the path where it is saved.
        """
        if append and os.path.exists(path):
            df.to_csv(path, mode='a', header=False, index=False)
        else:
            df.to_csv(path, mode='w', header=True, index=False)

        logging.info(f"DataFrame saved as: {os.path.basename(path)}")
        logging.info(f"Saved at location: {os.path.abspath(path)}")


class AtomicClockSynchronizer:
    """
    Provides utility methods for synchronizing operations with time, particularly aligning to the atomic clock.
    """

    @staticmethod
    def sleep_until_next_minute():
        """
        Pauses the execution until the start of the next minute.

        This method calculates the remaining seconds until the next minute begins and puts the
        current thread to sleep for that duration. It's useful for synchronizing tasks to begin
        at the start of a new minute.

        Note:
            This synchronization is based on the system clock and may not align perfectly with an
            actual atomic clock.
        """
        now = datetime.now()
        seconds_till_next_minute = 60 - now.second - now.microsecond / 1_000_000.0
        time.sleep(seconds_till_next_minute)


class FeatureFetcherSingleton:
    """
    Singleton class that fetches and stores feature data for a specific model version.

    This class ensures that there is only one instance of the feature fetcher throughout
    the application. It fetches the feature data once and then reuses it for subsequent requests.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Ensures that only one instance of the class is created.
        """
        if not cls._instance:
            cls._instance = super(FeatureFetcherSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initializes the FeatureFetcherSingleton instance.
        """
        self.features = None

    def get_features(self, model_version):
        """
        Retrieves features for the specified model version.

        If the features have not been fetched previously, it fetches and caches them.
        Subsequent calls return the cached features.

        Args:
            model_version (str): The model version for which features are to be fetched.

        Returns:
            list: A list of features for the specified model version.
        """
        if self.features is None:
            self.features = FeatureFetcherSingleton._fetch_features(version=model_version)

        return self.features

    @staticmethod
    def _fetch_features(version):
        """
        Fetches the features for a given model version.

        This is a placeholder method and should be replaced with actual logic to fetch features.

        Args:
            version (str): The model version for which features are to be fetched.

        Returns:
            list: A list of fetched features.
        """
        features = GetTrainedFeatures(model_version=version).get_the_features()
        return features


class LoadModelAndScaler:
    """
    Facilitates the loading of a machine learning model and its associated scaler.

    This class is designed to handle the loading of a pre-trained PyTorch model and
    a scaler, typically used for data normalization or standardization in the model's
    preprocessing pipeline.
    """

    def __init__(self, model_path, scaler_path, model_version=None):
        """
        Initializes the LoadModelAndScaler instance with model and scaler paths.

        Args:
            model_path (str): Path to the saved PyTorch model file.
            scaler_path (str): Path to the saved scaler file (e.g., using joblib).
            model_version (str, optional): Version of the model. Defaults to 'v0.0' if not provided.
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.version = model_version if model_version is not None else "v0.0"
        self.model = None
        self.scaler = None

    def load(self):
        """
        Loads the model and scaler from their respective file paths.

        The method automatically detects the availability of a GPU and loads the model
        accordingly. It assumes that the model is saved in PyTorch's format and the scaler
        is saved using joblib.

        Returns:
            tuple: A tuple containing the loaded model and scaler.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the PyTorch model
        self.model = torch.load(self.model_path, map_location=device)
        print(f"Model loaded from {self.model_path}")

        # Load the scaler
        self.scaler = joblib.load(self.scaler_path)
        print(f"Scaler loaded from {self.scaler_path}")

        return self.model, self.scaler


class GetTrainedFeatures:
    """
    Retrieves the list of features used in a specific version of the trained model.

    This class is responsible for loading the model registry and extracting the feature list
    for a given model version.
    """

    def __init__(self, model_version):
        """
        Initializes the GetTrainedFeatures instance.

        Args:
            model_version (str): The version of the model for which features are to be retrieved.
        """
        self.config = PathBuilder()
        self.model_version = model_version
        self.registry_df = self.load_model_registry()

    def load_model_registry(self):
        """
        Loads the model registry data from the specified path.

        Returns:
            pd.DataFrame: A DataFrame containing the model registry data.
        """
        model_registry_path = self.config.get_model_registry_path()
        data_loader = FFDataLoader()
        return data_loader.load_data(data_path=model_registry_path)

    def get_the_features(self):
        """
        Extracts the feature list for the specified model version from the model registry.

        Returns:
            list: A list of features used in the specified model version.
            If no record is found for the model version, returns an empty list.
        """
        filtered_df = self.registry_df[self.registry_df['version'].isin([self.model_version])].copy()

        if not filtered_df.empty:
            feature_list = ast.literal_eval(filtered_df['feature_list'].iloc[0])
            return feature_list
        else:
            return []


class GetModelParams:
    """
    Retrieves model hyperparameters from the model registry for specified model evaluations.

    This class is designed to access the model registry, extract, and parse the hyperparameters
    for a set of specified model versions.
    """

    def __init__(self, eval_list):
        """
        Initializes the GetModelParams instance.

        Args:
            eval_list (list): A list of model versions for which hyperparameters are to be retrieved.
        """
        self.config = PathBuilder()
        self.registry_df = FFDataLoader().load_data(data_path=self.config.get_model_registry_path())
        self.eval_list = eval_list

    def get_params(self):
        """
        Extracts and parses hyperparameters for the specified model versions.

        Returns:
            dict: A dictionary mapping model versions to their hyperparameters.
            If parsing errors occur, the problematic entries are skipped.
        """
        filtered_df = self.registry_df[self.registry_df['version'].isin(self.eval_list)]
        params_dict = {}

        for index, row in filtered_df.iterrows():
            param_str = row['model_hyperparameters']
            try:
                # Ensure JSON compatibility by replacing single quotes with double quotes
                param_str = param_str.replace("'", '"')
                actual_dict = json.loads(param_str)
                params_dict[row['model_version']] = actual_dict
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for model_version {row['version']}: {e}")

        return params_dict


class GetTrainedSampleSize:
    """
    Retrieves the sample size used for training for specified model evaluations.

    This class accesses the model registry to extract the training sample sizes for a given set of
    model versions, facilitating an understanding of the data volume used in each model's training.
    """

    def __init__(self, eval_list):
        """
        Initializes the GetTrainedSampleSize instance.

        Args:
            eval_list (list): A list of model versions for which training sample sizes are to be retrieved.
        """
        self.config = PathBuilder()
        self.registry_df = FFDataLoader().load_data(data_path=self.config.get_model_registry_path())
        self.eval_list = eval_list

    def get_sample_size(self):
        """
        Extracts the training sample sizes for the specified model versions.

        Returns:
            list: A list of sample sizes corresponding to each model version in the evaluation list.
        """
        filtered_df = self.registry_df[self.registry_df['version'].isin(self.eval_list)].copy()
        sample_sizes = filtered_df['total_samples'].to_list()
        return sample_sizes


class DataFrameExplorer:
    """
    A utility class for exploring and analyzing a DataFrame.

    This class offers various methods to inspect the data, such as viewing basic information,
    memory usage, data types, unique values, correlations, and more.
    """

    def __init__(self, file_path=None, dataframe=None):
        """
        Initializes the DataFrameExplorer with either a file path or an existing DataFrame.

        Args:
            file_path (str, optional): Path to a CSV file to load into a DataFrame.
            dataframe (pd.DataFrame, optional): An existing DataFrame to analyze.

        Raises:
            ValueError: If neither a file path nor a DataFrame is provided.
        """
        if file_path:
            self.df = pd.read_csv(file_path)
        elif dataframe is not None:
            self.df = dataframe
        else:
            raise ValueError("You must provide either a file path or a DataFrame.")

    def get_basic_info(self):
        """
        Displays basic information about the DataFrame including its shape, column names,
        data types, and count of missing values in each column.
        """
        print(f"Shape of the DataFrame: {self.df.shape}")
        print(f"Column Names: {self.df.columns.tolist()}")
        print(f"Data Types:\n{self.df.dtypes}")
        print(f"Missing Values:\n{self.df.isnull().sum()}")

    def get_memory_usage(self):
        """
        Calculates the memory usage of each column in the DataFrame.

        Returns:
            pd.Series: A series with memory usage of each column in bytes.
        """
        return self.df.memory_usage(deep=True)

    def get_data_types(self):
        """
        Retrieves the data types of each column in the DataFrame.

        Returns:
            pd.Series: A series with the data types of each column.
        """
        return self.df.dtypes

    def get_unique_values(self):
        """
        Counts the number of unique values in each column of the DataFrame.

        Returns:
            pd.Series: A series with the count of unique values for each column.
        """
        return self.df.nunique()

    def get_value_counts(self, column):
        """
        Calculates the frequency of each unique value in a specified column.

        Args:
            column (str): The column name for which to count unique values.

        Returns:
            pd.Series: A series with the count of unique values in the specified column.
        """
        return self.df[column].value_counts()

    def plot_correlation_matrix(self):
        """
        Plots a heatmap of the correlation matrix for the DataFrame's columns.
        """
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.show()

    def get_correlation_with_target(self, target_column):
        """
        Calculates the correlation of each feature with a specified target column.

        Args:
            target_column (str): The target column to correlate with.

        Returns:
            pd.Series: A series with correlation values of each column with the target.
        """
        return self.df.corr()[target_column].sort_values(ascending=False)

    def get_duplicate_rows_count(self):
        """
        Counts the number of duplicate rows in the DataFrame.

        Returns:
            int: The number of duplicate rows.
        """
        return self.df.duplicated().sum()


class Balance:
    """
    Balances the sample distribution of a binary target column in a DataFrame.

    This class provides functionality to balance the number of samples between two classes
    in a binary classification dataset by undersampling the majority class.
    """

    def __init__(self, data, column):
        """
        Initializes the Balance instance.

        Args:
            data (pd.DataFrame): The DataFrame containing the data to be balanced.
            column (str): The name of the binary target column in the DataFrame.
        """
        self.data = data
        self.column = column

    def balance_samples(self):
        """
        Balances the number of samples between the two classes in the target column.

        The method undersamples the majority class to match the count of the minority class,
        resulting in an equal number of samples for both classes.

        Returns:
            pd.DataFrame: A balanced DataFrame with an equal number of samples for both classes.
        """
        # Count the instances of each class
        count_class_0, count_class_1 = self.data[self.column].value_counts()

        # Divide the DataFrame by class
        df_class_0 = self.data[self.data[self.column] == 0]
        df_class_1 = self.data[self.data[self.column] == 1]

        # Perform undersampling of the majority class
        df_class_0_under = df_class_0.sample(count_class_1)
        self.data = pd.concat([df_class_0_under, df_class_1], axis=0)

        return self.data


class SampleData:
    """
    Provides functionality to sample a subset of data from a DataFrame.

    This class allows for random sampling of data from a DataFrame based on a specified
    percentage, facilitating operations on a smaller, more manageable subset of data.
    """

    @staticmethod
    def sample_data(n_percent, data):
        """
        Samples a specified percentage of data from a DataFrame.

        Args:
            n_percent (int): The percentage of data to sample, ranging from 0 to 100.
            data (pd.DataFrame): The DataFrame from which to sample.

        Returns:
            pd.DataFrame: A DataFrame containing the sampled subset of data.
        """
        if n_percent == 100:
            return data

        # Calculate the number of samples to select based on the percentage
        number_of_samples = int(len(data) * (n_percent / 100.0))

        # Sample the data using a fixed random state for reproducibility
        sampled_data = data.sample(n=number_of_samples, random_state=42)
        return sampled_data


class ForexDataInverterService:
    """
    Provides functionality to invert the 'Target' column in Forex data.

    This service class loads Forex data, inverts the 'Target' column based on the
    comparison of 'open' and 'close' prices, and saves the modified data.
    """

    def __init__(self, version):
        """
        Initializes the ForexDataInverterService with a specific data version.

        Args:
            version (str): The version of the data to be processed.
        """
        self.config = PathBuilder()
        self.version = version
        self.data_path = self.config.get_next_data_path(data_type='ForexData', training_data_version=version)

    def load_data(self):
        """
        Loads Forex data from a CSV file specified by the data path.
        """
        self.data = pd.read_csv(self.data_path)

    def invert_target(self):
        """
        Inverts the 'Target' column in the loaded Forex data.

        The 'Target' column is inverted based on the comparison of the 'open' and 'close' prices.
        The method assumes the existence of these columns in the DataFrame.
        """
        if 'Target' in self.data.columns:
            self.data.drop('Target', axis=1, inplace=True)

        self.data['Target'] = (self.data['open'].shift(-1) < self.data['close']).astype(int)
        self.data.dropna(inplace=True)  # Drop the last row since it will have a NaN 'Target'

    def save_data(self):
        """
        Saves the modified data with the inverted 'Target' column to a CSV file.
        """
        inverted_data_path = self.config.get_inverted_data_path('ForexData', self.version)
        self.data.to_csv(inverted_data_path, index=False)
        print(f"File saved to {inverted_data_path}")

    def run_inversion(self):
        """
        Executes the entire process: loading data, inverting the 'Target', and saving the modified data.
        """
        self.load_data()
        self.invert_target()
        self.save_data()
