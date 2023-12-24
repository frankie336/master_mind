import pandas as pd
from configs.path_builder import PathBuilder


class TradingNewsDataMergerService:
    def __init__(self, start_date, end_date, data, news_data, config, unseen_data=False, live_data=False):
        self.end_date = end_date
        self.start_date = start_date
        self.live_data = live_data
        self.unseen_data = unseen_data
        self.data = data
        self.news_data = news_data
        self.config = config
        self.config = PathBuilder(start_date=self.start_date, end_date=self.end_date,
                                  unseen_start_date=self.start_date, unseen_end_date=self.end_date)

        # TODO: WE NEED TO COME UP WITH SOME CLEAN METHOD OF PASSING TRAINING DATA VERSION HERE
        self.data_path = self.config.get_next_data_path(data_type='ForexData', training_data_version='11',
                                                        is_unseen=self.unseen_data)

    def merge_news(self):

        self._normalize_dates()
        self.data = self._merge_trading_with_news()

        return self.data

    def _normalize_dates(self):
        # Ensure the 'date' column is in datetime format
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.news_data['date'] = pd.to_datetime(self.news_data['date'])

        # Remove timezone information if it exists
        if self.data['date'].dt.tz is not None:
            self.data['date'] = self.data['date'].dt.tz_localize(None)
        if self.news_data['date'].dt.tz is not None:
            self.news_data['date'] = self.news_data['date'].dt.tz_localize(None)

        # Sort the news data by date
        self.news_data.sort_values('date', inplace=True)

    def _merge_trading_with_news(self):

        print("Starting news processing...")
        news_processor = NewsDataProcessingService(self.news_data)
        processed_news_data = news_processor.process_news_data()
        print("News processing completed. Sample of processed news data:")
        print(processed_news_data.head())  # Print sample of processed news data

        default_sentiment_score = -1  # Default sentiment score
        default_news_headline = 'None'  # Default news headline

        all_merged_rows = []  # List to hold all the merged rows
        for currency_pair, group in self.data.groupby('currency'):
            # print(f"\nProcessing currency pair: {currency_pair}")
            group_sorted = group.sort_values('date').copy()
            processed_news_data_sorted = processed_news_data.sort_values('date').copy()

            merged_group = pd.merge_asof(group_sorted, processed_news_data_sorted, on='date',
                                         tolerance=pd.Timedelta('1d'), direction='forward')

            # Set default values for rows without matching news
            merged_group['compound_sentiment_score'].fillna(default_sentiment_score, inplace=True)
            merged_group['news_headline_list'].fillna(default_news_headline, inplace=True)

            if not merged_group.empty:
                all_merged_rows.append(merged_group)
            else:
                print(f"No news data to merge for currency pair: {currency_pair}")

        if all_merged_rows:
            # Concatenate all merged rows into a DataFrame
            merged_results = pd.concat(all_merged_rows, ignore_index=True)
        else:
            print("No merged rows to concatenate.")
            return pd.DataFrame()  # Return an empty DataFrame if no rows to merge

        # print("\nFinal merged results:")
        # print(merged_results.head())  # Print sample of final merged results
        print("Merging process completed.")
        return merged_results
