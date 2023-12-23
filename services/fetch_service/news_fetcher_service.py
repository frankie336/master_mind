from time import time
from datetime import datetime
import random
import requests
import pandas as pd


class NewsFetcherService:
    """
    Fetches forex news data relevant to forex markets within a specified date range.

    Attributes:
        start_date (str): The start date for the news data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the news data in 'YYYY-MM-DD' format.
        api_key (str): The API key for accessing the news data source.
    """
    def __init__(
            self, sentiment_analyzer_service, api_key, start_date, end_date, ec_start_date, ec_end_date, is_live=False):

        self.start_date = start_date
        self.end_date = end_date
        self.ec_start_date = ec_start_date
        self.ec_end_date = ec_end_date
        self.api_key = api_key
        self.news_item = ''
        self.is_live = is_live
        self.sentiment_analyzer_service = sentiment_analyzer_service

    def fetch_news_data(self, enable_random_wait=False):

        start_date, end_date = self.start_date, self.end_date
        if self.is_live:
            start_date, end_date = self.ec_start_date, self.ec_end_date

        page = 0
        all_news = []
        while True:
            print(page)

            # Adding a random sleep here
            if enable_random_wait:
                time.sleep(random.uniform(0.5, 2.0))

            url = f"https://financialmodelingprep.com/api/v4/forex_news?page={page}&apikey={self.api_key}"
            response = requests.get(url)
            if response.status_code == 200 and response.json():
                news_data = response.json()
                for news_item in news_data:
                    print(news_item)

                    if enable_random_wait:
                        time.sleep(random.uniform(0.5, 2.0))

                    published_date = datetime.fromisoformat(news_item['publishedDate'][:-1])
                    if published_date < datetime.fromisoformat(start_date):
                        return pd.DataFrame(all_news)
                    if published_date <= datetime.fromisoformat(end_date):

                        # Calculate sentiment of the news text.
                        sentiment_score = self.sentiment_analyzer_service.analyze_sentiment(text=news_item['text'])

                        all_news.append({
                            'date': news_item['publishedDate'],
                            'sentiment_score': sentiment_score,
                            'news_headline': news_item.get('title', '')
                        })
                page += 1
            else:
                break

        news = pd.DataFrame(all_news)

        return news
