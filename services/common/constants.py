# services/common/constants.py
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('FINANCIAL_MODELING_PREP_API_KEY')


START_DATE = '2016-01-01'
END_DATE = '2023-11-21'

CURRENCY_PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]

INDICATORS = ['GDP', 'realGDP', 'nominalPotentialGDP', 'realGDPPerCapita',
                      'federalFunds', 'CPI', 'inflationRate',
                      'unemploymentRate', 'totalNonfarmPayroll',
                      'consumerSentiment', 'retailSales',
                      'durableGoods',
                      '3MonthOr90DayRatesAndYieldsCertificatesOfDeposit']

DATA_COLUMN_ORDERING = ['open', 'low', 'high', 'close', 'volume', 'time_of_day', 'Mid-price', 'Price Dynamics',
                        'LR_Slope', 'prev_Mid-price', 'prev_low', 'prev_high', 'prev_close', 'prev_volume',
                        'prev_open', 'Mid-price-lag-1', 'Mid-price-lag-2', 'Mid-price-lag-3', 'Mid-price-lag-4',
                        'Mid-price-lag-5', 'Mid-price-lag-6', 'Mid-price-lag-7', 'Mid-price-lag-8', 'Mid-price-lag-9',
                        'Mid-price-lag-10', 'Mid-price-lag-11', 'Mid-price-lag-12', 'Mid-price-lag-13',
                        'Mid-price-lag-14', 'Mid-price-lag-15', 'Mid-price-lag-16', 'Mid-price-lag-17',
                        'Mid-price-lag-18', 'Mid-price-lag-19', 'Mid-price-lag-20', 'Mid-price-lag-21',
                        'Mid-price-lag-22', 'Mid-price-lag-23', 'Mid-price-lag-24', 'Mid-price-lag-25',
                        'Mid-price-lag-26', 'Mid-price-lag-27', 'Mid-price-lag-28', 'Mid-price-lag-29',
                        'Mid-price-lag-30', 'Mid-price-lag-31', 'Mid-price-lag-32', 'Mid-price-lag-33',
                        'Mid-price-lag-34', 'Mid-price-lag-35', 'Mid-price-lag-36', 'Mid-price-lag-37',
                        'Mid-price-lag-38', 'Mid-price-lag-39', 'Mid-price-lag-40', 'Mid-price-lag-41',
                        'Mid-price-lag-42', 'Mid-price-lag-43', 'Mid-price-lag-44', 'Mid-price-lag-45',
                        'Mid-price-lag-46', 'Mid-price-lag-47', 'Mid-price-lag-48', 'Mid-price-lag-49',
                        'Mid-price-lag-50', 'Mid-price-lag-51', 'Mid-price-lag-52', 'Mid-price-lag-53',
                        'Mid-price-lag-54', 'Mid-price-lag-55', 'Mid-price-lag-56', 'Mid-price-lag-57',
                        'Mid-price-lag-58', 'Mid-price-lag-59', 'Mid-price-lag-60', 'volume-lag-1', 'volume-lag-2',
                        'volume-lag-3', 'volume-lag-4', 'volume-lag-5', 'volume-lag-6', 'volume-lag-7', 'volume-lag-8',
                        'volume-lag-9', 'volume-lag-10', 'volume-lag-11', 'volume-lag-12', 'volume-lag-13',
                        'volume-lag-14', 'volume-lag-15', 'volume-lag-16', 'volume-lag-17', 'volume-lag-18',
                        'volume-lag-19', 'volume-lag-20', 'volume-lag-21', 'volume-lag-22', 'volume-lag-23',
                        'volume-lag-24', 'volume-lag-25', 'volume-lag-26', 'volume-lag-27', 'volume-lag-28',
                        'volume-lag-29', 'volume-lag-30', 'volume-lag-31', 'volume-lag-32', 'volume-lag-33',
                        'volume-lag-34', 'volume-lag-35', 'volume-lag-36', 'volume-lag-37', 'volume-lag-38',
                        'volume-lag-39', 'volume-lag-40', 'volume-lag-41', 'volume-lag-42', 'volume-lag-43',
                        'volume-lag-44', 'volume-lag-45', 'volume-lag-46', 'volume-lag-47', 'volume-lag-48',
                        'volume-lag-49', 'volume-lag-50', 'volume-lag-51', 'volume-lag-52', 'volume-lag-53',
                        'volume-lag-54', 'volume-lag-55', 'volume-lag-56', 'volume-lag-57', 'volume-lag-58',
                        'volume-lag-59', 'volume-lag-60', 'SMA', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                        'upper_band', 'middle_band', 'lower_band', 'slowk', 'slowd', 'ATR', 'OBV', 'ADX', 'CCI',
                        'MFI', 'Target', 'previous', 'estimate', 'actual', 'change', 'changePercentage',
                        'event_encoded', 'impact_encoded', 'compound_sentiment_score',
                        '3MonthOr90DayRatesAndYieldsCertificatesOfDeposit', 'CPI', 'GDP', 'consumerSentiment',
                        'durableGoods', 'federalFunds', 'inflationRate', 'nominalPotentialGDP', 'realGDP',
                        'realGDPPerCapita', 'retailSales', 'totalNonfarmPayroll', 'unemploymentRate']


