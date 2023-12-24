# services/feature_engineering_service/technicalIn_dicator_service.py
import talib
import pandas as pd


class TechnicalIndicatorService:
    def __init__(self, data):
        self.data = data

    def add_sma(self):
        # Simple Moving Average
        return self.data['Mid-price'].rolling(window=14).mean()

    def add_rsi(self):
        # Relative Strength Index
        return talib.RSI(self.data['Mid-price'], timeperiod=14)

    def add_macd(self):
        # Moving Average Convergence Divergence
        macd, macdsignal, macdhist = talib.MACD(self.data['Mid-price'])
        return macd, macdsignal, macdhist

    def add_bollinger_bands(self):
        # Bollinger Bands
        upperband, middleband, lowerband = talib.BBANDS(self.data['Mid-price'], timeperiod=20)
        return upperband, middleband, lowerband

    def add_stochastic_oscillator(self):
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(self.data['high'], self.data['low'], self.data['close'])
        return slowk, slowd

    def add_atr(self):
        # Average True Range
        return talib.ATR(self.data['high'], self.data['low'], self.data['close'], timeperiod=14)

    def add_obv(self):
        # On-Balance Volume
        return talib.OBV(self.data['close'], self.data['volume'])

    def add_adx(self):
        # Average Directional Movement Index
        return talib.ADX(self.data['high'], self.data['low'], self.data['close'], timeperiod=14)

    def add_cci(self):
        # Commodity Channel Index
        return talib.CCI(self.data['high'], self.data['low'], self.data['close'], timeperiod=14)

    def add_mfi(self):
        # Money Flow Index
        return talib.MFI(self.data['high'], self.data['low'], self.data['close'], self.data['volume'], timeperiod=14)

    def add_technical_indicators(self):
        # Create a dictionary to store all new technical indicators
        indicators = {
            'SMA': self.add_sma(),
            'RSI': self.add_rsi(),
            'MACD': self.add_macd()[0],
            'MACD_signal': self.add_macd()[1],
            'MACD_hist': self.add_macd()[2],
            'upper_band': self.add_bollinger_bands()[0],
            'middle_band': self.add_bollinger_bands()[1],
            'lower_band': self.add_bollinger_bands()[2],
            'slowk': self.add_stochastic_oscillator()[0],
            'slowd': self.add_stochastic_oscillator()[1],
            'ATR': self.add_atr(),
            'OBV': self.add_obv(),
            'ADX': self.add_adx(),
            'CCI': self.add_cci(),
            'MFI': self.add_mfi()
        }

        # Convert the dictionary to a DataFrame
        indicators_df = pd.DataFrame(indicators)

        # Concatenate the new indicators DataFrame with the original data DataFrame
        self.data = pd.concat([self.data, indicators_df], axis=1)

        return self.data
