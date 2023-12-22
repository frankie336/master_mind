import pandas as pd


class CategoricalFeatureService:
    def __init__(self, data):
        self.data = data

    def encode_currency_pairs(self):
        """
        Encodes the 'currency' column with integers using label encoding.

        Returns:
            pd.DataFrame: DataFrame with the encoded currency column.
        """
        label_encoder = LabelEncoder()
        currency_encoded = label_encoder.fit_transform(self.data['currency'])

        # Convert the NumPy array to a DataFrame
        currency_encoded_df = pd.DataFrame({'CurrencyPair_encoded': currency_encoded})

        # Concatenate the new DataFrame with the existing DataFrame
        self.data = pd.concat([self.data, currency_encoded_df], axis=1)

        return self.data

    def encode_events_and_impact(self):
        # Encode events
        unique_events = self.data['event'].dropna().unique()
        event_map = {event: idx for idx, event in enumerate(unique_events, start=1)}
        event_map[None] = -1  # Special code for rows with no events
        self.data['event_encoded'] = self.data['event'].map(event_map).fillna(-1)

        # Encode impact
        unique_impacts = self.data['impact'].dropna().unique()
        impact_map = {impact: idx for idx, impact in enumerate(unique_impacts, start=1)}
        impact_map[None] = -1  # Special code for rows with no impact data
        self.data['impact_encoded'] = self.data['impact'].map(impact_map).fillna(-1)

        # Handle NaN values in other event-dependent columns
        event_dependent_columns = ['actual', 'previous', 'change', 'changePercentage', 'estimate']
        for col in event_dependent_columns:
            self.data[col] = self.data[col].fillna(-1)  # Replace NaN with -1 or another suitable default

        return self.data
