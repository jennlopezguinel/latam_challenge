import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Tuple, Union, List
import sys

class DelayModel:

    def __init__(self):
        self._model = None

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        data['OPERA'] = data['OPERA'].where(data['OPERA'].isin([
            "American Airlines", "Sky Airline", "Grupo LATAM", "Copa Air"
        ]), "Other")
            
        data['Fecha-I'] = pd.to_datetime(data['Fecha-I'], errors='coerce')
        data['Fecha-O'] = pd.to_datetime(data['Fecha-O'], errors='coerce')
        data.dropna(subset=['Fecha-I', 'Fecha-O'], inplace=True)

        def get_period_day(date):
            if pd.isnull(date):
                return None
            date_time = date.time()
            if datetime.strptime("05:00", '%H:%M').time() <= date_time <= datetime.strptime("11:59", '%H:%M').time():
                return 'morning'
            elif datetime.strptime("12:00", '%H:%M').time() <= date_time <= datetime.strptime("18:59", '%H:%M').time():
                return 'afternoon'
            else:
                return 'night'

        data['period_day'] = data['Fecha-I'].apply(get_period_day)

        def is_high_season(fecha):
            if pd.isnull(fecha):
                return 0
            month_day = fecha.strftime('%m-%d')
            return int(
                '12-15' <= month_day <= '12-31' or
                '01-01' <= month_day <= '03-03' or
                '07-15' <= month_day <= '07-31' or
                '09-11' <= month_day <= '09-30'
            )

        data['high_season'] = data['Fecha-I'].apply(is_high_season)

        def get_min_diff(row):
            if pd.isnull(row['Fecha-I']) or pd.isnull(row['Fecha-O']):
                return 0
            return (row['Fecha-O'] - row['Fecha-I']).total_seconds() / 60

        data['min_diff'] = data.apply(get_min_diff, axis=1)
        data['delay'] = (data['min_diff'] > 15).astype(int)

        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['DIANOM'], prefix='DIANOM'),
            pd.get_dummies(data['SIGLADES'], prefix='SIGLADES'),
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)

        if target_column:
            return features, data[target_column]

        return features

    def fit(self, features, target):
        self._model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self._model.fit(features, target.to_numpy())

    def predict(self, features):
        if self._model is None:
            raise ValueError("Model has not been trained. Please train the model before predicting.")
        return self._model.predict(features).tolist()

    def evaluate(self, features, target):
        predictions = self._model.predict(features)
        auc = roc_auc_score(target, predictions)
        accuracy = accuracy_score(target, predictions)
        return auc, accuracy, predictions

data = pd.read_csv('data/data.csv', low_memory=False)

model = DelayModel()
features, target = model.preprocess(data, target_column='delay')
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

model.fit(x_train, y_train)
auc, accuracy, predictions = model.evaluate(x_test, y_test)

print("AUC: " + str(auc))
print("Accuracy: " + str(accuracy))
print("Predictions: ")
print(predictions)
