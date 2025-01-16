import unittest
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from challenge.model import DelayModel

class TestModel(unittest.TestCase):

    FEATURES_COLS = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    TARGET_COL = "delay"

    def setUp(self) -> None:
        self.model = DelayModel()
        try:
            self.data = pd.read_csv(filepath_or_buffer="data/data.csv", low_memory=False)
        except FileNotFoundError:
            raise FileNotFoundError("The file data/data.csv could not be found. Ensure the file exists in the specified location.")

    def test_model_preprocess_for_training(self):
        features, target = self.model.preprocess(data=self.data, target_column=self.TARGET_COL)

        expected_columns = pd.get_dummies(self.data['OPERA'].where(self.data['OPERA'].isin([
            "American Airlines", "Sky Airline", "Grupo LATAM", "Copa Air"
        ]), "Other"), prefix='OPERA').columns.tolist() + \
                        pd.get_dummies(self.data['TIPOVUELO'], prefix='TIPOVUELO').columns.tolist() + \
                        pd.get_dummies(self.data['DIANOM'], prefix='DIANOM').columns.tolist() + \
                        pd.get_dummies(self.data['SIGLADES'], prefix='SIGLADES').columns.tolist() + \
                        pd.get_dummies(self.data['MES'], prefix='MES').columns.tolist()

        # Validar el número y nombres de columnas generadas
        self.assertEqual(set(features.columns), set(expected_columns), "Feature columns do not match expected set.")
        self.assertEqual(features.shape[1], len(expected_columns), "Feature count does not match expected columns.")

    def test_model_fit(self):
        features, target = self.model.preprocess(
            data=self.data,
            target_column=self.TARGET_COL
        )

        x_train, x_validation, y_train, y_validation = train_test_split(features, target, test_size=0.33, random_state=42)

        self.model.fit(features=x_train, target=y_train)

        predicted_target = self.model._model.predict(x_validation)

        report = classification_report(y_validation, predicted_target, output_dict=True)

        self.assertLess(report["0"].get("recall", 0), 0.95, "Recall for class 0 exceeds expected threshold.")
        self.assertGreater(report["1"].get("recall", 0), 0.60, "Recall for class 1 is below expected threshold.")
        self.assertGreater(report["1"].get("f1-score", 0), 0.30, "F1-score for class 1 is below expected threshold.")

    def test_model_predict(self):
        features, target = self.model.preprocess(data=self.data, target_column=self.TARGET_COL)
    
        self.model.fit(features=features, target=target)

        predicted_targets = self.model.predict(features=features)

        self.assertIsInstance(predicted_targets, list, "Predicted targets should be a list.")
        self.assertEqual(len(predicted_targets), features.shape[0], "Number of predictions does not match number of samples.")
        self.assertTrue(all(isinstance(predicted_target, int) for predicted_target in predicted_targets), "All predicted targets should be integers.")

if __name__ == '__main__':
    unittest.main()
