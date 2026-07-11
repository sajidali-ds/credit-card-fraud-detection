import logging

import joblib
import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FraudPredictor:
    def __init__(self, model_path=config.BEST_MODEL_PATH, scaler_path=config.SCALER_PATH):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        logger.info("Loaded model from %s and scaler from %s", model_path, scaler_path)

    def predict(self, transactions: pd.DataFrame, threshold: float = 0.5):
        """
        transactions: DataFrame with the same raw columns as creditcard.csv
                      (Time, V1..V28, Amount) - WITHOUT the 'Class' column.
        Returns the original dataframe with two extra columns:
          fraud_probability, is_fraud
        """
        X = transactions.copy()
        cols_to_scale = [c for c in ["Time", "Amount"] if c in X.columns]
        X[cols_to_scale] = self.scaler.transform(X[cols_to_scale])

        probabilities = self.model.predict_proba(X)[:, 1]

        result = transactions.copy()
        result["fraud_probability"] = probabilities
        result["is_fraud"] = (probabilities >= threshold).astype(int)
        return result


if __name__ == "__main__":
    # Example usage with a couple of rows pulled from the test set.
    from src.data_preprocessing import load_raw_data

    df = load_raw_data()
    sample = df.drop(columns=["Class"]).sample(5, random_state=config.RANDOM_STATE)

    predictor = FraudPredictor()
    scored = predictor.predict(sample)
    print(scored[["Amount", "fraud_probability", "is_fraud"]])
