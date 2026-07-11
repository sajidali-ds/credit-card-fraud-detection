import logging
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and run basic sanity checks."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download 'creditcard.csv' from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            f"and place it at: {path}"
        )

    df = pd.read_csv(path)
    logger.info("Loaded data with shape: %s", df.shape)

    required_cols = {"Time", "Amount", "Class"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and nulls (data-quality step every real pipeline needs)."""
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    after = len(df)
    logger.info("Removed %d duplicate/null rows (%d -> %d)", before - after, before, after)
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Scale 'Time' and 'Amount' only. The V1-V28 columns are already PCA
    components from the original dataset and are already on a similar scale.
    RobustScaler is used because fraud/'Amount' data has extreme outliers.
    """
    scaler = RobustScaler()
    cols_to_scale = [c for c in ["Time", "Amount"] if c in X_train.columns]

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    joblib.dump(scaler, config.SCALER_PATH)
    logger.info("Scaler fitted on columns %s and saved to %s", cols_to_scale, config.SCALER_PATH)

    return X_train, X_test, scaler


def get_train_test_data():
    """
    Full pipeline entry point: load -> clean -> split -> scale.
    Returns X_train, X_test, y_train, y_test ready for modelling.
    """
    df = load_raw_data()
    df = clean_data(df)
    X, y = split_features_target(df)

    logger.info(
        "Class distribution -> genuine: %d (%.4f%%), fraud: %d (%.4f%%)",
        (y == 0).sum(), 100 * (y == 0).mean(),
        (y == 1).sum(), 100 * (y == 1).mean(),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,  # critical for imbalanced data: keep fraud ratio equal in both sets
    )

    X_train, X_test, _ = scale_features(X_train, X_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_data()
    print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
    print("Train fraud cases:", y_train.sum(), " Test fraud cases:", y_test.sum())
