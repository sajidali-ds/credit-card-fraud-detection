import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import clean_data, scale_features, split_features_target


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(42)
    n_normal, n_fraud = 200, 10

    def make_rows(n, fraud_flag):
        data = {"Time": rng.integers(0, 172792, n)}
        for i in range(1, 29):
            data[f"V{i}"] = rng.normal(0 if not fraud_flag else 2, 1, n)
        data["Amount"] = rng.exponential(50, n)
        data["Class"] = fraud_flag
        return pd.DataFrame(data)

    df = pd.concat([make_rows(n_normal, 0), make_rows(n_fraud, 1)], ignore_index=True)
    return df


def test_clean_data_removes_duplicates(synthetic_df):
    dup_df = pd.concat([synthetic_df, synthetic_df.iloc[:5]], ignore_index=True)
    cleaned = clean_data(dup_df)
    assert len(cleaned) == len(synthetic_df)


def test_clean_data_removes_nulls(synthetic_df):
    df_with_nulls = synthetic_df.copy()
    df_with_nulls.loc[0, "Amount"] = np.nan
    cleaned = clean_data(df_with_nulls)
    assert cleaned.isnull().sum().sum() == 0
    assert len(cleaned) == len(synthetic_df) - 1


def test_split_features_target(synthetic_df):
    X, y = split_features_target(synthetic_df)
    assert "Class" not in X.columns
    assert set(y.unique()) == {0, 1}
    assert len(X) == len(y)


def test_scale_features_only_scales_time_and_amount(synthetic_df, tmp_path, monkeypatch):
    from src import config
    monkeypatch.setattr(config, "SCALER_PATH", str(tmp_path / "scaler.joblib"))

    X, y = split_features_target(synthetic_df)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # V columns should be untouched
    pd.testing.assert_frame_equal(X_train_scaled[["V1", "V2"]], X_train[["V1", "V2"]])
    # Time/Amount should be transformed (not identical to raw values)
    assert not X_train_scaled["Amount"].equals(X_train["Amount"])


def test_class_imbalance_is_severe(synthetic_df):
    """Sanity check that our synthetic data mimics the real dataset's imbalance direction."""
    fraud_ratio = synthetic_df["Class"].mean()
    assert fraud_ratio < 0.5  # fraud must be the minority class
