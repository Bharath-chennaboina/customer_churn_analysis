"""
tests/test_pipeline.py
-----------------------
Unit tests for the churn analysis pipeline.
Run with: pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal synthetic dataframe matching Telco schema."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "customerID": [f"C{i:04d}" for i in range(n)],
        "gender": np.random.choice(["Male", "Female"], n),
        "SeniorCitizen": np.random.choice([0, 1], n),
        "Partner": np.random.choice(["Yes", "No"], n),
        "Dependents": np.random.choice(["Yes", "No"], n),
        "tenure": np.random.randint(0, 72, n),
        "PhoneService": np.random.choice(["Yes", "No"], n),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n),
        "PaymentMethod": np.random.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], n),
        "MonthlyCharges": np.random.uniform(18, 120, n).round(2),
        "TotalCharges": (np.random.uniform(18, 120, n) * np.random.randint(1, 72, n)).astype(str),
        "Churn": np.random.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })


@pytest.fixture
def config():
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─── Data Loading Tests ───────────────────────────────────────────────────────

class TestDataLoader:
    def test_clean_data_target_encoded(self, sample_df, config):
        from data_loader import clean_data
        df = clean_data(sample_df, config)
        assert set(df["Churn"].unique()).issubset({0, 1})

    def test_clean_data_total_charges_numeric(self, sample_df, config):
        from data_loader import clean_data
        df = clean_data(sample_df, config)
        assert pd.api.types.is_float_dtype(df["TotalCharges"])

    def test_clean_data_no_missing_target(self, sample_df, config):
        from data_loader import clean_data
        df = clean_data(sample_df, config)
        assert df["Churn"].isna().sum() == 0

    def test_clean_data_shape_preserved(self, sample_df, config):
        from data_loader import clean_data
        df = clean_data(sample_df, config)
        assert df.shape[0] == len(sample_df)


# ─── Feature Engineering Tests ────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_num_services_column_exists(self, sample_df, config):
        from data_loader import clean_data
        from features import engineer_features
        df = clean_data(sample_df, config)
        df_fe = engineer_features(df)
        assert "num_services" in df_fe.columns

    def test_num_services_non_negative(self, sample_df, config):
        from data_loader import clean_data
        from features import engineer_features
        df = clean_data(sample_df, config)
        df_fe = engineer_features(df)
        assert (df_fe["num_services"] >= 0).all()

    def test_tenure_group_column_exists(self, sample_df, config):
        from data_loader import clean_data
        from features import engineer_features
        df = clean_data(sample_df, config)
        df_fe = engineer_features(df)
        assert "tenure_group" in df_fe.columns

    def test_has_streaming_binary(self, sample_df, config):
        from data_loader import clean_data
        from features import engineer_features
        df = clean_data(sample_df, config)
        df_fe = engineer_features(df)
        assert set(df_fe["has_streaming"].unique()).issubset({0, 1})

    def test_prepare_features_train_test_shapes(self, sample_df, config):
        from data_loader import clean_data
        from features import prepare_features
        df = clean_data(sample_df, config)
        X_train, X_test, y_train, y_test, features = prepare_features(df, config)
        assert len(X_train) + len(X_test) == len(df)
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == len(X_train)


# ─── Retention Tests ──────────────────────────────────────────────────────────

class TestRetention:
    def test_segment_tier_labels(self, sample_df, config):
        from data_loader import clean_data
        from retention import segment_customers
        df = clean_data(sample_df, config)
        df["churn_prob"] = np.random.uniform(0, 1, len(df))
        df = segment_customers(df, config)
        valid_tiers = {"🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"}
        assert set(df["risk_tier"].unique()).issubset(valid_tiers)

    def test_clv_non_negative(self, sample_df, config):
        from data_loader import clean_data
        from retention import compute_clv
        df = clean_data(sample_df, config)
        df["churn_prob"] = np.random.uniform(0, 1, len(df))
        clv = compute_clv(df, config)
        assert (clv >= 0).all()
