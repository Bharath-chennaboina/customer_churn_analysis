"""
features.py
-----------
Feature engineering and preprocessing pipeline for churn prediction.
Applies encoding, scaling, and derives new features.
"""

import pandas as pd
import numpy as np
import yaml
import logging
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive new features from existing columns."""
    df = df.copy()

    # Tenure groups
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 48, 72],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"]
    )

    # Charges per month vs average (deviation from median)
    median_charge = df["MonthlyCharges"].median()
    df["charges_above_median"] = (df["MonthlyCharges"] > median_charge).astype(int)

    # Number of services subscribed
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v in ["Yes", 1, "1"]), axis=1
    )

    # Avg charge per service
    df["charge_per_service"] = df.apply(
        lambda r: r["MonthlyCharges"] / r["num_services"] if r["num_services"] > 0 else r["MonthlyCharges"],
        axis=1
    )

    # Has any streaming service
    df["has_streaming"] = (
        (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
    ).astype(int)

    # Has security services
    df["has_security"] = (
        (df["OnlineSecurity"] == "Yes") | (df["OnlineBackup"] == "Yes") |
        (df["DeviceProtection"] == "Yes") | (df["TechSupport"] == "Yes")
    ).astype(int)

    # Is paperless & auto-pay (engagement proxy)
    auto_pay_methods = ["Bank transfer (automatic)", "Credit card (automatic)"]
    df["is_autopay"] = df["PaymentMethod"].isin(auto_pay_methods).astype(int)

    logger.info(f"Feature engineering done. Shape: {df.shape}")
    return df


def encode_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Encode categorical features using Label Encoding."""
    df = df.copy()
    cat_cols = config["features"]["categorical_cols"] + ["tenure_group"]

    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df


def prepare_features(df: pd.DataFrame, config: dict):
    """
    Full preprocessing pipeline:
    1. Engineer features
    2. Encode categoricals
    3. Drop unused columns
    4. Scale numericals
    5. Split into train/test

    Returns X_train, X_test, y_train, y_test, feature_names
    """
    df = engineer_features(df)
    df = encode_features(df, config)

    target = config["data"]["target_column"]
    drop_cols = config["features"]["drop_cols"]

    # Drop customer ID and target
    feature_df = df.drop(columns=drop_cols + [target], errors="ignore")
    y = df[target]
    X = feature_df

    feature_names = list(X.columns)
    logger.info(f"Final feature set: {len(feature_names)} features")

    # Scale numerical columns
    num_cols = config["features"]["numerical_cols"] + [
        "num_services", "charge_per_service", "charges_above_median"
    ]
    num_cols = [c for c in num_cols if c in X.columns]

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y
    )

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Train churn rate: {y_train.mean():.2%} | Test churn rate: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    from data_loader import load_raw_data, clean_data, load_config as dl_config

    config = load_config()
    df = load_raw_data(config["data"]["raw_path"])
    df_clean = clean_data(df, config)
    X_train, X_test, y_train, y_test, features = prepare_features(df_clean, config)

    os.makedirs("data/processed", exist_ok=True)
    processed = pd.concat([X_train, y_train], axis=1)
    processed.to_csv(config["data"]["processed_path"], index=False)
    logger.info(f"Saved processed data to {config['data']['processed_path']}")
