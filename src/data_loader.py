"""
data_loader.py
--------------
Downloads, loads, and cleans the Telco Customer Churn dataset.
Saves processed raw data to data/raw/.
"""

import os
import requests
import pandas as pd
import numpy as np
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_data(url: str, save_path: str) -> None:
    """Download dataset from URL if not already present."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        logger.info(f"Data already exists at {save_path}. Skipping download.")
        return
    logger.info(f"Downloading dataset from {url}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)
    logger.info(f"Dataset saved to {save_path}")


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV into DataFrame."""
    df = pd.read_csv(path)
    logger.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns from {path}")
    return df


def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Clean the raw dataframe:
    - Fix TotalCharges dtype
    - Handle missing values
    - Encode binary target
    """
    df = df.copy()

    # Fix TotalCharges: convert spaces to NaN then to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_tc = df["TotalCharges"].isna().sum()
    if missing_tc > 0:
        logger.info(f"Filling {missing_tc} missing TotalCharges with tenure × MonthlyCharges")
        mask = df["TotalCharges"].isna()
        df.loc[mask, "TotalCharges"] = df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"]

    # Encode target: Yes → 1, No → 0
    target_col = config["data"]["target_column"]
    df[target_col] = df[target_col].map({"Yes": 1, "No": 0})
    logger.info(f"Churn distribution:\n{df[target_col].value_counts()}")

    # SeniorCitizen is already 0/1 — ensure int
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Strip whitespace in string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    logger.info(f"Cleaned data shape: {df.shape}")
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved cleaned data to {path}")


def get_basic_stats(df: pd.DataFrame) -> None:
    target = "Churn"
    total = len(df)
    churned = df[target].sum()
    churn_rate = churned / total * 100
    print(f"\n{'='*50}")
    print(f"  DATASET OVERVIEW")
    print(f"{'='*50}")
    print(f"  Total Customers   : {total:,}")
    print(f"  Churned           : {churned:,} ({churn_rate:.1f}%)")
    print(f"  Retained          : {total - churned:,} ({100-churn_rate:.1f}%)")
    print(f"  Features          : {df.shape[1] - 2} (excl. ID & target)")
    print(f"  Missing Values    : {df.isna().sum().sum()}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    config = load_config()
    download_data(config["data"]["download_url"], config["data"]["raw_path"])
    df = load_raw_data(config["data"]["raw_path"])
    df_clean = clean_data(df, config)
    get_basic_stats(df_clean)
    save_data(df_clean, config["data"]["raw_path"])  # overwrite with cleaned
    logger.info("Data loading & cleaning complete.")
