"""
utils.py
--------
Shared utility functions for the churn analysis project.
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_plot_style():
    """Apply consistent dark-style plotting theme."""
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#0f3460",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "grid.color": "#0f3460",
        "grid.alpha": 0.5,
        "font.family": "monospace",
    })


def plot_churn_by_feature(df: pd.DataFrame, feature: str, save_dir: str = "reports") -> None:
    """Bar chart: churn rate broken down by a categorical feature."""
    os.makedirs(save_dir, exist_ok=True)
    churn_rate = df.groupby(feature)["Churn"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(churn_rate.index, churn_rate.values * 100,
                  color=sns.color_palette("Reds_r", len(churn_rate)), edgecolor="white")

    for bar, val in zip(bars, churn_rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val*100:.1f}%",
                ha="center", va="bottom", fontsize=9)

    ax.set_title(f"Churn Rate by {feature}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel(feature)
    plt.xticks(rotation=20)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/churn_by_{feature}.png", dpi=150)
    plt.close()


def plot_numerical_distribution(df: pd.DataFrame, feature: str, save_dir: str = "reports") -> None:
    """KDE plot of a numerical feature split by churn."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, color, name in [(0, "#4CAF50", "No Churn"), (1, "#E53935", "Churn")]:
        subset = df[df["Churn"] == label][feature].dropna()
        subset.plot.kde(ax=ax, color=color, lw=2, label=name)

    ax.set_title(f"{feature} Distribution by Churn", fontsize=13, fontweight="bold")
    ax.set_xlabel(feature)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{feature}_distribution.png", dpi=150)
    plt.close()


def correlation_heatmap(df: pd.DataFrame, save_dir: str = "reports") -> None:
    """Correlation heatmap of numeric columns vs churn."""
    os.makedirs(save_dir, exist_ok=True)
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()[["Churn"]].drop("Churn").sort_values("Churn", ascending=False)

    fig, ax = plt.subplots(figsize=(5, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn_r",
                center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation with Churn", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/correlation_heatmap.png", dpi=150)
    plt.close()
    logger.info("Saved correlation_heatmap.png")


def save_metrics_csv(results: list, save_path: str = "reports/model_metrics.csv") -> None:
    """Save model evaluation metrics to CSV."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(results).to_csv(save_path, index=False)
    logger.info(f"Saved metrics to {save_path}")


def print_banner(text: str, width: int = 60) -> None:
    border = "=" * width
    print(f"\n{border}")
    print(f"  {text}")
    print(f"{border}\n")
