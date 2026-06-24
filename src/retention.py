"""
retention.py
------------
Generates churn probability scores for all customers,
segments them by risk tier, computes CLV, and produces
targeted retention strategy recommendations.
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_clv(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Simplified CLV = Monthly Margin × (1 / discount_rate) × (1 - churn_prob)
    Monthly Margin = MonthlyCharges × margin_rate
    """
    margin = config["retention"]["clv_monthly_margin"]
    discount_rate = config["retention"]["clv_discount_rate"] / 12  # monthly rate
    monthly_margin = df["MonthlyCharges"] * margin
    # Gordon Growth / perpetuity model: CLV = margin / discount_rate
    # Adjusted by churn probability
    clv = (monthly_margin / discount_rate) * (1 - df["churn_prob"])
    return clv.round(2)


def segment_customers(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Assign risk tier and retention strategy to each customer."""
    high_thresh = config["retention"]["high_risk_threshold"]
    med_thresh = config["retention"]["medium_risk_threshold"]

    def assign_tier(p):
        if p >= high_thresh:
            return "🔴 High Risk"
        elif p >= med_thresh:
            return "🟡 Medium Risk"
        else:
            return "🟢 Low Risk"

    strategies = {
        "🔴 High Risk": (
            "Immediate personal outreach via phone/email. "
            "Offer contract upgrade incentive (15–20% discount for 1-year lock-in). "
            "Assign dedicated account manager. Escalate unresolved tech issues."
        ),
        "🟡 Medium Risk": (
            "Proactive check-in within 7 days. "
            "Offer loyalty rewards or service bundle upgrades. "
            "Invite to beta features or premium support trial. "
            "Send personalized usage & savings report."
        ),
        "🟢 Low Risk": (
            "Routine engagement via newsletter/email. "
            "Upsell complementary services (streaming, security add-ons). "
            "Reward referrals and long-tenure milestones. "
            "Gather NPS feedback to maintain satisfaction."
        ),
    }

    df["risk_tier"] = df["churn_prob"].apply(assign_tier)
    df["retention_strategy"] = df["risk_tier"].map(strategies)
    return df


def generate_retention_report(df: pd.DataFrame, save_dir="reports") -> None:
    os.makedirs(save_dir, exist_ok=True)

    # ── Tier summary ───────────────────────────────────────────────────
    tier_summary = df.groupby("risk_tier").agg(
        num_customers=("churn_prob", "count"),
        avg_churn_prob=("churn_prob", "mean"),
        avg_monthly_charges=("MonthlyCharges", "mean"),
        avg_clv=("clv", "mean"),
        total_clv=("clv", "sum"),
    ).round(2)

    print("\n" + "=" * 70)
    print("  CUSTOMER RETENTION STRATEGY — RISK TIER SUMMARY")
    print("=" * 70)
    print(tier_summary.to_string())
    print("=" * 70 + "\n")

    # ── Churn probability distribution plot ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution
    axes[0].hist(df["churn_prob"], bins=40, color="#E53935", edgecolor="white", alpha=0.85)
    axes[0].axvline(0.40, color="#FB8C00", linestyle="--", lw=2, label="Medium threshold (0.40)")
    axes[0].axvline(0.70, color="#B71C1C", linestyle="--", lw=2, label="High threshold (0.70)")
    axes[0].set_xlabel("Churn Probability")
    axes[0].set_ylabel("Number of Customers")
    axes[0].set_title("Churn Probability Distribution", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Tier pie
    tier_counts = df["risk_tier"].value_counts()
    colors_pie = ["#E53935", "#FB8C00", "#43A047"]
    axes[1].pie(
        tier_counts.values,
        labels=tier_counts.index,
        colors=colors_pie,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    axes[1].set_title("Customer Risk Tier Distribution", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/retention_overview.png", dpi=150)
    plt.close()
    logger.info("Saved retention_overview.png")

    # ── CLV vs Churn Probability scatter ──────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    tier_color_map = {
        "🔴 High Risk": "#E53935",
        "🟡 Medium Risk": "#FB8C00",
        "🟢 Low Risk": "#43A047"
    }
    for tier, group in df.groupby("risk_tier"):
        ax.scatter(
            group["churn_prob"], group["clv"],
            c=tier_color_map[tier], alpha=0.5, s=20, label=tier
        )
    ax.set_xlabel("Churn Probability")
    ax.set_ylabel("Customer Lifetime Value (₹)")
    ax.set_title("CLV vs Churn Probability by Risk Tier", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/clv_vs_churn.png", dpi=150)
    plt.close()
    logger.info("Saved clv_vs_churn.png")

    # ── Save segmented customer list ──────────────────────────────────
    output_path = f"{save_dir}/customer_segments.csv"
    df[["churn_prob", "risk_tier", "clv", "retention_strategy",
        "MonthlyCharges", "tenure", "Contract"]].to_csv(output_path)
    logger.info(f"Saved customer segments to {output_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from data_loader import load_raw_data, clean_data
    from features import engineer_features, encode_features, prepare_features

    config = load_config()

    # Load and preprocess data
    df_raw = load_raw_data(config["data"]["raw_path"])
    df_clean = clean_data(df_raw, config)
    X_train, X_test, y_train, y_test, feature_names = prepare_features(df_clean, config)

    # Load trained model
    model = joblib.load(config["model"]["model_path"])

    # Get predictions on full dataset
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    df_fe = engineer_features(df_clean)
    df_enc = encode_features(df_fe, config)
    drop_cols = config["features"]["drop_cols"] + [config["data"]["target_column"]]
    X_full = df_enc.drop(columns=drop_cols, errors="ignore")

    # Align columns to training feature order
    X_full = X_full.reindex(columns=feature_names, fill_value=0)

    probs = model.predict_proba(X_full)[:, 1]

    # Build results dataframe
    df_result = df_clean.copy()
    df_result["churn_prob"] = probs
    df_result["clv"] = compute_clv(df_result, config)
    df_result = segment_customers(df_result, config)

    generate_retention_report(df_result)
    logger.info("Retention strategy generation complete.")
