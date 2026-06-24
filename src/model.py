"""
model.py
--------
Model training, cross-validation, evaluation, and SHAP explainability.
Trains Logistic Regression, Random Forest, XGBoost, LightGBM.
Saves best model (LightGBM) to models/.
"""

import os
import joblib
import yaml
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_models(config: dict) -> dict:
    lgb_p = config["model"]["lightgbm_params"]
    xgb_p = config["model"]["xgboost_params"]

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=config["model"]["random_state"]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=config["model"]["random_state"], n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            **xgb_p,
            use_label_encoder=False, eval_metric="logloss",
            random_state=config["model"]["random_state"]
        ),
        "LightGBM": lgb.LGBMClassifier(
            **lgb_p, random_state=config["model"]["random_state"],
            verbose=-1
        ),
    }


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "model": model_name,
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
    }
    logger.info(f"[{model_name}] AUC={results['roc_auc']} | F1={results['f1']} | "
                f"Precision={results['precision']} | Recall={results['recall']}")
    return results


def cross_validate_model(model, X_train, y_train, config: dict) -> float:
    cv = StratifiedKFold(n_splits=config["model"]["cv_folds"], shuffle=True,
                         random_state=config["model"]["random_state"])
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    logger.info(f"CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()


def plot_results(results: list, X_test, y_test, models: dict, save_dir="reports") -> None:
    os.makedirs(save_dir, exist_ok=True)

    # ── 1. Model Comparison Bar Chart ──────────────────────────────────────
    df_results = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics = ["roc_auc", "f1", "precision", "recall"]
    x = np.arange(len(df_results))
    width = 0.2
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, df_results[metric], width, label=metric.upper(), color=color, alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df_results["model"], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/model_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved model_comparison.png")

    # ── 2. ROC Curves ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curves.png", dpi=150)
    plt.close()
    logger.info("Saved roc_curves.png")

    # ── 3. Confusion Matrix (best model — LightGBM) ──────────────────────
    best_model = models["LightGBM"]
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — LightGBM", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=150)
    plt.close()
    logger.info("Saved confusion_matrix.png")


def plot_shap(model, X_test, feature_names, save_dir="reports") -> None:
    if not SHAP_AVAILABLE:
        logger.warning("shap not installed — skipping SHAP plots.")
        return

    os.makedirs(save_dir, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    X_sample = X_test.iloc[:300]  # use sample for speed
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved shap_summary.png")

    # Feature importance bar
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved shap_importance.png")


def save_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str):
    return joblib.load(path)


def print_summary(results: list) -> None:
    print(f"\n{'='*65}")
    print(f"  {'MODEL':<25} {'AUC':>7} {'F1':>7} {'PREC':>8} {'RECALL':>8}")
    print(f"{'='*65}")
    for r in sorted(results, key=lambda x: x["roc_auc"], reverse=True):
        print(f"  {r['model']:<25} {r['roc_auc']:>7.4f} {r['f1']:>7.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from data_loader import load_raw_data, clean_data
    from features import prepare_features

    config = load_config()
    df = load_raw_data(config["data"]["raw_path"])
    df_clean = clean_data(df, config)
    X_train, X_test, y_train, y_test, feature_names = prepare_features(df_clean, config)

    models = build_models(config)
    results = []

    for name, model in models.items():
        logger.info(f"\nTraining: {name}")
        model.fit(X_train, y_train)
        cv_auc = cross_validate_model(model, X_train, y_train, config)
        eval_res = evaluate_model(model, X_test, y_test, name)
        eval_res["cv_auc"] = round(cv_auc, 4)
        results.append(eval_res)

    print_summary(results)

    # Save best model (LightGBM)
    best_model = models["LightGBM"]
    save_model(best_model, config["model"]["model_path"])

    # Generate plots
    plot_results(results, X_test, y_test, models)
    plot_shap(best_model, X_test, feature_names)

    logger.info("Training pipeline complete.")
