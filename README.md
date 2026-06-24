# 📉 Customer Churn Analysis & Retention Strategy

A complete, end-to-end machine learning project for predicting customer churn and designing data-driven retention strategies — using a telecom dataset as a case study.

---

## 🗂️ Project Structure

```
customer_churn_analysis/
├── data/
│   ├── raw/                   # Original dataset (Telco Customer Churn)
│   └── processed/             # Cleaned & feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Feature Engineering & Preprocessing
│   ├── 03_modeling.ipynb       # Model Training & Evaluation
│   └── 04_retention_strategy.ipynb  # Retention Insights & CLV
├── src/
│   ├── data_loader.py          # Data ingestion & cleaning
│   ├── features.py             # Feature engineering pipeline
│   ├── model.py                # Model training & evaluation
│   ├── retention.py            # Retention strategy logic
│   └── utils.py                # Helper functions
├── models/
│   └── churn_model.pkl         # Saved trained model
├── reports/
│   └── churn_analysis_report.md  # Summary report with findings
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── tests/
│   └── test_pipeline.py        # Unit tests
├── requirements.txt
├── config.yaml
└── README.md
```

---

## 🎯 Objectives

1. **Predict** which customers are likely to churn using ML models
2. **Understand** key drivers behind churn through feature importance & SHAP
3. **Segment** customers by churn risk and CLV (Customer Lifetime Value)
4. **Recommend** targeted retention strategies per customer segment

---

## 📊 Dataset

**Telco Customer Churn** (IBM Sample Dataset)
- ~7,000 customers
- 20 features: demographics, services subscribed, billing info
- Target: `Churn` (Yes/No)

> **Auto-download:** The data loader fetches it automatically via URL on first run.
> Alternatively, download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/raw/`.

---

## 🤖 Models Used

| Model | ROC-AUC | F1 Score |
|---|---|---|
| Logistic Regression | ~0.84 | ~0.60 |
| Random Forest | ~0.86 | ~0.63 |
| XGBoost | ~0.88 | ~0.65 |
| LightGBM *(best)* | ~0.89 | ~0.67 |

---

## 🔑 Key Features Driving Churn

- **Contract type** — Month-to-month contracts churn 3x more
- **Tenure** — Customers < 6 months have highest churn rate (~50%)
- **Internet Service** — Fiber optic users churn more than DSL
- **Monthly Charges** — Higher charges correlate with churn
- **Tech Support** — No tech support → higher churn

---

## 🛡️ Retention Strategy Framework

| Risk Segment | Churn Probability | Strategy |
|---|---|---|
| 🔴 High Risk | > 70% | Immediate outreach, discount offers, contract upgrade |
| 🟡 Medium Risk | 40–70% | Loyalty rewards, service bundling, proactive support |
| 🟢 Low Risk | < 40% | Engagement campaigns, upsell opportunities |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/customer_churn_analysis.git
cd customer_churn_analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Full Pipeline
```bash
python src/data_loader.py      # Download & clean data
python src/features.py         # Feature engineering
python src/model.py            # Train & evaluate models
python src/retention.py        # Generate retention recommendations
```

### 4. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

### 5. Run Notebooks (in order)
```
notebooks/01_eda.ipynb
notebooks/02_preprocessing.ipynb
notebooks/03_modeling.ipynb
notebooks/04_retention_strategy.ipynb
```

---

## 📈 Sample Outputs

**Churn Rate by Contract Type:**
```
Month-to-Month : 42.7% churn
One Year       :  9.3% churn
Two Year       :  2.8% churn
```

**Top 5 SHAP Features:**
```
1. Contract_Month-to-month  (+0.41)
2. tenure                   (-0.38)
3. MonthlyCharges            (+0.29)
4. InternetService_Fiber     (+0.22)
5. TechSupport_No            (+0.18)
```

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **pandas, numpy** — Data manipulation
- **scikit-learn** — ML pipeline & models
- **xgboost, lightgbm** — Gradient boosting
- **shap** — Model explainability
- **matplotlib, seaborn, plotly** — Visualization
- **streamlit** — Interactive dashboard
- **joblib** — Model persistence
- **pytest** — Unit testing

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

**Bharath**  
[GitHub](https://github.com/YOUR_USERNAME) | [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
