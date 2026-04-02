# Corporate Credit Risk Model

An end-to-end machine learning project that assesses the probability of default
for 2,029 real S&P-rated US companies across 12 sectors.

## Results
| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 68.7% | **82.0%** |
| Precision | 0.640 | **0.838** |
| Recall | 0.607 | **0.717** |
| F1 Score | 0.623 | **0.773** |
| ROC-AUC | 0.726 | **0.891** |

**91.7% correct lending decisions on real company data.**

## Project Structure
```
corporate-credit-risk-model/
├── data/               ← Place corporate_rating.csv here
├── src/
│   ├── data_loader.py          ← Loads & labels dataset
│   ├── eda.py                  ← Exploratory data analysis
│   ├── feature_engineering.py  ← 41 features + Altman Z-Score
│   ├── model.py                ← Train & evaluate models
│   ├── scoring.py              ← Risk score 0-1000 + classification
│   └── linkedin_visuals.py     ← Presentation charts
├── outputs/            ← Charts and model files saved here
├── requirements.txt
└── main.py             ← Run full pipeline
```

## Features Built
- **Liquidity ratios** — Current, Quick, Cash ratio
- **Profitability ratios** — ROA, ROE, ROCE, Net/Operating margins
- **Leverage ratios** — Debt/Equity, Debt ratio, Equity multiplier
- **Cash flow ratios** — FCF ratio, Operating CF ratio
- **Altman Z-Score** — Safe / Grey / Distress zone classification
- **Credit health composite score**

## Dataset
[Corporate Credit Rating Dataset](https://www.kaggle.com/datasets/agewerc/corporate-credit-rating)
— 2,029 records of real S&P rated US companies with 30 financial features.

## Tech Stack
- Python · pandas · numpy
- scikit-learn · Random Forest · Logistic Regression
- matplotlib · seaborn

## How to Run
```bash
pip install -r requirements.txt
# Place corporate_rating.csv in data/ folder
python main.py
```

## Risk Classification
| Score | Band | Decision |
|---|---|---|
| 700–1000 | Low Risk | Approve |
| 500–699 | Medium Risk | Conditional |
| 0–499 | High Risk | Decline |