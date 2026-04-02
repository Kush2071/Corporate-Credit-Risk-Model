import sys
import os
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_loader import load_data
from feature_engineering import run_feature_engineering
from model import run_model_training
from scoring import run_scoring, load_models, score_applicant, print_applicant_report



def print_banner():
    print("""
==================================================
    CREDIT RISK MODEL — FULL PIPELINE
    Corporate Credit Rating · 2,029 Companies
==================================================
    """)


def score_new_applicant(scaler, model, feature_names):
    """
    Scores a brand new hypothetical company.
    Automatically aligns features to match training order.
    """
    print("\n" + "="*50)
    print("  NEW COMPANY SCORING")
    print("="*50)

    # Example: A mid-size manufacturing company
    new_company_raw = {
        # Liquidity
        "currentRatio":                         1.8,
        "quickRatio":                           1.2,
        "cashRatio":                            0.4,
        "daysOfSalesOutstanding":               45.0,

        # Profitability
        "netProfitMargin":                      0.08,
        "pretaxProfitMargin":                   0.11,
        "grossProfitMargin":                    0.35,
        "operatingProfitMargin":                0.13,
        "returnOnAssets":                       0.07,
        "returnOnCapitalEmployed":              0.12,
        "returnOnEquity":                       0.15,

        # Efficiency
        "assetTurnover":                        0.85,
        "fixedAssetTurnover":                   1.40,

        # Leverage
        "debtEquityRatio":                      1.20,
        "debtRatio":                            0.45,
        "effectiveTaxRate":                     0.22,
        "companyEquityMultiplier":              2.20,

        # Cash flow
        "freeCashFlowOperatingCashFlowRatio":   0.65,
        "freeCashFlowPerShare":                 3.50,
        "cashPerShare":                         5.00,
        "operatingCashFlowPerShare":            5.40,
        "operatingCashFlowSalesRatio":          0.12,

        # Valuation
        "enterpriseValueMultiple":              10.5,
        "ebitPerRevenue":                       0.13,
        "payablesTurnover":                     6.50,

        # Sector
        "sector_code":                          7,
        "high_risk_sector":                     0,

        # Engineered features
        "liquidity_gap":                        1.8  - 1.2,
        "cash_coverage":                        0.4  * 1.8,
        "margin_gap":                           0.13 - 0.08,
        "profitability_score":                  (0.07*0.35 + 0.15*0.25 + 0.12*0.25 + 0.08*0.15),
        "earnings_quality":                     0.13,
        "debt_burden":                          (0.45*0.5 + 1.20*0.3 + 2.20*0.2),
        "high_leverage":                        0,
        "cashflow_strength":                    (0.65*0.5 + 0.12*0.5),
        "cash_vs_operations":                   5.00 / (5.40 + 0.01),
        "asset_efficiency":                     (0.85*0.5 + 1.40*0.3 + 0.07*0.2),
        "working_capital_efficiency":           6.50 / (45.0 + 1),
        "altman_z_proxy":                       (0.07*3.3 + 0.13*1.0 + 0.85*1.0 + (1/(0.45+0.01))*0.6),
        "altman_zone":                          1,
        "credit_health_score":                  0.75,
    }

    # ── Key fix: align columns to exact training order ────────────────────
    df_input = pd.DataFrame([new_company_raw])

    # Add any missing columns as 0
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0

    # Reorder to exactly match training feature order
    df_input = df_input[feature_names]

    result = score_applicant(df_input.iloc[0].to_dict(), scaler, model)
    print_applicant_report(result, "New Company — Mid-size Manufacturer")
    return result

def show_final_summary(results):
    """
    Prints the final project summary.
    """
    total    = len(results)
    approved = (results["decision"] == "Approve").sum()
    cond     = (results["decision"] == "Conditional").sum()
    declined = (results["decision"] == "Decline").sum()
    avg_score = results["credit_score"].mean()

    # Accuracy on known defaults
    correct = (
        ((results["decision"] == "Approve")   & (results["actual_default"] == 0)) |
        ((results["decision"] == "Conditional")& (results["actual_default"] == 0)) |
        ((results["decision"] == "Decline")    & (results["actual_default"] == 1))
    ).sum()

    print(f"""
==================================================
  FINAL PROJECT SUMMARY
==================================================
  Total applicants scored : {total}
  Approved                : {approved} ({approved/total*100:.1f}%)
  Conditional             : {cond} ({cond/total*100:.1f}%)
  Declined                : {declined} ({declined/total*100:.1f}%)
  Average credit score    : {avg_score:.0f} / 1000

  Correct decisions       : {correct} / {total} ({correct/total*100:.1f}%)

  Output files saved:
    outputs/scored_applicants.csv
    outputs/logistic_regression.pkl
    outputs/random_forest.pkl
    outputs/scaler.pkl
    outputs/01_target_distribution.png
    outputs/02_credit_amount_duration.png
    outputs/03_age_distribution.png
    outputs/04_default_by_checking.png
    outputs/05_correlation_heatmap.png
    outputs/logistic_regression_evaluation.png
    outputs/random_forest_evaluation.png
    outputs/feature_importances.png
==================================================
  PROJECT COMPLETE
==================================================
    """)


def main():
    print_banner()

    # Step 1 — Load data
    print("STEP 1 — Loading data...")
    df = load_data()

    # Step 2 — Feature engineering
    print("\nSTEP 2 — Feature engineering...")
    X, y = run_feature_engineering(df)

    # Step 3 — Train models
    print("\nSTEP 3 — Training models...")
    lr_model, rf_model, scaler, feature_names = run_model_training()

    # Step 4 — Score all applicants
    print("\nSTEP 4 — Scoring applicants...")
    results = run_scoring()

    # Step 5 — Score a brand new applicant
    print("\nSTEP 5 — Scoring a new applicant...")
    score_new_applicant(scaler, rf_model, feature_names)

    # Step 6 — Final summary
    show_final_summary(results)


if __name__ == "__main__":
    main()