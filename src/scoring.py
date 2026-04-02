import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data
from feature_engineering import run_feature_engineering


def load_models():
    """
    Loads saved models and scaler from outputs/ folder.
    """
    output_dir = Path(__file__).resolve().parent.parent / "outputs"

    lr_model = pickle.load(open(output_dir / "logistic_regression.pkl", "rb"))
    rf_model = pickle.load(open(output_dir / "random_forest.pkl",        "rb"))
    scaler   = pickle.load(open(output_dir / "scaler.pkl",               "rb"))

    print("Models loaded successfully.")
    return lr_model, rf_model, scaler


def probability_to_score(prob_default):
    """
    Converts default probability (0-1) to a credit score (0-1000).
    Higher score = lower risk (like a real credit score).
    """
    score = int((1 - prob_default) * 1000)
    score = max(0, min(1000, score))
    return score


def classify_risk(score):
    """
    Classifies risk band based on credit score.
    """
    if score >= 700:
        return "Low Risk",    "Approve",      "#639922"
    elif score >= 500:
        return "Medium Risk", "Conditional",  "#EF9F27"
    else:
        return "High Risk",   "Decline",      "#E24B4A"


def score_applicant(applicant_data, scaler, model):
    """
    Scores a single applicant dict and returns full risk profile.
    """
    import pandas as pd
    df_input     = pd.DataFrame([applicant_data])
    X_scaled     = scaler.transform(df_input)
    prob_default = model.predict_proba(X_scaled)[0][1]

    score                    = probability_to_score(prob_default)
    risk_band, decision, _   = classify_risk(score)

    return {
        "probability_of_default": round(prob_default * 100, 1),
        "credit_score":           score,
        "risk_band":              risk_band,
        "decision":               decision,
    }


def score_dataframe(X, scaler, model):
    """
    Scores an entire DataFrame and returns results.
    """
    X_scaled     = scaler.transform(X)
    probs        = model.predict_proba(X_scaled)[:, 1]
    scores       = [probability_to_score(p) for p in probs]
    risk_bands   = [classify_risk(s) for s in scores]

    results = pd.DataFrame({
        "probability_of_default": (probs * 100).round(1),
        "credit_score":           scores,
        "risk_band":              [r[0] for r in risk_bands],
        "decision":               [r[1] for r in risk_bands],
    })

    return results


def print_applicant_report(result, applicant_id="Applicant"):
    """
    Prints a clean single applicant risk report.
    """
    band     = result["risk_band"]
    decision = result["decision"]
    score    = result["credit_score"]
    pd_pct   = result["probability_of_default"]

    border = "=" * 50
    print(f"\n{border}")
    print(f"  CREDIT RISK REPORT — {applicant_id}")
    print(f"{border}")
    print(f"  Credit Score         : {score} / 1000")
    print(f"  Probability of Default: {pd_pct}%")
    print(f"  Risk Band            : {band}")
    print(f"  Decision             : {decision}")
    print(f"{border}")

    if decision == "Approve":
        print("  Offer standard interest rate and full credit limit.")
    elif decision == "Conditional":
        print("  Offer with higher interest rate and reduced credit limit.")
        print("  Request additional income verification.")
    else:
        print("  Application declined.")
        print("  Suggest reapplication after 6 months with improved profile.")
    print(f"{border}\n")


def run_scoring():
    """
    Full scoring pipeline — scores all test applicants
    and shows sample individual reports.
    """
    output_dir = Path(__file__).resolve().parent.parent / "outputs"

    # Load data and models
    df = load_data()
    X, y = run_feature_engineering(df)
    lr_model, rf_model, scaler = load_models()

    # Score entire dataset with Random Forest (better precision)
    print("\nScoring all 1000 applicants...")
    results = score_dataframe(X, scaler, rf_model)
    results["actual_default"] = y.values

    # Distribution summary
    print("\n" + "="*50)
    print("  RISK BAND DISTRIBUTION")
    print("="*50)
    band_counts = results["risk_band"].value_counts()
    for band, count in band_counts.items():
        pct = count / len(results) * 100
        print(f"  {band:<14}: {count:>4} applicants ({pct:.1f}%)")

    print(f"\n  Average credit score : {results['credit_score'].mean():.0f}")
    print(f"  Average PD           : {results['probability_of_default'].mean():.1f}%")

    # Save full results
    save_path = output_dir / "scored_applicants.csv"
    results.to_csv(save_path, index=False)
    print(f"\n  Full results saved to: scored_applicants.csv")

    # Sample individual reports
    print("\n" + "="*50)
    print("  SAMPLE INDIVIDUAL REPORTS")
    print("="*50)

    # Pick 3 interesting applicants — one from each risk band
    for band, label in [("Low Risk", "Low Risk Sample"),
                        ("Medium Risk", "Medium Risk Sample"),
                        ("High Risk", "High Risk Sample")]:
        idx = results[results["risk_band"] == band].index[0]
        applicant_features = X.iloc[idx].to_dict()
        result = score_applicant(applicant_features, scaler, rf_model)
        print_applicant_report(result, label)

    return results


if __name__ == "__main__":
    results = run_scoring()
    print("Preview of scored results:")
    print(results.head(10).to_string(index=False))