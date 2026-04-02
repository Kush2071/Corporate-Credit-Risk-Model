import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data


def handle_missing_values(df):
    """
    Fills missing values with column medians.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)
    return df


def encode_categorical(df):
    """
    Encodes sector into numeric values.
    """
    sector_map = {
        "Consumer Durables":          1,
        "Consumer Non-Durables":      2,
        "Consumer Services":          3,
        "Energy":                     4,
        "Finance":                    5,
        "Health Care":                6,
        "Capital Goods":              7,
        "Technology":                 8,
        "Transportation":             9,
        "Utilities":                  10,
        "Basic Industries":           11,
        "Public Utilities":           12,
    }
    df["sector_code"] = df["Sector"].map(sector_map).fillna(0)

    # Sector risk flag — energy, finance, utilities tend to be higher risk
    high_risk_sectors = ["Energy", "Finance", "Basic Industries"]
    df["high_risk_sector"] = df["Sector"].apply(
        lambda x: 1 if x in high_risk_sectors else 0
    )

    return df


def create_financial_ratios(df):
    """
    Creates new financial features from existing columns.
    These mirror what a real credit analyst computes.
    """

    # ── Liquidity Ratios ──────────────────────────────────────────────────
    # How easily can the company pay short term debts?

    # Liquidity gap — current vs quick ratio (large gap = lots of inventory risk)
    df["liquidity_gap"] = df["currentRatio"] - df["quickRatio"]

    # Cash coverage strength
    df["cash_coverage"] = df["cashRatio"] * df["currentRatio"]

    # ── Profitability Ratios ──────────────────────────────────────────────
    # How efficiently is the company generating profit?

    # Profit quality — operating vs net margin gap
    df["margin_gap"] = df["operatingProfitMargin"] - df["netProfitMargin"]

    # Overall profitability score
    df["profitability_score"] = (
        df["returnOnAssets"]          * 0.35 +
        df["returnOnEquity"]          * 0.25 +
        df["returnOnCapitalEmployed"] * 0.25 +
        df["netProfitMargin"]         * 0.15
    )

    # Earnings quality — EBIT as % of revenue
    df["earnings_quality"] = df["ebitPerRevenue"]

    # ── Leverage / Debt Ratios ────────────────────────────────────────────
    # How much debt is the company carrying?

    # Debt burden score — higher = more leveraged = riskier
    df["debt_burden"] = (
        df["debtRatio"]        * 0.5 +
        df["debtEquityRatio"]  * 0.3 +
        df["companyEquityMultiplier"] * 0.2
    )

    # Leverage risk flag
    df["high_leverage"] = (df["debtEquityRatio"] > 2.0).astype(int)

    # ── Cash Flow Ratios ──────────────────────────────────────────────────
    # Is the company generating real cash?

    # Cash flow strength
    df["cashflow_strength"] = (
        df["freeCashFlowOperatingCashFlowRatio"] * 0.5 +
        df["operatingCashFlowSalesRatio"]        * 0.5
    )

    # Cash per share vs operating cash flow per share
    df["cash_vs_operations"] = (
        df["cashPerShare"] / (df["operatingCashFlowPerShare"].abs() + 0.01)
    )

    # ── Efficiency Ratios ─────────────────────────────────────────────────
    # How well is the company using its assets?

    # Asset efficiency score
    df["asset_efficiency"] = (
        df["assetTurnover"]      * 0.5 +
        df["fixedAssetTurnover"] * 0.3 +
        df["returnOnAssets"]     * 0.2
    )

    # Working capital efficiency
    df["working_capital_efficiency"] = (
        df["payablesTurnover"] / (df["daysOfSalesOutstanding"] + 1)
    )

    # ── Altman Z-Score Proxy ──────────────────────────────────────────────
    # Classic corporate bankruptcy predictor (simplified version)
    # Z > 2.99 = safe zone, 1.81-2.99 = grey zone, < 1.81 = distress zone
    df["altman_z_proxy"] = (
        df["returnOnAssets"]       * 3.3  +
        df["ebitPerRevenue"]       * 1.0  +
        df["assetTurnover"]        * 1.0  +
        (1 / (df["debtRatio"] + 0.01)) * 0.6
    )

    # Altman zone classification
    df["altman_zone"] = pd.cut(
        df["altman_z_proxy"],
        bins=[-np.inf, 1.81, 2.99, np.inf],
        labels=[2, 1, 0]  # 2=distress, 1=grey, 0=safe
    ).astype(int)

    # ── Overall Credit Health Score ───────────────────────────────────────
    df["credit_health_score"] = (
        df["profitability_score"]  * 0.30 +
        df["cashflow_strength"]    * 0.25 +
        df["asset_efficiency"]     * 0.20 +
        (1 / (df["debt_burden"] + 0.01)) * 0.15 +
        df["currentRatio"]         * 0.10
    )

    return df


def select_features(df):
    """
    Selects final features for the model.
    Returns X (features) and y (target).
    """

    feature_cols = [
        # Original financial ratios
        "currentRatio", "quickRatio", "cashRatio",
        "daysOfSalesOutstanding",
        "netProfitMargin", "pretaxProfitMargin",
        "grossProfitMargin", "operatingProfitMargin",
        "returnOnAssets", "returnOnCapitalEmployed", "returnOnEquity",
        "assetTurnover", "fixedAssetTurnover",
        "debtEquityRatio", "debtRatio",
        "effectiveTaxRate",
        "freeCashFlowOperatingCashFlowRatio",
        "freeCashFlowPerShare", "cashPerShare",
        "companyEquityMultiplier", "ebitPerRevenue",
        "enterpriseValueMultiple",
        "operatingCashFlowPerShare", "operatingCashFlowSalesRatio",
        "payablesTurnover",

        # Encoded categorical
        "sector_code", "high_risk_sector",

        # Engineered features
        "liquidity_gap", "cash_coverage",
        "margin_gap", "profitability_score", "earnings_quality",
        "debt_burden", "high_leverage",
        "cashflow_strength", "cash_vs_operations",
        "asset_efficiency", "working_capital_efficiency",
        "altman_z_proxy", "altman_zone",
        "credit_health_score",
    ]

    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0)
    y = df["target"]

    return X, y


def run_feature_engineering(df):
    """
    Full pipeline — clean, encode, engineer, select.
    """
    print("=" * 50)
    print("FEATURE ENGINEERING")
    print("=" * 50)

    print("Step 1: Handling missing values...")
    df = handle_missing_values(df)

    print("Step 2: Encoding sectors...")
    df = encode_categorical(df)

    print("Step 3: Creating financial ratios...")
    df = create_financial_ratios(df)

    print("Step 4: Selecting features...")
    X, y = select_features(df)

    print(f"\nFinal feature matrix : {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"Target distribution  : {y.value_counts().to_dict()}")

    print("\nSample engineered features (first 3 rows):")
    sample_cols = ["profitability_score", "debt_burden",
                   "altman_z_proxy", "credit_health_score"]
    print(X[sample_cols].head(3).round(3).to_string(index=False))

    # Altman zone breakdown
    print(f"\nAltman Z-Score zones:")
    print(f"  Safe zone (Z > 2.99)     : {(df['altman_zone']==0).sum()} companies")
    print(f"  Grey zone (1.81–2.99)    : {(df['altman_zone']==1).sum()} companies")
    print(f"  Distress zone (Z < 1.81) : {(df['altman_zone']==2).sum()} companies")

    print("\nFeature engineering complete!")
    return X, y


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        X, y = run_feature_engineering(df)