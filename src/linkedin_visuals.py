import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data
from feature_engineering import run_feature_engineering

def set_style():
    plt.rcParams.update({
        "figure.facecolor":   "#FFFFFF",
        "axes.facecolor":     "#F8F9FA",
        "axes.grid":          True,
        "grid.color":         "#E0E0E0",
        "grid.linewidth":     0.8,
        "font.family":        "DejaVu Sans",
        "axes.titlesize":     14,
        "axes.titleweight":   "bold",
        "axes.labelsize":     11,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "figure.dpi":         180,
    })

def chart1_model_comparison(output_dir):
    """
    Bar chart comparing both models across all metrics.
    """
    set_style()
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    lr_vals  = [0.687, 0.640, 0.607, 0.623, 0.726]
    rf_vals  = [0.820, 0.838, 0.717, 0.773, 0.891]

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, lr_vals, width, label="Logistic Regression",
                   color="#378ADD", alpha=0.85, zorder=3)
    bars2 = ax.bar(x + width/2, rf_vals, width, label="Random Forest",
                   color="#1D9E75", alpha=0.85, zorder=3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=9, color="#0C447C", fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=9, color="#085041", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison\nLogistic Regression vs Random Forest")
    ax.legend(loc="lower right")
    ax.axhline(0.8, color="#E24B4A", linestyle="--",
               linewidth=1.2, alpha=0.6, label="0.80 threshold")

    plt.tight_layout()
    path = output_dir / "linkedin_01_model_comparison.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path.name}")


def chart2_risk_distribution(output_dir):
    """
    Donut chart of risk band distribution.
    """
    set_style()
    results = pd.read_csv(
        Path(__file__).resolve().parent.parent / "outputs" / "scored_applicants.csv"
    )

    band_counts = results["risk_band"].value_counts()
    labels  = band_counts.index.tolist()
    sizes   = band_counts.values.tolist()
    colors  = []
    for l in labels:
        if l == "Low Risk":    colors.append("#639922")
        elif l == "Medium Risk": colors.append("#EF9F27")
        else:                  colors.append("#E24B4A")

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors,
        autopct="%1.1f%%", startangle=90,
        pctdistance=0.78, wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 3}
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
        at.set_color("white")

    legend_labels = [f"{l}  ({c} companies)" for l, c in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), fontsize=11, frameon=False)

    ax.text(0, 0, f"{len(results)}\nCompanies", ha="center", va="center",
            fontsize=16, fontweight="bold", color="#2C2C2A")
    ax.set_title("Corporate Credit Risk Distribution\nS&P Rated US Companies",
                 fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    path = output_dir / "linkedin_02_risk_distribution.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path.name}")


def chart3_altman_zones(output_dir):
    """
    Altman Z-Score zone breakdown by actual rating.
    """
    set_style()
    df = load_data()
    df, _ = run_feature_engineering_internal(df)

    zone_map   = {0: "Safe Zone", 1: "Grey Zone", 2: "Distress Zone"}
    df["zone"] = df["altman_zone"].map(zone_map)

    rating_order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
    existing     = [r for r in rating_order if r in df["Rating"].values]

    zone_counts = df.groupby(["Rating", "zone"]).size().unstack(fill_value=0)
    zone_counts = zone_counts.reindex(existing)

    colors = {"Safe Zone": "#639922", "Grey Zone": "#EF9F27", "Distress Zone": "#E24B4A"}
    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(len(zone_counts))
    for zone in ["Safe Zone", "Grey Zone", "Distress Zone"]:
        if zone in zone_counts.columns:
            vals = zone_counts[zone].values
            ax.bar(zone_counts.index, vals, bottom=bottom,
                   label=zone, color=colors[zone], alpha=0.85,
                   edgecolor="white", linewidth=0.8, zorder=3)
            bottom += vals

    ax.set_xlabel("S&P Credit Rating")
    ax.set_ylabel("Number of Companies")
    ax.set_title("Altman Z-Score Zones by Credit Rating\nSafe · Grey · Distress Zone Breakdown")
    ax.legend(loc="upper right")
    ax.axvline(3.5, color="#2C2C2A", linestyle="--",
               linewidth=1.5, alpha=0.5)
    ax.text(1.5, ax.get_ylim()[1]*0.92, "Investment Grade",
            ha="center", fontsize=10, color="#085041", fontweight="bold")
    ax.text(5.5, ax.get_ylim()[1]*0.92, "Speculative Grade",
            ha="center", fontsize=10, color="#A32D2D", fontweight="bold")

    plt.tight_layout()
    path = output_dir / "linkedin_03_altman_zones.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path.name}")


def chart4_feature_importance(output_dir):
    """
    Top 15 features — clean horizontal bar chart.
    """
    set_style()
    import pickle
    models_dir = Path(__file__).resolve().parent.parent / "outputs"
    rf_model   = pickle.load(open(models_dir / "random_forest.pkl", "rb"))

    df       = load_data()
    X, y     = run_feature_engineering(df)
    features = X.columns.tolist()

    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1][:15]
    top_features = [features[i] for i in indices]
    top_values   = importances[indices]

    # Clean up feature names for display
    name_map = {
        "credit_health_score":                  "Credit Health Score",
        "altman_z_proxy":                       "Altman Z-Score",
        "profitability_score":                  "Profitability Score",
        "debt_burden":                          "Debt Burden",
        "returnOnEquity":                       "Return on Equity",
        "returnOnAssets":                       "Return on Assets",
        "ebitPerRevenue":                       "EBIT / Revenue",
        "debtEquityRatio":                      "Debt / Equity Ratio",
        "netProfitMargin":                      "Net Profit Margin",
        "operatingProfitMargin":                "Operating Profit Margin",
        "cashflow_strength":                    "Cash Flow Strength",
        "asset_efficiency":                     "Asset Efficiency",
        "currentRatio":                         "Current Ratio",
        "grossProfitMargin":                    "Gross Profit Margin",
        "debtRatio":                            "Debt Ratio",
        "returnOnCapitalEmployed":              "Return on Capital Employed",
        "freeCashFlowOperatingCashFlowRatio":   "FCF / Operating CF",
        "companyEquityMultiplier":              "Equity Multiplier",
        "enterpriseValueMultiple":              "EV Multiple",
        "working_capital_efficiency":           "Working Capital Efficiency",
    }
    display_names = [name_map.get(f, f) for f in top_features]

    colors = ["#185FA5" if v > np.median(top_values) else "#85B7EB"
              for v in top_values]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(display_names[::-1], top_values[::-1],
                   color=colors[::-1], alpha=0.9, zorder=3,
                   edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, top_values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, color="#0C447C")

    ax.set_xlabel("Importance Score")
    ax.set_title("Top 15 Most Important Features\nRandom Forest — Corporate Credit Risk Model")
    ax.set_xlim(0, max(top_values) * 1.18)

    plt.tight_layout()
    path = output_dir / "linkedin_04_feature_importance.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path.name}")


def chart5_sector_default_rates(output_dir):
    """
    Default rate by sector — horizontal bar chart.
    """
    set_style()
    df = load_data()

    sector_stats = df.groupby("Sector")["target"].agg(["mean", "count"]).reset_index()
    sector_stats.columns = ["Sector", "default_rate", "count"]
    sector_stats["default_rate"] *= 100
    sector_stats = sector_stats.sort_values("default_rate", ascending=True)

    colors = ["#E24B4A" if r > 50 else "#EF9F27" if r > 35 else "#639922"
              for r in sector_stats["default_rate"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(sector_stats["Sector"], sector_stats["default_rate"],
                   color=colors, alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=0.5)

    for bar, (_, row) in zip(bars, sector_stats.iterrows()):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2,
                f"{row['default_rate']:.1f}%  (n={int(row['count'])})",
                va="center", fontsize=9)

    ax.axvline(42.6, color="#2C2C2A", linestyle="--",
               linewidth=1.2, alpha=0.6)
    ax.text(43.5, -0.6, "Avg 42.6%", fontsize=9,
            color="#2C2C2A", alpha=0.8)

    ax.set_xlabel("Speculative Grade Rate (%)")
    ax.set_title("Speculative Grade Rate by Sector\nWhich Industries Carry the Most Credit Risk?")
    ax.set_xlim(0, 100)

    low   = mpatches.Patch(color="#639922", label="Low risk sector (< 35%)")
    med   = mpatches.Patch(color="#EF9F27", label="Medium risk sector (35–50%)")
    high  = mpatches.Patch(color="#E24B4A", label="High risk sector (> 50%)")
    ax.legend(handles=[low, med, high], loc="lower right", fontsize=9)

    plt.tight_layout()
    path = output_dir / "linkedin_05_sector_risk.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path.name}")


def run_feature_engineering_internal(df):
    """Helper to get df with engineered columns attached."""
    from feature_engineering import (
        handle_missing_values, encode_categorical,
        create_financial_ratios, select_features
    )
    df = handle_missing_values(df)
    df = encode_categorical(df)
    df = create_financial_ratios(df)
    X, y = select_features(df)
    return df, X


def main():
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    print("Generating LinkedIn presentation charts...\n")

    chart1_model_comparison(output_dir)
    chart2_risk_distribution(output_dir)
    chart3_altman_zones(output_dir)
    chart4_feature_importance(output_dir)
    chart5_sector_default_rates(output_dir)

    print(f"""
==================================================
  ALL CHARTS SAVED TO outputs/ FOLDER
==================================================
  linkedin_01_model_comparison.png
  linkedin_02_risk_distribution.png
  linkedin_03_altman_zones.png
  linkedin_04_feature_importance.png
  linkedin_05_sector_risk.png
==================================================
  Ready to post on LinkedIn!
==================================================
    """)


if __name__ == "__main__":
    main()