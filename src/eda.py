import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from data_loader import load_data

def run_eda(df):
    """
    Runs exploratory data analysis and saves plots to outputs/ folder.
    """

    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 120

    # ── 1. Default rate overview ──────────────────────────────────────────
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Total applicants : {len(df)}")
    print(f"Default (1)      : {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
    print(f"No Default (0)   : {(df['target']==0).sum()} ({(df['target']==0).mean()*100:.1f}%)")
    print(f"Missing values   : {df.isnull().sum().sum()}")

    # ── 2. Target distribution pie chart ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    df["target"].value_counts().plot.pie(
        labels=["No Default", "Default"],
        colors=["#639922", "#E24B4A"],
        autopct="%1.1f%%",
        startangle=90,
        ax=ax
    )
    ax.set_ylabel("")
    ax.set_title("Default vs No Default")
    plt.tight_layout()
    plt.savefig(output_dir / "01_target_distribution.png")
    plt.show()
    print("\nSaved: 01_target_distribution.png")

    # ── 3. Credit amount distribution ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df[df["target"]==0]["credit_amount"], bins=30,
                 color="#639922", alpha=0.7, label="No Default")
    axes[0].hist(df[df["target"]==1]["credit_amount"], bins=30,
                 color="#E24B4A", alpha=0.7, label="Default")
    axes[0].set_title("Credit Amount by Default Status")
    axes[0].set_xlabel("Credit Amount (DM)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(df[df["target"]==0]["duration"], bins=20,
                 color="#639922", alpha=0.7, label="No Default")
    axes[1].hist(df[df["target"]==1]["duration"], bins=20,
                 color="#E24B4A", alpha=0.7, label="Default")
    axes[1].set_title("Loan Duration by Default Status")
    axes[1].set_xlabel("Duration (months)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "02_credit_amount_duration.png")
    plt.show()
    print("Saved: 02_credit_amount_duration.png")

    # ── 4. Age distribution ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(df[df["target"]==0]["age"], color="#639922",
                fill=True, alpha=0.4, label="No Default", ax=ax)
    sns.kdeplot(df[df["target"]==1]["age"], color="#E24B4A",
                fill=True, alpha=0.4, label="Default", ax=ax)
    ax.set_title("Age Distribution by Default Status")
    ax.set_xlabel("Age (years)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "03_age_distribution.png")
    plt.show()
    print("Saved: 03_age_distribution.png")

    # ── 5. Default rate by checking account status ────────────────────────
    checking_map = {
        "A11": "< 0 DM",
        "A12": "0–200 DM",
        "A13": "> 200 DM",
        "A14": "No account"
    }
    df["checking_label"] = df["checking_account"].map(checking_map)
    default_by_checking = df.groupby("checking_label")["target"].mean() * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(default_by_checking.index, default_by_checking.values,
                  color=["#E24B4A" if v > 30 else "#378ADD"
                         for v in default_by_checking.values])
    ax.axhline(30, color="gray", linestyle="--", linewidth=1, label="Overall avg (30%)")
    ax.set_title("Default Rate by Checking Account Status")
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel("Checking Account")
    ax.legend()
    for bar, val in zip(bars, default_by_checking.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "04_default_by_checking.png")
    plt.show()
    print("Saved: 04_default_by_checking.png")

    # ── 6. Correlation heatmap (numeric columns) ──────────────────────────
    numeric_cols = ["duration", "credit_amount", "installment_rate",
                    "residence_years", "age", "existing_credits",
                    "dependents", "target"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn_r",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap — Numeric Features")
    plt.tight_layout()
    plt.savefig(output_dir / "05_correlation_heatmap.png")
    plt.show()
    print("Saved: 05_correlation_heatmap.png")

    print("\nEDA complete! All charts saved to outputs/ folder.")
    df.drop(columns=["checking_label"], inplace=True)
    return df


if __name__ == "__main__":
    df = load_data()
    run_eda(df)