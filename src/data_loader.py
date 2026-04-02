import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    """
    Loads the Corporate Credit Rating dataset.
    Place corporate_rating.csv in the data/ folder.
    """

    data_dir  = Path(__file__).resolve().parent.parent / "data"
    file_path = data_dir / "corporate_rating.csv"

    if not file_path.exists():
        print("ERROR: corporate_rating.csv not found in data/ folder.")
        print(f"Expected path: {file_path}")
        return None

    print("Loading Corporate Credit Rating dataset...")
    df = pd.read_csv(file_path)

    print(f"Dataset loaded   : {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Companies        : {df['Name'].nunique()} unique companies")
    print(f"Sectors          : {df['Sector'].nunique()} unique sectors")
    print(f"\nRating distribution:")
    print(df['Rating'].value_counts().to_string())

    # ── Convert letter ratings to binary default label ────────────────────
    # Investment grade (AAA, AA, A, BBB) = 0 (no default risk)
    # Speculative / junk (BB, B, CCC, CC, C, D) = 1 (default risk)
    investment_grade = ["AAA", "AA", "A", "BBB"]
    df["target"] = df["Rating"].apply(
        lambda x: 0 if x in investment_grade else 1
    )

    # Also create a numeric rating score (higher = safer)
    rating_score_map = {
        "AAA": 10, "AA": 9, "A": 8, "BBB": 7,
        "BB": 6,  "B": 5,  "CCC": 4,
        "CC": 3,  "C": 2,  "D": 1
    }
    df["rating_score"] = df["Rating"].map(rating_score_map)

    default_rate = df["target"].mean() * 100
    inv_count    = (df["target"] == 0).sum()
    spec_count   = (df["target"] == 1).sum()

    print(f"\nInvestment grade (AAA–BBB) : {inv_count} ({100-default_rate:.1f}%)")
    print(f"Speculative grade (BB–D)   : {spec_count} ({default_rate:.1f}%)")
    print(f"Default rate               : {default_rate:.1f}%")

    return df


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print("\nFirst 3 rows:")
        print(df[['Name', 'Sector', 'Rating', 'target']].head(3))