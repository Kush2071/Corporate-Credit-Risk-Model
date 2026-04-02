import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data
from feature_engineering import run_feature_engineering


def split_and_scale(X, y):
    """
    Splits data into train/test sets and scales features.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"Training set : {X_train_scaled.shape[0]} samples")
    print(f"Test set     : {X_test_scaled.shape[0]} samples")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model.
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Logistic Regression trained.")
    return model


def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    print("Random Forest trained.")
    return model


def evaluate_model(model, X_test, y_test, model_name, output_dir):
    """
    Evaluates model and prints all key metrics.
    """
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc*100:.1f}%")
    print(f"  Precision : {prec:.3f}")
    print(f"  Recall    : {rec:.3f}")
    print(f"  F1 Score  : {f1:.3f}")
    print(f"  ROC-AUC   : {auc:.3f}")

    # Confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Default", "Default"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"{model_name} — Confusion Matrix")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, color="#378ADD", lw=2,
                 label=f"ROC curve (AUC = {auc:.2f})")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"{model_name} — ROC Curve")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_")
    plt.savefig(output_dir / f"{fname}_evaluation.png")
    plt.show()
    print(f"  Saved: {fname}_evaluation.png")

    return {"model": model_name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "auc": auc}


def plot_feature_importance(model, feature_names, output_dir):
    """
    Plots feature importances from Random Forest.
    """
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:15]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        [feature_names[i] for i in indices][::-1],
        importances[indices][::-1],
        color="#378ADD"
    )
    ax.set_title("Top 15 Feature Importances — Random Forest")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importances.png")
    plt.show()
    print("Saved: feature_importances.png")


def save_models(lr_model, rf_model, scaler, output_dir):
    """
    Saves trained models and scaler to disk.
    """
    pickle.dump(lr_model, open(output_dir / "logistic_regression.pkl", "wb"))
    pickle.dump(rf_model, open(output_dir / "random_forest.pkl",        "wb"))
    pickle.dump(scaler,   open(output_dir / "scaler.pkl",               "wb"))
    print("\nModels saved to outputs/ folder.")


def run_model_training():
    """
    Full model training pipeline.
    """
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Load and prepare data
    df = load_data()
    X, y = run_feature_engineering(df)

    print("\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)

    # Split and scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # Train both models
    print()
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate both models
    lr_results = evaluate_model(lr_model, X_test, y_test,
                                "Logistic Regression", output_dir)
    rf_results = evaluate_model(rf_model, X_test, y_test,
                                "Random Forest", output_dir)

    # Feature importance
    plot_feature_importance(rf_model, list(X.columns), output_dir)

    # Compare models
    print(f"\n{'='*50}")
    print("  MODEL COMPARISON")
    print(f"{'='*50}")
    print(f"  {'Metric':<12} {'Log. Reg':>10} {'Rand. Forest':>14}")
    print(f"  {'-'*38}")
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        print(f"  {metric:<12} "
              f"{lr_results[metric]:>10.3f} "
              f"{rf_results[metric]:>14.3f}")

    # Save models
    save_models(lr_model, rf_model, scaler, output_dir)

    return lr_model, rf_model, scaler, X.columns.tolist()


if __name__ == "__main__":
    lr_model, rf_model, scaler, feature_names = run_model_training()