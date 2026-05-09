"""
Train ML model for spoof detection.
Run in PyCharm with visualizations.
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:
    sns = None

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_TRAINING_DATA_PATH = DATA_DIR / "training_data.csv"
CORRELATION_MATRIX_PATH = DATA_DIR / "correlation_matrix.png"
FEATURE_DISTRIBUTIONS_PATH = DATA_DIR / "feature_distributions.png"
ROC_CURVES_PATH = DATA_DIR / "roc_curves.png"
DEFAULT_MODEL_OUTPUT_PATH = MODELS_DIR / "spoof_classifier.pkl"


def load_data(csv_path: str | Path = DEFAULT_TRAINING_DATA_PATH):
    """Load training data from CSV."""
    df = pd.read_csv(csv_path)

    print("=" * 60)
    print("DATASET INFO")
    print("=" * 60)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print("\nSample data:")
    print(df.head())

    return df


def prepare_features(df):
    """Separate features and labels."""
    feature_cols = [col for col in df.columns if col not in ["label", "filename"]]
    X = df[feature_cols].values
    y = df["label"].values

    print("\n" + "=" * 60)
    print("FEATURES")
    print("=" * 60)
    print(f"Feature names: {feature_cols}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("\nFeature Statistics:")
    for i, col in enumerate(feature_cols):
        print(f"  {col:20s}: min={X[:, i].min():.3f}, max={X[:, i].max():.3f}, mean={X[:, i].mean():.3f}")

    return X, y, feature_cols


def visualize_data(df, feature_cols):
    """Create visualizations."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    plt.figure(figsize=(12, 8))
    corr = df[feature_cols + ["label"]].corr()
    if sns is not None:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    else:
        plt.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
        plt.colorbar()
        labels = corr.columns.tolist()
        ticks = np.arange(len(labels))
        plt.xticks(ticks, labels, rotation=45, ha="right")
        plt.yticks(ticks, labels)
        for row_index, row in enumerate(corr.values):
            for col_index, value in enumerate(row):
                plt.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(CORRELATION_MATRIX_PATH, dpi=150)
    print(f"Saved: {CORRELATION_MATRIX_PATH}")
    plt.close()

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()

    for i, col in enumerate(feature_cols):
        df[df["label"] == 0][col].hist(ax=axes[i], bins=20, alpha=0.6, label="LIVE", color="green")
        df[df["label"] == 1][col].hist(ax=axes[i], bins=20, alpha=0.6, label="SPOOF", color="red")
        axes[i].set_title(col)
        axes[i].legend()
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FEATURE_DISTRIBUTIONS_PATH, dpi=150)
    print(f"Saved: {FEATURE_DISTRIBUTIONS_PATH}")
    plt.close()


def train_models(X_train, y_train, X_test, y_test, feature_cols):
    """Train and evaluate multiple models."""
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000, C=0.1),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42, C=1.0),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Training: {name}")
        print(f"{'=' * 60}")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["LIVE", "SPOOF"]))

        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        print(f"  True LIVE: {cm[0, 0]}")
        print(f"  False SPOOF (LIVE->SPOOF): {cm[0, 1]}")
        print(f"  False LIVE (SPOOF->LIVE): {cm[1, 0]}")
        print(f"  True SPOOF: {cm[1, 1]}")

        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC: {auc:.3f}")

        if hasattr(model, "feature_importances_"):
            print("\nFeature Importance:")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            for i in indices:
                print(f"  {feature_cols[i]:20s}: {importances[i]:.4f}")

        results[name] = {
            "model": model,
            "cv_scores": cv_scores,
            "test_accuracy": (y_test == y_pred).mean(),
            "auc": auc,
            "confusion_matrix": cm,
        }

    return results


def plot_roc_curves(results, X_test, y_test):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))

    for name, result in results.items():
        model = result["model"]
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = result["auc"]

        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - Model Comparison", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_CURVES_PATH, dpi=150)
    print(f"\nSaved: {ROC_CURVES_PATH}")
    plt.close()


def save_best_model(results, feature_cols, output_path: str | Path = DEFAULT_MODEL_OUTPUT_PATH):
    """Save the best model."""
    best_name = max(results.keys(), key=lambda k: results[k]["test_accuracy"])
    best_model = results[best_name]["model"]

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_name}")
    print("=" * 60)
    print(f"Test Accuracy: {results[best_name]['test_accuracy']:.3f}")
    print(f"ROC-AUC: {results[best_name]['auc']:.3f}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        "model": best_model,
        "feature_names": feature_cols,
        "model_name": best_name,
        "test_accuracy": results[best_name]["test_accuracy"],
        "auc": results[best_name]["auc"],
        "confusion_matrix": results[best_name]["confusion_matrix"],
    }

    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved to: {output_path}")

    return best_model


def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("SPOOF DETECTION MODEL TRAINING")
    print("=" * 60)

    df = load_data()
    X, y, feature_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTrain/Test Split:")
    print(f"  Train: {X_train.shape[0]} samples ({np.sum(y_train == 0)} LIVE, {np.sum(y_train == 1)} SPOOF)")
    print(f"  Test:  {X_test.shape[0]} samples ({np.sum(y_test == 0)} LIVE, {np.sum(y_test == 1)} SPOOF)")

    visualize_data(df, feature_cols)
    results = train_models(X_train, y_train, X_test, y_test, feature_cols)
    plot_roc_curves(results, X_test, y_test)
    save_best_model(results, feature_cols)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  {CORRELATION_MATRIX_PATH}")
    print(f"  {FEATURE_DISTRIBUTIONS_PATH}")
    print(f"  {ROC_CURVES_PATH}")
    print(f"  {DEFAULT_MODEL_OUTPUT_PATH}")
    print("\nNext steps:")
    print("  1. Check visualizations")
    print("  2. Review model performance")
    print("  3. Integrate model into live_liveness_preview.py")


if __name__ == "__main__":
    main()
