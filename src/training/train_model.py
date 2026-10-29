
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "bank-full.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1. Load the real dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, sep=";")

    print("=" * 60)
    print("BANKWISE AI - BASELINE MODEL TRAINING")
    print("=" * 60)
    print("Original dataset shape:", df.shape)

    # 2. Remove exact duplicate rows
    original_rows = len(df)
    df = df.drop_duplicates().copy()

    print("Duplicate rows removed:", original_rows - len(df))
    print("Dataset shape after cleaning:", df.shape)

    # 3. Separate target and features
    y = df["y"].map({"no": 0, "yes": 1})

    # Exclude duration because it is only known after the call.
    # This prevents data leakage for pre-call predictions.
    X = df.drop(columns=["y", "duration"])

    if y.isnull().any():
        raise ValueError("Unexpected target values found in column 'y'.")

    # 4. Identify numerical and categorical columns
    numeric_features = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print("\nNumerical features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("Excluded feature: duration")

    # 5. Build preprocessing pipelines
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    # 6. Create the baseline model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # 7. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # 8. Train the model
    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)

    # 9. Evaluate using the held-out test set
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "Logistic Regression",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features_used": X.columns.tolist(),
        "excluded_duration_to_prevent_leakage": True,
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision_yes": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),
        "recall_yes": float(
            recall_score(
                y_test, predictions, zero_division=0
            )
        ),
        "f1_yes": float(
            f1_score(
                y_test, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
    }

    print("\nEVALUATION METRICS")
    print("-" * 40)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No subscription", "Subscription"],
            zero_division=0,
        )
    )

    # 10. Save the complete fitted pipeline
    model_path = MODEL_DIR / "baseline_model.joblib"
    joblib.dump(model, model_path)

    # 11. Save metrics as JSON
    metrics_path = REPORT_DIR / "baseline_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nSUCCESS: Baseline model trained and evaluated.")
    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)


if __name__ == "__main__":
    main()
