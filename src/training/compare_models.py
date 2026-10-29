from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "bank-full.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    # Load the real dataset.
    df = pd.read_csv(DATA_PATH, sep=";")

    # Exclude duration because it is not known before contacting a customer.
    X = df.drop(columns=["y", "duration"])
    y = (df["y"] == "yes").astype(int)

    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        ),
    }

    results = []

    for model_name, classifier in models.items():
        print(f"\nTraining {model_name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model": model_name,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_yes": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall_yes": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1_yes": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }

        results.append(metrics)

        filename = model_name.lower().replace(" ", "_") + "_model.joblib"
        model_path = MODELS_DIR / filename
        joblib.dump(pipeline, model_path)

        print("Saved model:", model_path)
        print("Metrics:", metrics)

    results_path = REPORTS_DIR / "model_comparison.json"
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    csv_path = REPORTS_DIR / "model_comparison.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)

    best_model = max(results, key=lambda item: item["f1_yes"])

    print("\n" + "=" * 55)
    print("FINAL MODEL COMPARISON")
    print("=" * 55)
    print(pd.DataFrame(results).round(4).to_string(index=False))
    print("\nBest model by positive-class F1 score:", best_model["model"])
    print("Comparison JSON:", results_path)
    print("Comparison CSV:", csv_path)


if __name__ == "__main__":
    main()