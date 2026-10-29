from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "bank-full.csv"
MODEL_PATH = ROOT / "models" / "random_forest_model.joblib"
REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH, sep=";")

    # Match the exact test split used during model evaluation.
    X = df.drop(columns=["y", "duration"])
    y = (df["y"] == "yes").astype(int)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(X_test)[:, 1]

    results = []

    print("\nRANDOM FOREST: THRESHOLD COMPARISON")
    print("-" * 65)
    print(
        f"{'Threshold':>10} {'Precision':>12} "
        f"{'Recall':>10} {'F1 score':>10}"
    )

    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        results.append({
            "threshold": threshold,
            "precision_yes": precision,
            "recall_yes": recall,
            "f1_yes": f1,
        })

        print(
            f"{threshold:>10.2f} {precision:>12.4f} "
            f"{recall:>10.4f} {f1:>10.4f}"
        )

    results_df = pd.DataFrame(results)
    output_path = REPORTS_DIR / "threshold_comparison.csv"
    results_df.to_csv(output_path, index=False)

    best = results_df.loc[results_df["f1_yes"].idxmax()]

    print("\nBest threshold by F1 score:")
    print(best.to_string())
    print("\nResults saved:", output_path)


if __name__ == "__main__":
    main()