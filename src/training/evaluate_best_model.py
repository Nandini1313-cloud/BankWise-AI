from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "bank-full.csv"
METRICS_PATH = ROOT / "reports" / "model_comparison.json"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "outputs" / "figures"

def main():
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    best = max(results, key=lambda item: item["f1_yes"])
    model_name = best["model"]
    model_file = model_name.lower().replace(" ", "_") + "_model.joblib"
    model_path = MODELS_DIR / model_file

    model = joblib.load(model_path)

    df = pd.read_csv(DATA_PATH, sep=";")
    X = df.drop(columns=["y", "duration"])
    y = (df["y"] == "yes").astype(int)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No subscription", "Subscription"],
        digits=4,
        zero_division=0,
    )

    print(f"\nSELECTED MODEL: {model_name}")
    print("\nCLASSIFICATION REPORT")
    print(report)
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / "classification_report.txt"
    report_path.write_text(
        f"Selected model: {model_name}\n\n{report}\n"
        f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}\n",
        encoding="utf-8",
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No", "Yes"],
    )
    display.plot(values_format="d")
    plt.title(f"BankWise AI - Confusion Matrix ({model_name})")
    plt.tight_layout()

    figure_path = FIGURES_DIR / "confusion_matrix.png"
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nSaved report:", report_path)
    print("Saved confusion matrix:", figure_path)

if __name__ == "__main__":
    main()