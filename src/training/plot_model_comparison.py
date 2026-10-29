
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "reports" / "model_comparison.json"
FIGURES_DIR = ROOT / "outputs" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Comparison results not found: {METRICS_PATH}"
        )

    with METRICS_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    df = pd.DataFrame(results)

    metric_columns = [
        "accuracy",
        "precision_yes",
        "recall_yes",
        "f1_yes",
        "roc_auc",
    ]

    print("\nMODEL COMPARISON RESULTS")
    print(df[["model"] + metric_columns].round(4).to_string(index=False))

    sns.set_theme(style="whitegrid")

    # Plot comparison of evaluation metrics
    plot_df = df.melt(
        id_vars="model",
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="model",
    )

    plt.title("BankWise AI - Model Performance Comparison")
    plt.xlabel("Evaluation Metric")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.legend(title="Model")
    plt.tight_layout()

    chart_path = FIGURES_DIR / "model_comparison.png"
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Save comparison as a CSV file
    csv_path = ROOT / "reports" / "model_comparison.csv"
    df.to_csv(csv_path, index=False)

    print("\nChart saved:", chart_path)
    print("CSV report saved:", csv_path)


if __name__ == "__main__":
    main()
