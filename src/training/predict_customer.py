from pathlib import Path
import argparse

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "random_forest_model.joblib"

# Provisional threshold from threshold exploration.
THRESHOLD = 0.40


def main():
    parser = argparse.ArgumentParser(
        description="Predict potential term-deposit subscription."
    )

    parser.add_argument("--age", type=int, default=35)
    parser.add_argument("--job", default="management")
    parser.add_argument("--marital", default="married")
    parser.add_argument("--education", default="tertiary")
    parser.add_argument("--default", dest="default_status", default="no")
    parser.add_argument("--balance", type=int, default=1500)
    parser.add_argument("--housing", default="yes")
    parser.add_argument("--loan", default="no")
    parser.add_argument("--contact", default="cellular")
    parser.add_argument("--day", type=int, default=15)
    parser.add_argument("--month", default="may")
    parser.add_argument("--campaign", type=int, default=1)
    parser.add_argument("--pdays", type=int, default=-1)
    parser.add_argument("--previous", type=int, default=0)
    parser.add_argument("--poutcome", default="unknown")

    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Train the models first using compare_models.py."
        )

    customer = pd.DataFrame([{
        "age": args.age,
        "job": args.job,
        "marital": args.marital,
        "education": args.education,
        "default": args.default_status,
        "balance": args.balance,
        "housing": args.housing,
        "loan": args.loan,
        "contact": args.contact,
        "day": args.day,
        "month": args.month,
        "campaign": args.campaign,
        "pdays": args.pdays,
        "previous": args.previous,
        "poutcome": args.poutcome,
    }])

    model = joblib.load(MODEL_PATH)
    probability = float(model.predict_proba(customer)[0, 1])
    prediction = probability >= THRESHOLD

    print("\nBANKWISE AI — CUSTOMER PREDICTION")
    print("-" * 40)
    print(f"Predicted subscription probability: {probability:.1%}")
    print(f"Decision threshold: {THRESHOLD:.0%}")

    if prediction:
        print("Prediction: Potential subscriber")
    else:
        print("Prediction: Less likely to subscribe")

    print(
        "\nNote: This is a model estimate, not a guarantee "
        "of customer behavior."
    )


if __name__ == "__main__":
    main()