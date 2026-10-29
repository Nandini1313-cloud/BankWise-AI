# BankWise AI
## Machine Learning-Based Bank Term Deposit Subscription Prediction

BankWise AI is an end-to-end machine learning project that estimates whether a bank customer may subscribe to a term deposit using historical bank marketing data. The project includes data exploration, model training, model evaluation, and an interactive Streamlit dashboard.

## Features

- Customer subscription probability prediction
- Logistic Regression, Random Forest, and Gradient Boosting comparison
- Model performance metrics and confusion matrix
- Interactive dataset explorer with CSV download
- Downloadable evaluation reports
- Automated model pipeline tests

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib, Seaborn, and Plotly
- Streamlit
- Joblib and Pytest

## Dataset

The project uses the UCI Bank Marketing dataset.

Dataset file: `data/raw/bank-full.csv`

- Records: 45,211
- Columns: 17
- Target: `y` (`yes` or `no`)

The `duration` feature is excluded because call duration is only known after the call and is unsuitable for predicting subscription before contacting a customer.

Dataset source: https://archive.ics.uci.edu/dataset/222/bank+marketing

## Project Structure

```text
BankWise-AI/
├── app/
├── dashboard/
│   └── app.py
├── data/
│   └── raw/
├── docs/
│   ├── project_report.md
│   └── viva_questions.md
├── models/
├── notebooks/
├── outputs/
│   └── figures/
├── presentation/
├── reports/
├── src/
│   └── training/
├── tests/
│   └── test_model.py
├── requirements.txt
└── README.md
```

## Installation

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Run the dashboard

```powershell
python -m streamlit run dashboard\app.py
```

The dashboard normally opens at `http://localhost:8501`.

## Run Tests

```powershell
python -m pytest tests\test_model.py -v
```

The recorded test run passed all five tests.

## Initial Model Comparison

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 75.48% | 86.07% |
| Precision (subscription) | 26.62% | 42.31% |
| Recall (subscription) | 62.38% | 52.55% |
| F1-score (subscription) | 37.32% | 46.88% |
| ROC-AUC | 77.22% | 80.25% |

These are initial comparison results. Re-run the final evaluation scripts before publishing final results, especially after adding Gradient Boosting.

## Limitations

- Predictions are estimates, not guarantees.
- Historical data may not reflect current customer behavior.
- Subscription classes are imbalanced.
- Further calibration, fairness, privacy, and independent validation are needed before real-world deployment.

## Future Improvements

- Cross-validation and hyperparameter tuning
- Probability calibration and decision threshold analysis
- Model monitoring and version tracking
- Fairness and privacy assessment
- Controlled deployment and feedback collection

## Author

Nandini Jadhav

B.Tech in Artificial Intelligence and Machine Learning

## License

Academic project. Add an appropriate open-source license before distributing the project publicly.