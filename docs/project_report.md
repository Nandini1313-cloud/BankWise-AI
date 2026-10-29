# BankWise AI
## Machine Learning-Based Bank Term Deposit Subscription Prediction

### 1. Abstract
BankWise AI is a machine learning project that estimates whether a bank customer may subscribe to a term deposit based on customer demographics, account information, and previous marketing campaign details. The system combines data exploration, preprocessing, model comparison, prediction, and an interactive Streamlit dashboard.

### 2. Problem Statement
Bank marketing campaigns require time and resources. Contacting every customer without prioritization may reduce campaign efficiency. BankWise AI investigates whether historical campaign data can help estimate subscription likelihood and support campaign analysis.

### 3. Objectives
- Explore a real bank marketing dataset.
- Clean and preprocess numerical and categorical features.
- Train and compare multiple classification models.
- Evaluate subscription predictions using precision, recall, F1-score, accuracy, and ROC-AUC.
- Build an interactive dashboard for customer predictions and model evaluation.
- Test the model prediction pipeline automatically.

### 4. Dataset
The project uses the UCI Bank Marketing dataset, stored locally as `data/raw/bank-full.csv`.

The dataset contains 45,211 records and 17 columns. The target column is `y`, which indicates whether a customer subscribed to a term deposit (`yes` or `no`).

The `duration` feature is excluded from the prediction model because call duration is only known after a call and would not be available for pre-contact targeting.

### 5. Methodology
1. Load and inspect the dataset.
2. Explore the target distribution and numerical features.
3. Separate the target from the input features.
4. Exclude `y` and `duration` from the model inputs.
5. Split the dataset into training and testing sets using an 80:20 stratified split with random state 42.
6. Impute missing values, scale numerical features, and one-hot encode categorical features through a preprocessing pipeline.
7. Train Logistic Regression, Random Forest, and Gradient Boosting models.
8. Compare model performance using the same test split.
9. Select a model according to positive-class F1-score for the dashboard.
10. Evaluate the selected model and test its prediction pipeline.

### 6. Model Evaluation
The initial comparison of Logistic Regression and Random Forest produced these results:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.7548 | 0.8607 |
| Precision (yes) | 0.2662 | 0.4231 |
| Recall (yes) | 0.6238 | 0.5255 |
| F1-score (yes) | 0.3732 | 0.4688 |
| ROC-AUC | 0.7722 | 0.8025 |

A subsequent evaluation of the selected Random Forest model reported:

- Accuracy: 88.07%
- Subscription precision: 48.83%
- Subscription recall: 41.30%
- Subscription F1-score: 44.75%
- ROC-AUC: 0.7913

These values come from different recorded evaluations and should not be combined as if they were one identical run. Re-run the final model comparison and evaluation scripts before publishing final results.

### 7. System Features
- Overview page with dataset and subscription statistics.
- Customer prediction form.
- Model comparison and evaluation metrics.
- Dataset filtering and CSV download.
- Downloadable evaluation reports.
- Confusion matrix visualization.
- Automated model pipeline tests.

### 8. Technology Stack
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Plotly
- Joblib
- Streamlit
- Pytest

### 9. Testing
The automated test suite contains five tests covering dataset availability, model file availability, prediction input compatibility, probability validity, and valid output classes.

Recorded test result: 5 passed.

### 10. Limitations
- Historical campaign data may not represent current customer behavior.
- Model estimates do not establish causation.
- Subscription classes are imbalanced.
- Prediction quality depends on data quality and model calibration.
- A prediction must not be treated as a guarantee of customer behavior.
- Fairness, privacy, drift, and operational performance require further evaluation before real-world deployment.

### 11. Future Scope
- Validate models using cross-validation and an untouched holdout set.
- Tune hyperparameters and decision thresholds without using the test set for selection.
- Evaluate calibration and precision-recall trade-offs.
- Add monitoring and model version tracking.
- Conduct fairness and privacy assessments.
- Deploy the dashboard in a controlled environment after validation.

### 12. Conclusion
BankWise AI demonstrates an end-to-end machine learning workflow for bank term deposit subscription prediction. It includes real-data exploration, preprocessing, model training, performance comparison, interactive prediction, visualization, and automated tests. Further validation is necessary before using its estimates for real marketing decisions.

### 13. References
1. Moro, S., Cortez, P., and Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, 62, 22–31.
2. UCI Machine Learning Repository. Bank Marketing Dataset. https://archive.ics.uci.edu/dataset/222/bank+marketing