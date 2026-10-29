from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "bank-full.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "outputs" / "figures"

st.set_page_config(
    page_title="BankWise AI | Customer Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0b1120; color: #e5e7eb; }
    [data-testid="stSidebar"] { background-color: #111827; }
    .hero {
        padding: 28px 30px;
        border-radius: 18px;
        background: linear-gradient(120deg, #14233f, #183c4a);
        border: 1px solid #29465c;
        margin-bottom: 24px;
    }
    .hero h1 { color: #ffffff; margin-bottom: 8px; }
    .hero p { color: #c2d2e3; font-size: 16px; }
    div[data-testid="stMetric"] {
        background: #141e30;
        border: 1px solid #29374b;
        padding: 18px;
        border-radius: 14px;
    }
    .note {
        background: #142238;
        border-left: 4px solid #38bdf8;
        padding: 14px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH, sep=";")


@st.cache_data
def load_comparison():
    path = REPORTS_DIR / "model_comparison.json"
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as file:
        return pd.DataFrame(json.load(file))


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


df = load_data()
comparison = load_comparison()

st.sidebar.markdown("# 🏦 BankWise AI")
st.sidebar.caption("Customer intelligence platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Customer Prediction",
        "Model Performance",
        "Dataset Explorer",
        "Reports",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning • Responsible AI • Banking Analytics")

st.markdown("""
<div class="hero">
    <h1>BankWise AI</h1>
    <p>Intelligent customer subscription prediction powered by machine learning.</p>
</div>
""", unsafe_allow_html=True)

if df is None:
    st.error(
        "Dataset not found. Expected file: "
        "data/raw/bank-full.csv"
    )
    st.stop()

if "y" not in df.columns:
    st.error("The dataset must contain the target column 'y'.")
    st.stop()

if page == "Overview":
    st.subheader("Business Overview")
    st.write(
        "Explore customer data, estimate subscription likelihood, "
        "and review model evaluation results."
    )

    total = len(df)
    subscribers = int((df["y"] == "yes").sum())
    non_subscribers = int((df["y"] == "no").sum())
    subscription_rate = subscribers / total * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{total:,}")
    c2.metric("Subscribed", f"{subscribers:,}")
    c3.metric("Not Subscribed", f"{non_subscribers:,}")
    c4.metric("Subscription Rate", f"{subscription_rate:.2f}%")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Subscription Distribution")
        target_counts = (
            df["y"].value_counts()
            .rename_axis("Outcome")
            .reset_index(name="Customers")
        )
        fig = px.pie(
            target_counts,
            names="Outcome",
            values="Customers",
            hole=0.62,
            color_discrete_sequence=["#38bdf8", "#64748b"],
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Model Comparison")
        if not comparison.empty:
            chart = comparison.melt(
                id_vars="model",
                value_vars=[
                    "accuracy", "precision_yes", "recall_yes",
                    "f1_yes", "roc_auc",
                ],
                var_name="Metric",
                value_name="Score",
            )
            fig = px.bar(
                chart,
                x="Metric",
                y="Score",
                color="model",
                barmode="group",
                template="plotly_dark",
                title="Evaluation Metrics",
            )
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run model comparison to display evaluation metrics.")

    st.markdown(
        '<div class="note">Important: Predictions are estimates, not '
        'guarantees of customer behavior. Use them as decision support, '
        'not as the sole basis for decisions about individuals.</div>',
        unsafe_allow_html=True,
    )


elif page == "Customer Prediction":
    st.subheader("Customer Subscription Prediction")
    st.write(
        "Enter customer information to estimate the likelihood "
        "of subscribing to a bank term deposit."
    )

    if comparison.empty:
        st.error("Model comparison report is missing. Train the models first.")
        st.stop()

    best = max(comparison.to_dict("records"), key=lambda x: x["f1_yes"])
    model_name = best["model"]
    model_filename = model_name.lower().replace(" ", "_") + "_model.joblib"
    model_path = MODELS_DIR / model_filename

    if not model_path.exists():
        st.error(f"Trained model file not found: {model_path}")
        st.stop()

    st.caption(f"Model selected by positive-class F1 score: {model_name}")

    with st.form("prediction_form"):
        st.markdown("#### Customer Profile")
        a, b, c = st.columns(3)

        with a:
            age = st.number_input("Age", 18, 100, 35)
            job = st.selectbox(
                "Job",
                [
                    "admin.", "blue-collar", "entrepreneur", "housemaid",
                    "management", "retired", "self-employed", "services",
                    "student", "technician", "unemployed", "unknown",
                ],
            )
            marital = st.selectbox(
                "Marital Status", ["married", "single", "divorced"]
            )
            education = st.selectbox(
                "Education",
                [
                    "primary", "secondary", "tertiary", "unknown"
                ],
            )

        with b:
            default_balance = int(df["balance"].median())
            balance = st.number_input(
                "Account Balance", value=default_balance
            )
            housing = st.selectbox("Housing Loan", ["yes", "no"])
            loan = st.selectbox("Personal Loan", ["yes", "no"])
            contact = st.selectbox(
                "Contact Type", ["cellular", "telephone", "unknown"]
            )
            month = st.selectbox(
                "Last Contact Month",
                [
                    "jan", "feb", "mar", "apr", "may", "jun",
                    "jul", "aug", "sep", "oct", "nov", "dec",
                ],
                index=4,
            )

        with c:
            day = st.number_input("Last Contact Day", 1, 31, 15)
            campaign = st.number_input(
                "Contacts During This Campaign", 1, 100, 1
            )
            pdays = st.number_input(
                "Days Since Previous Contact",
                -1, 999, -1,
                help="-1 means the customer was not previously contacted.",
            )
            previous = st.number_input(
                "Previous Campaign Contacts", 0, 100, 0
            )
            poutcome = st.selectbox(
                "Previous Campaign Outcome",
                ["unknown", "failure", "other", "success"],
            )

        submitted = st.form_submit_button(
            "Predict Subscription Likelihood",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        input_data = pd.DataFrame([{
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": "no",
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
        }])

        try:
            model = load_model(str(model_path))
            probability = float(model.predict_proba(input_data)[0][1])
            prediction = int(probability >= 0.5)

            st.markdown("### Prediction Result")
            r1, r2 = st.columns(2)
            r1.metric("Estimated Subscription Probability", f"{probability:.1%}")
            r2.metric(
                "Model Classification",
                "Likely to Subscribe" if prediction else "Less Likely to Subscribe",
            )
            st.progress(probability)

            st.caption(
                "This probability is an estimate from the trained model. "
                "It is not a guarantee or a causal explanation."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


elif page == "Model Performance":
    st.subheader("Model Performance & Evaluation")

    if comparison.empty:
        st.warning("No model comparison results found.")
        st.stop()

    metric_columns = [
        "accuracy", "precision_yes", "recall_yes", "f1_yes", "roc_auc"
    ]

    st.dataframe(
        comparison[["model"] + metric_columns].style.format(
            {column: "{:.4f}" for column in metric_columns}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Compare Models")
    chart_data = comparison.melt(
        id_vars="model",
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Score",
    )
    fig = px.bar(
        chart_data,
        x="Metric",
        y="Score",
        color="model",
        barmode="group",
        template="plotly_dark",
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    matrix_path = FIGURES_DIR / "confusion_matrix.png"
    if matrix_path.exists():
        st.markdown("#### Saved Confusion Matrix")
        st.image(str(matrix_path), use_container_width=True)

    report_path = REPORTS_DIR / "classification_report.txt"
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as file:
            st.text(file.read())


elif page == "Dataset Explorer":
    st.subheader("Dataset Explorer")

    search_column = st.selectbox("Filter by column", df.columns)
    values = sorted(df[search_column].dropna().astype(str).unique().tolist())
    selected_value = st.selectbox(
        "Choose a value",
        ["All"] + values,
    )

    filtered = df
    if selected_value != "All":
        filtered = df[
            df[search_column].astype(str) == selected_value
        ]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
    st.dataframe(filtered.head(1000), use_container_width=True)

    with st.expander("Dataset Summary"):
        st.dataframe(
            df.describe(include="all").transpose(),
            use_container_width=True,
        )

    st.download_button(
        "Download Filtered Data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="bankwise_filtered_data.csv",
        mime="text/csv",
    )


elif page == "Reports":
    st.subheader("Project Reports & Artifacts")

    files = [
        REPORTS_DIR / "baseline_metrics.json",
        REPORTS_DIR / "model_comparison.json",
        REPORTS_DIR / "model_comparison.csv",
        REPORTS_DIR / "classification_report.txt",
        FIGURES_DIR / "model_comparison.png",
        FIGURES_DIR / "confusion_matrix.png",
    ]

    for file_path in files:
        if file_path.exists():
            st.markdown(f"**{file_path.name}**")
            if file_path.suffix.lower() == ".png":
                st.image(str(file_path), use_container_width=True)
            else:
                st.download_button(
                    f"Download {file_path.name}",
                    data=file_path.read_bytes(),
                    file_name=file_path.name,
                    key=str(file_path),
                )
        else:
            st.caption(f"Not generated yet: {file_path.name}")

st.sidebar.markdown("---")
st.sidebar.caption("BankWise AI • Academic ML Project")
