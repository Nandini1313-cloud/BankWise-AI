
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# BANKWISE AI - PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "bank-full.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "outputs" / "figures"

st.set_page_config(
    page_title="BankWise AI | Customer Intelligence",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG = "#080D1A"
PANEL = "rgba(20, 30, 55, 0.58)"
BORDER = "rgba(148, 163, 184, 0.18)"
BLUE = "#60A5FA"
CYAN = "#67E8F9"
VIOLET = "#A78BFA"
GREEN = "#6EE7B7"
TEXT = "#F1F5F9"
MUTED = "#A5B4CC"


# ============================================================
# GLASSMORPHISM DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #080D1A;
        --panel: rgba(20, 30, 55, 0.58);
        --border: rgba(148, 163, 184, 0.18);
        --text: #F1F5F9;
        --muted: #A5B4CC;
        --blue: #60A5FA;
        --cyan: #67E8F9;
        --violet: #A78BFA;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 5% 0%,
                rgba(37, 99, 235, 0.19), transparent 35%),
            radial-gradient(ellipse at 95% 8%,
                rgba(124, 58, 237, 0.17), transparent 32%),
            radial-gradient(ellipse at 60% 100%,
                rgba(8, 145, 178, 0.10), transparent 35%),
            #080D1A;
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(8, 13, 26, 0.45);
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1600px;
    }

    /* SIDEBAR: IMPROVED VISIBILITY AND NAVIGATION */

    [data-testid="stSidebar"] {
        min-width: 270px;
        width: 270px;
        background: rgba(9, 15, 31, 0.96);
        border-right: 1px solid rgba(148, 163, 184, 0.22);
    }

    [data-testid="stSidebar"] > div:first-child {
        background:
            radial-gradient(circle at 0% 0%,
                rgba(59, 130, 246, 0.16), transparent 45%),
            rgba(9, 15, 31, 0.96);
        padding-top: 1.2rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--muted);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        color: #F1F5F9 !important;
        font-weight: 700;
        font-size: 13px;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
        gap: 10px;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
        padding: 12px 14px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.045);
        transition: background 160ms ease, border-color 160ms ease;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover {
        background: rgba(59, 130, 246, 0.16);
        border-color: rgba(96, 165, 250, 0.48);
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label p {
        color: #F1F5F9 !important;
        font-weight: 600;
    }

    h1, h2, h3, h4 {
        color: var(--text) !important;
        letter-spacing: -0.035em;
    }

    p, li, label {
        color: #CBD5E1;
    }

    .brand {
        padding: 18px 17px;
        margin-bottom: 18px;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            rgba(59, 130, 246, 0.20),
            rgba(139, 92, 246, 0.12)
        );
        border: 1px solid rgba(147, 197, 253, 0.24);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.20);
    }

    .brand-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        border-radius: 13px;
        margin-bottom: 12px;
        color: white;
        font-size: 21px;
        font-weight: 800;
        background: linear-gradient(135deg, #2563EB, #7C3AED);
        box-shadow: 0 5px 24px rgba(59, 130, 246, 0.32);
    }

    .brand h2 {
        margin: 0;
        font-size: 23px;
        font-weight: 800;
    }

    .brand p {
        margin: 6px 0 0 0;
        font-size: 12px;
        color: #A5B4CC;
        line-height: 1.6;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 32px;
        margin: 0 0 28px 0;
        border-radius: 24px;
        border: 1px solid rgba(147, 197, 253, 0.24);
        background:
            linear-gradient(115deg,
                rgba(30, 64, 125, 0.50),
                rgba(76, 29, 149, 0.30),
                rgba(15, 118, 110, 0.18)),
            rgba(17, 25, 46, 0.62);
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.24);
        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        right: -70px;
        top: -100px;
        border-radius: 50%;
        background: rgba(96, 165, 250, 0.15);
        filter: blur(35px);
        pointer-events: none;
    }

    .eyebrow {
        color: #93C5FD;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero h1 {
        font-size: clamp(30px, 4vw, 45px);
        font-weight: 800;
        margin: 0 0 12px 0;
        line-height: 1.12;
    }

    .hero p {
        color: #C4D2E5;
        max-width: 720px;
        font-size: 15px;
        line-height: 1.8;
        margin: 0;
    }

    .section-label {
        color: #93C5FD;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin: 10px 0 8px 0;
    }

    .glass-card {
        height: 100%;
        min-height: 128px;
        padding: 21px;
        border: 1px solid var(--border);
        border-radius: 19px;
        background: linear-gradient(
            145deg,
            rgba(30, 41, 68, 0.70),
            rgba(15, 23, 42, 0.52)
        );
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        transition: transform 180ms ease, border-color 180ms ease,
                    box-shadow 180ms ease;
    }

    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(96, 165, 250, 0.45);
        box-shadow: 0 17px 35px rgba(0, 0, 0, 0.23);
    }

    .metric-label {
        color: #A5B4CC;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 13px;
    }

    .metric-value {
        color: #F8FAFC;
        font-size: clamp(23px, 2.3vw, 31px);
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }

    .metric-foot {
        color: #8192AE;
        font-size: 11px;
        margin-top: 10px;
    }

    .glass-panel {
        padding: 22px;
        border: 1px solid var(--border);
        border-radius: 20px;
        background: var(--panel);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.13);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }

    .result-panel {
        padding: 25px;
        border: 1px solid rgba(96, 165, 250, 0.30);
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.17),
            rgba(124, 58, 237, 0.12)
        );
    }

    .status-pill {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 30px;
        color: #BFDBFE;
        background: rgba(59, 130, 246, 0.14);
        border: 1px solid rgba(96, 165, 250, 0.28);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    .note {
        padding: 16px 18px;
        border: 1px solid rgba(103, 232, 249, 0.22);
        border-left: 4px solid #67E8F9;
        border-radius: 13px;
        background: rgba(8, 145, 178, 0.08);
        color: #C7D8E9;
        line-height: 1.75;
        font-size: 13px;
    }

    .stButton > button,
    .stFormSubmitButton > button,
    .stDownloadButton > button {
        border: 1px solid rgba(147, 197, 253, 0.30);
        border-radius: 12px;
        background: linear-gradient(110deg, #2563EB, #6D5AE6);
        color: #FFFFFF;
        font-weight: 700;
        min-height: 43px;
        transition: all 160ms ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #93C5FD;
        box-shadow: 0 7px 23px rgba(59, 130, 246, 0.24);
        transform: translateY(-1px);
        color: white;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background: rgba(15, 23, 42, 0.72);
        border-color: rgba(148, 163, 184, 0.25);
        border-radius: 11px;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="input"] input {
        color: #F1F5F9;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px;
        backdrop-filter: blur(16px);
    }

    [data-testid="stMetricLabel"] {
        color: #A5B4CC;
    }

    [data-testid="stMetricValue"] {
        color: #F8FAFC;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(20, 30, 55, 0.35);
    }

    hr {
        border-color: rgba(148, 163, 184, 0.15);
    }

    .footer {
        margin-top: 35px;
        padding: 18px 0 8px 0;
        border-top: 1px solid var(--border);
        color: #71819B;
        font-size: 11px;
        text-align: center;
        line-height: 1.8;
    }

    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 1rem 2rem 1rem;
        }

        .hero {
            padding: 23px;
            border-radius: 18px;
        }

        .glass-card {
            min-height: 110px;
            padding: 16px;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            transition: none !important;
            animation: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA AND MODEL LOADING
# ============================================================

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

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        result = pd.DataFrame(data)

        required = {
            "model", "accuracy", "precision_yes",
            "recall_yes", "f1_yes", "roc_auc",
        }

        if not required.issubset(result.columns):
            return pd.DataFrame()

        for column in required - {"model"}:
            result[column] = pd.to_numeric(
                result[column], errors="coerce"
            )

        return result.dropna(
            subset=[
                "accuracy", "precision_yes",
                "recall_yes", "f1_yes", "roc_auc",
            ]
        )
    except (OSError, ValueError, TypeError):
        return pd.DataFrame()


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


def glass_metric(label, value, footnote, accent="#60A5FA"):
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{accent}">
                {value}
            </div>
            <div class="metric-foot">{footnote}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_chart(fig, height=350):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, sans-serif",
            color="#CBD5E1",
            size=12,
        ),
        title_font=dict(size=16, color="#F1F5F9"),
        margin=dict(l=12, r=12, t=55, b=15),
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor="#111C32",
            bordercolor="#475569",
            font_color="#F8FAFC",
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="rgba(148,163,184,0.15)",
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.10)",
        zeroline=False,
    )
    return fig


def section_heading(eyebrow, title, description=None):
    st.markdown(
        f'<div class="section-label">{eyebrow}</div>',
        unsafe_allow_html=True,
    )
    st.subheader(title)

    if description:
        st.caption(description)


def model_filename(model_name):
    return model_name.lower().replace(" ", "_") + "_model.joblib"


# ============================================================
# LOAD APP DATA
# ============================================================

df = load_data()
comparison = load_comparison()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">B</div>
            <h2>BankWise AI</h2>
            <p>Customer Intelligence Platform<br>
            Machine Learning • Banking Analytics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">WORKSPACE</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate to",
        [
            "Overview",
            "Customer Prediction",
            "Model Performance",
            "Dataset Explorer",
            "Reports",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="padding:10px 3px;">
            <div class="section-label">PROJECT STATUS</div>
            <p style="font-size:12px;line-height:1.8;">
                <span style="color:#6EE7B7;">●</span>
                Local project workspace<br>
                <span style="color:#93C5FD;">●</span>
                Historical bank marketing data<br>
                <span style="color:#C4B5FD;">●</span>
                Trained ML model artifacts
            </p>
            <p style="font-size:11px;color:#71819B;">
                Academic ML project
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# GLOBAL HERO
# ============================================================

page_descriptions = {
    "Overview": (
        "BUSINESS INTELLIGENCE",
        "Understand your customers.<br>Make data-informed decisions.",
        "Explore bank marketing outcomes, compare model performance, "
        "and understand customer subscription patterns.",
    ),
    "Customer Prediction": (
        "PREDICTIVE ANALYTICS",
        "Explore subscription likelihood.",
        "Enter a customer profile and evaluate the trained model's estimate.",
    ),
    "Model Performance": (
        "MODEL EVALUATION",
        "Measure model performance.",
        "Compare classification metrics and inspect saved evaluation reports.",
    ),
    "Dataset Explorer": (
        "DATA EXPLORATION",
        "Discover patterns in your data.",
        "Filter the historical dataset, inspect records, and export selected data.",
    ),
    "Reports": (
        "PROJECT ARTIFACTS",
        "Your results, in one place.",
        "Access generated model metrics, evaluation reports, and figures.",
    ),
}

eyebrow, hero_title, hero_description = page_descriptions[page]

st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">{eyebrow}</div>
        <h1>{hero_title}</h1>
        <p>{hero_description}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# VALIDATE DATASET
# ============================================================

if df is None:
    st.error(
        "Dataset not found. Expected file: data/raw/bank-full.csv"
    )
    st.stop()

if "y" not in df.columns:
    st.error("The dataset must contain the target column 'y'.")
    st.stop()

if df.empty:
    st.error("The dataset is empty. Check the source CSV file.")
    st.stop()


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":
    section_heading(
        "OVERVIEW",
        "Business overview",
        "A snapshot of the historical bank marketing dataset.",
    )

    total = len(df)
    subscribers = int((df["y"] == "yes").sum())
    non_subscribers = int((df["y"] == "no").sum())
    subscription_rate = subscribers / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    with c1:
        glass_metric(
            "TOTAL RECORDS",
            f"{total:,}",
            "Historical customer records",
            BLUE,
        )

    with c2:
        glass_metric(
            "SUBSCRIBED",
            f"{subscribers:,}",
            "Positive target outcomes",
            GREEN,
        )

    with c3:
        glass_metric(
            "NOT SUBSCRIBED",
            f"{non_subscribers:,}",
            "Negative target outcomes",
            VIOLET,
        )

    with c4:
        glass_metric(
            "SUBSCRIPTION RATE",
            f"{subscription_rate:.2f}%",
            "Observed in this dataset",
            CYAN,
        )

    st.markdown("")
    left, right = st.columns([0.92, 1.35], gap="large")

    with left:
        with st.container(border=True):
            section_heading(
                "CUSTOMER OUTCOMES",
                "Subscription distribution",
            )

            target_counts = (
                df["y"].value_counts()
                .rename_axis("Outcome")
                .reset_index(name="Customers")
            )

            fig = px.pie(
                target_counts,
                names="Outcome",
                values="Customers",
                hole=0.68,
                color="Outcome",
                color_discrete_map={
                    "yes": "#6EE7B7",
                    "no": "#60A5FA",
                },
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent",
                marker=dict(
                    line=dict(
                        color="rgba(8,13,26,0.85)",
                        width=3,
                    )
                ),
            )

            style_chart(fig, 325)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    with right:
        with st.container(border=True):
            section_heading(
                "MODEL INTELLIGENCE",
                "Performance comparison",
                "Scores range from 0 to 1; higher is better for each displayed metric.",
            )

            if not comparison.empty:
                chart = comparison.melt(
                    id_vars="model",
                    value_vars=[
                        "accuracy", "precision_yes",
                        "recall_yes", "f1_yes", "roc_auc",
                    ],
                    var_name="Metric",
                    value_name="Score",
                )

                chart["Metric"] = chart["Metric"].map({
                    "accuracy": "Accuracy",
                    "precision_yes": "Precision",
                    "recall_yes": "Recall",
                    "f1_yes": "F1 Score",
                    "roc_auc": "ROC-AUC",
                })

                fig = px.bar(
                    chart,
                    x="Metric",
                    y="Score",
                    color="model",
                    barmode="group",
                    color_discrete_sequence=[
                        "#60A5FA", "#A78BFA", "#67E8F9"
                    ],
                )
                fig.update_yaxes(range=[0, 1], tickformat=".0%")
                style_chart(fig, 325)

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.info(
                    "No valid model comparison report was found. "
                    "Run the model comparison script to generate it."
                )

    st.markdown(
        """
        <div class="note">
            <strong>Responsible use:</strong> Historical patterns and
            model predictions are estimates, not guarantees of customer
            behavior. Use them as decision support, not as the sole basis
            for decisions about individuals.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 2: CUSTOMER PREDICTION
# ============================================================

elif page == "Customer Prediction":
    section_heading(
        "PREDICTION STUDIO",
        "Customer subscription prediction",
        "Complete the customer profile. Fields are prefilled with sensible "
        "defaults and can be changed before prediction.",
    )

    if comparison.empty:
        st.error(
            "The model comparison report is missing or invalid. "
            "Run your model comparison script first."
        )
        st.stop()

    best = comparison.sort_values(
        "f1_yes", ascending=False
    ).iloc[0]

    model_name = str(best["model"])
    model_path = MODELS_DIR / model_filename(model_name)

    if not model_path.exists():
        st.error(f"Trained model file not found: {model_path}")
        st.stop()

    st.markdown(
        f"""
        <div style="margin:8px 0 20px 0;">
            <span class="status-pill">SELECTED MODEL</span>
            <span style="margin-left:10px;font-weight:700;color:#E2E8F0;">
                {model_name}
            </span>
            <span style="margin-left:10px;color:#94A3B8;font-size:12px;">
                Positive-class F1: {best["f1_yes"]:.4f}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_balance = int(pd.to_numeric(
        df["balance"], errors="coerce"
    ).median())

    with st.form("prediction_form"):
        st.markdown(
            '<div class="section-label">01 / PERSONAL PROFILE</div>',
            unsafe_allow_html=True,
        )

        a, b, c = st.columns(3, gap="large")

        with a:
            age = st.number_input("Age", 18, 100, 35)

            job = st.selectbox(
                "Job",
                [
                    "admin.", "blue-collar", "entrepreneur",
                    "housemaid", "management", "retired",
                    "self-employed", "services", "student",
                    "technician", "unemployed", "unknown",
                ],
            )

            marital = st.selectbox(
                "Marital Status",
                ["married", "single", "divorced"],
            )

            education = st.selectbox(
                "Education",
                ["primary", "secondary", "tertiary", "unknown"],
            )

        with b:
            balance = st.number_input(
                "Account Balance",
                value=default_balance,
                help="Enter the customer's recorded account balance.",
            )

            housing = st.selectbox("Housing Loan", ["yes", "no"])
            loan = st.selectbox("Personal Loan", ["yes", "no"])

            contact = st.selectbox(
                "Contact Type",
                ["cellular", "telephone", "unknown"],
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
                "Contacts During This Campaign",
                1, 100, 1,
            )

            pdays = st.number_input(
                "Days Since Previous Contact",
                -1, 999, -1,
                help="-1 means the customer was not previously contacted.",
            )

            previous = st.number_input(
                "Previous Campaign Contacts",
                0, 100, 0,
            )

            poutcome = st.selectbox(
                "Previous Campaign Outcome",
                ["unknown", "failure", "other", "success"],
            )

        st.markdown("---")
        submitted = st.form_submit_button(
            "Generate Prediction",
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
            probabilities = model.predict_proba(input_data)[0]

            # Robustly identify the positive subscription class.
            # Supports numeric 0/1 and text "no"/"yes" class labels.
            classes = list(model.classes_)

            yes_index = next(
                (
                    index
                    for index, class_value in enumerate(classes)
                    if str(class_value).strip().lower()
                    in {"1", "yes", "true"}
                ),
                None,
            )

            if yes_index is None:
                raise ValueError(
                    "Could not identify the subscription class. "
                    f"Model classes: {classes}"
                )

            probability = float(probabilities[yes_index])

            # Preserve the existing 0.5 classification threshold.
            prediction = int(probability >= 0.5)

            st.markdown("")
            section_heading(
                "RESULT",
                "Prediction summary",
                "The probability and classification are generated by the selected trained model.",
            )

            left, right = st.columns([1, 1.15], gap="large")

            with left:
                with st.container(border=True):
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=probability * 100,
                            number={
                                "suffix": "%",
                                "font": {
                                    "size": 38,
                                    "color": "#F8FAFC",
                                },
                            },
                            title={
                                "text": "Estimated probability",
                                "font": {
                                    "size": 15,
                                    "color": "#CBD5E1",
                                },
                            },
                            gauge={
                                "axis": {
                                    "range": [0, 100],
                                    "ticksuffix": "%",
                                    "tickcolor": "#94A3B8",
                                },
                                "bar": {"color": "#60A5FA"},
                                "bgcolor": "rgba(0,0,0,0)",
                                "borderwidth": 0,
                                "steps": [
                                    {
                                        "range": [0, 50],
                                        "color": "rgba(96,165,250,0.10)",
                                    },
                                    {
                                        "range": [50, 100],
                                        "color": "rgba(167,139,250,0.14)",
                                    },
                                ],
                                "threshold": {
                                    "line": {
                                        "color": "#67E8F9",
                                        "width": 3,
                                    },
                                    "thickness": 0.8,
                                    "value": 50,
                                },
                            },
                        )
                    )

                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={"color": "#F1F5F9"},
                        height=280,
                        margin=dict(l=20, r=20, t=45, b=10),
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

            with right:
                label = (
                    "Likely to Subscribe"
                    if prediction
                    else "Less Likely to Subscribe"
                )

                color = GREEN if prediction else BLUE

                st.markdown(
                    f"""
                    <div class="result-panel">
                        <div class="section-label">MODEL CLASSIFICATION</div>
                        <h2 style="color:{color} !important;
                            font-size:27px;margin:8px 0 14px 0;">
                            {label}
                        </h2>
                        <p style="color:#CBD5E1;line-height:1.8;">
                            Estimated subscription probability:
                            <strong>{probability:.1%}</strong>
                        </p>
                        <div class="metric-foot">
                            Classification threshold: 50%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("")
                st.markdown(
                    """
                    <div class="note">
                        <strong>How to interpret this:</strong> The
                        probability is a model estimate based on the
                        supplied customer features. It is not a guarantee,
                        a causal explanation, or a substitute for human
                        judgment.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.download_button(
                    "Download Prediction Details",
                    data=pd.DataFrame([{
                        "model": model_name,
                        "estimated_subscription_probability": probability,
                        "classification": label,
                        "threshold": 0.5,
                    }]).to_csv(index=False).encode("utf-8"),
                    file_name="bankwise_prediction.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ============================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":
    section_heading(
        "MODEL LAB",
        "Model performance and evaluation",
        "Review the saved evaluation results from your model comparison.",
    )

    if comparison.empty:
        st.warning(
            "No valid model comparison results were found. "
            "Run the model comparison script first."
        )
        st.stop()

    metric_columns = [
        "accuracy", "precision_yes", "recall_yes", "f1_yes", "roc_auc"
    ]

    best = comparison.sort_values(
        "f1_yes", ascending=False
    ).iloc[0]

    metric_labels = [
        ("ACCURACY", "accuracy", BLUE),
        ("PRECISION", "precision_yes", VIOLET),
        ("RECALL", "recall_yes", CYAN),
        ("F1 SCORE", "f1_yes", GREEN),
        ("ROC-AUC", "roc_auc", "#F9A8D4"),
    ]

    metric_cols = st.columns(5, gap="small")

    for column, (label, key, accent) in zip(
        metric_cols, metric_labels
    ):
        with column:
            glass_metric(
                label,
                f"{best[key]:.3f}",
                str(best["model"]),
                accent,
            )

    st.markdown("")
    with st.container(border=True):
        section_heading(
            "BENCHMARK",
            "Compare all trained models",
            "Metrics shown here come directly from model_comparison.json.",
        )

        st.dataframe(
            comparison[["model"] + metric_columns].style.format(
                {column: "{:.4f}" for column in metric_columns}
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.container(border=True):
        section_heading(
            "VISUAL ANALYTICS",
            "Evaluation metrics by model",
        )

        chart_data = comparison.melt(
            id_vars="model",
            value_vars=metric_columns,
            var_name="Metric",
            value_name="Score",
        )

        chart_data["Metric"] = chart_data["Metric"].map({
            "accuracy": "Accuracy",
            "precision_yes": "Precision",
            "recall_yes": "Recall",
            "f1_yes": "F1 Score",
            "roc_auc": "ROC-AUC",
        })

        fig = px.bar(
            chart_data,
            x="Metric",
            y="Score",
            color="model",
            barmode="group",
            color_discrete_sequence=[
                "#60A5FA", "#A78BFA", "#67E8F9"
            ],
            hover_data={"Score": ":.4f"},
        )

        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        style_chart(fig, 390)

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    matrix_path = FIGURES_DIR / "confusion_matrix.png"

    if matrix_path.exists():
        with st.container(border=True):
            section_heading(
                "CLASSIFICATION DIAGNOSTICS",
                "Saved confusion matrix",
                "This image is loaded from your existing evaluation output.",
            )
            st.image(str(matrix_path), use_container_width=True)

    report_path = REPORTS_DIR / "classification_report.txt"

    if report_path.exists():
        with st.expander("View classification report"):
            with report_path.open("r", encoding="utf-8") as file:
                st.code(file.read(), language="text")

    st.caption(
        "Metrics depend on the saved evaluation run and test split. "
        "Compare models using consistent evaluation data."
    )


# ============================================================
# PAGE 4: DATASET EXPLORER
# ============================================================

elif page == "Dataset Explorer":
    section_heading(
        "DATA WORKSPACE",
        "Dataset explorer",
        "Search and filter the original dataset without modifying the source file.",
    )

    total_records = len(df)
    total_columns = len(df.columns)
    missing_cells = int(df.isna().sum().sum())

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        glass_metric(
            "TOTAL ROWS", f"{total_records:,}",
            "Available dataset records", BLUE
        )

    with c2:
        glass_metric(
            "FEATURE COLUMNS", f"{total_columns:,}",
            "Includes the target column", VIOLET
        )

    with c3:
        glass_metric(
            "MISSING CELLS", f"{missing_cells:,}",
            "Null values in loaded data", CYAN
        )

    st.markdown("")

    with st.container(border=True):
        section_heading(
            "FILTERS",
            "Find customer records",
            "Choose a column and value. Select All to view every record.",
        )

        filter_left, filter_right = st.columns(2, gap="large")

        with filter_left:
            search_column = st.selectbox(
                "Filter by column",
                df.columns.tolist(),
            )

        values = sorted(
            df[search_column].dropna().astype(str).unique().tolist()
        )

        with filter_right:
            selected_value = st.selectbox(
                "Choose a value",
                ["All"] + values,
            )

    filtered = df

    if selected_value != "All":
        filtered = df[
            df[search_column].astype(str) == selected_value
        ]

    st.markdown("")
    st.caption(
        f"Showing {len(filtered):,} matching records "
        f"out of {len(df):,} total records."
    )

    with st.container(border=True):
        st.dataframe(
            filtered.head(1000),
            use_container_width=True,
            hide_index=True,
            height=420,
        )

        if len(filtered) > 1000:
            st.info(
                "The table displays the first 1,000 matching rows. "
                "The CSV download includes all matching rows."
            )

        st.download_button(
            "Export Filtered Data (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="bankwise_filtered_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("Explore dataset statistics"):
        st.dataframe(
            df.describe(include="all").transpose(),
            use_container_width=True,
        )

    with st.expander("Inspect missing values by column"):
        missing_summary = (
            df.isna().sum()
            .rename("Missing Values")
            .to_frame()
        )
        missing_summary["Missing Percentage"] = (
            missing_summary["Missing Values"] / len(df) * 100
        )
        st.dataframe(
            missing_summary,
            use_container_width=True,
        )


# ============================================================
# PAGE 5: REPORTS
# ============================================================

elif page == "Reports":
    section_heading(
        "REPORT CENTER",
        "Reports and project artifacts",
        "View and download generated outputs from your existing project.",
    )

    files = [
        REPORTS_DIR / "baseline_metrics.json",
        REPORTS_DIR / "model_comparison.json",
        REPORTS_DIR / "model_comparison.csv",
        REPORTS_DIR / "classification_report.txt",
        FIGURES_DIR / "model_comparison.png",
        FIGURES_DIR / "confusion_matrix.png",
    ]

    available = [path for path in files if path.exists()]
    missing = [path for path in files if not path.exists()]

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        glass_metric(
            "AVAILABLE ARTIFACTS",
            str(len(available)),
            "Files ready to view or download",
            GREEN,
        )

    with c2:
        glass_metric(
            "NOT GENERATED",
            str(len(missing)),
            "Expected files currently missing",
            VIOLET,
        )

    st.markdown("")

    for index, file_path in enumerate(files):
        if not file_path.exists():
            continue

        with st.container(border=True):
            left, right = st.columns([3, 1], gap="medium")

            with left:
                suffix_label = file_path.suffix.upper().replace(".", "")
                st.markdown(
                    f"**{file_path.stem.replace('_', ' ').title()}**"
                )
                st.caption(
                    f"{suffix_label} file · {file_path.parent.name}"
                )

                if file_path.suffix.lower() == ".png":
                    with st.expander("Preview image"):
                        st.image(
                            str(file_path),
                            use_container_width=True,
                        )

                elif file_path.suffix.lower() == ".txt":
                    with st.expander("Preview report"):
                        with file_path.open(
                            "r", encoding="utf-8"
                        ) as file:
                            st.code(file.read(), language="text")

            with right:
                st.download_button(
                    "Download",
                    data=file_path.read_bytes(),
                    file_name=file_path.name,
                    key=f"download_{index}_{file_path.name}",
                    use_container_width=True,
                )

    if missing:
        with st.expander("Expected files not generated yet"):
            for file_path in missing:
                st.caption(f"Not available: {file_path.name}")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <strong style="color:#A5B4CC;">BANKWISE AI</strong><br>
        Machine Learning · Customer Intelligence · Responsible Analytics<br>
        Academic project — predictions are estimates, not guarantees.
    </div>
    """,
    unsafe_allow_html=True,
)
