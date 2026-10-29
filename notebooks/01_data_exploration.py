
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bank-full.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the real dataset
df = pd.read_csv(DATA_PATH, sep=";")

print("=" * 60)
print("BANKWISE AI - DATA EXPLORATION")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")

print("\nFirst five rows:")
print(df.head().to_string())

print("\nColumn data types:")
print(df.dtypes.to_string())

print("\nMissing values per column:")
print(df.isnull().sum().to_string())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nTarget distribution:")
print(df["y"].value_counts())

print("\nTarget percentages:")
print((df["y"].value_counts(normalize=True) * 100).round(2))

print("\nNumerical summary:")
print(df.describe().round(2).to_string())

print("\nCategorical summary:")
print(df.describe(include="object").to_string())

# Plot target distribution
sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="y")
plt.title("Bank Marketing: Subscription Distribution")
plt.xlabel("Subscribed to Term Deposit")
plt.ylabel("Number of Customers")
plt.tight_layout()

target_plot = OUTPUT_DIR / "target_distribution.png"
plt.savefig(target_plot, dpi=150)
plt.close()

# Plot numerical feature distributions
numeric_columns = df.select_dtypes(include="number").columns

df[numeric_columns].hist(
    figsize=(14, 10),
    bins=30,
    edgecolor="black"
)

plt.suptitle("Numerical Feature Distributions", fontsize=16)
plt.tight_layout()

numeric_plot = OUTPUT_DIR / "numerical_distributions.png"
plt.savefig(numeric_plot, dpi=150)
plt.close()

# Save a concise report
report_path = PROJECT_ROOT / "reports" / "data_quality_report.txt"

with report_path.open("w", encoding="utf-8") as report:
    report.write("BANKWISE AI - DATA QUALITY REPORT\n")
    report.write("=" * 40 + "\n")
    report.write(f"Dataset shape: {df.shape}\n")
    report.write(f"Total missing values: {df.isnull().sum().sum()}\n")
    report.write(f"Duplicate rows: {df.duplicated().sum()}\n\n")
    report.write("Column data types:\n")
    report.write(df.dtypes.to_string())
    report.write("\n\nTarget distribution:\n")
    report.write(df["y"].value_counts().to_string())
    report.write("\n\nTarget percentages:\n")
    report.write(
        (df["y"].value_counts(normalize=True) * 100)
        .round(2)
        .to_string()
    )

print("\nExploration completed successfully.")
print(f"Target chart: {target_plot}")
print(f"Numerical charts: {numeric_plot}")
print(f"Quality report: {report_path}")
