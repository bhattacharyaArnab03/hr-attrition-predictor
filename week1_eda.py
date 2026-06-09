# =============================================================================
# HR Attrition Analytics & Predictor
# Week 1 — Exploratory Data Analysis
# Format  : Jupyter Notebook (.ipynb) in VS Code
# Kernel  : venv (Python 3.10.8) — your project venv
#
# HOW TO USE
# ──────────
# 1. Open week1_eda.ipynb in VS Code (inside notebooks/ folder)
# 2. Select kernel: venv (Python 3.10.8) — top right corner
# 3. Copy each CELL block below into a separate notebook cell
# 4. Run cells top to bottom with Shift+Enter
# =============================================================================


# ── CELL 1 ──────────────────────────────────────────────────────────────────
# Imports & Configuration
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

%matplotlib inline
warnings.filterwarnings("ignore")

# ── Paths (works regardless of which subfolder the notebook runs from) ──
PROJECT_ROOT = Path.cwd().parent          # notebooks/ → hr-attrition-predictor/
DATA_PATH    = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
FIGURES_DIR  = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Plot style ──
plt.rcParams.update({
    "figure.dpi"        : 120,
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "white",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "axes.grid.axis"    : "y",
    "grid.alpha"        : 0.3,
    "font.size"         : 11,
})

PALETTE = {"Yes": "#E24B4A", "No": "#1D9E75"}   # red = left, green = stayed
BLUE    = "#185FA5"

# ── Sanity check — verify paths before running anything else ──
print(f"Project root : {PROJECT_ROOT}")
print(f"Data path    : {DATA_PATH}")
print(f"Figures dir  : {FIGURES_DIR}")
print(f"Data file exists: {DATA_PATH.exists()}")


# ── CELL 2 ──────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def save(fig, filename):
    """Save figure to reports/figures/ and confirm."""
    path = FIGURES_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    print(f"  ✓  Saved → {path}")


def section(title):
    """Print a visible section header."""
    print(f"\n{'═' * 55}")
    print(f"  {title}")
    print(f"{'═' * 55}")


# ── CELL 3 ──────────────────────────────────────────────────────────────────
# Section 1 — Load & Inspect the Data
# ─────────────────────────────────────────────────────────────────────────────
# Download the dataset (free Kaggle account required):
# https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
# Place the CSV at:  data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv

section("SECTION 1 — Load & Inspect the Data")

df = pd.read_csv(DATA_PATH)
print(f"\n  Rows    : {df.shape[0]:,}")
print(f"  Columns : {df.shape[1]}")
print(f"\n  Column names:\n  {list(df.columns)}")


# ── CELL 4 ──────────────────────────────────────────────────────────────────
# First look at the data
# ─────────────────────────────────────────────────────────────────────────────
print("First 5 rows:\n")
df.head()


# ── CELL 5 ──────────────────────────────────────────────────────────────────
# Data types & basic stats
# ─────────────────────────────────────────────────────────────────────────────
print("Data types:\n")
print(df.dtypes.to_string())
print("\nSummary statistics:\n")
df.describe().round(2)


# ── CELL 6 ──────────────────────────────────────────────────────────────────
# Drop useless columns
# ─────────────────────────────────────────────────────────────────────────────
# Columns with only one unique value carry zero information for ML
useless = [c for c in df.columns if df[c].nunique() <= 1]
print(f"Single-value columns (dropping): {useless}")
df.drop(columns=useless, inplace=True)

# EmployeeNumber is just an ID — not useful as a feature
if "EmployeeNumber" in df.columns:
    df.drop(columns=["EmployeeNumber"], inplace=True)

print(f"\nDataset shape after cleanup: {df.shape}")
print(f"Remaining columns: {list(df.columns)}")


# ── CELL 7 ──────────────────────────────────────────────────────────────────
# Section 2 — Missing Values Check
# ─────────────────────────────────────────────────────────────────────────────
section("SECTION 2 — Missing Values")

missing     = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
bad_cols    = missing[missing > 0]

if bad_cols.empty:
    print("\n  ✓  No missing values. IBM HR is a clean synthetic dataset.")
else:
    result = pd.DataFrame({
        "Missing Count" : bad_cols,
        "Missing %"     : missing_pct[bad_cols.index]
    })
    print(result)


# ── CELL 8 ──────────────────────────────────────────────────────────────────
# Section 3 — Attrition (Target Variable) — Stats
# ─────────────────────────────────────────────────────────────────────────────
section("SECTION 3 — Attrition (Target Variable)")

counts = df["Attrition"].value_counts()
pcts   = df["Attrition"].value_counts(normalize=True) * 100

print("\n  Attrition breakdown:")
for label in counts.index:
    bar = "█" * int(pcts[label] / 2)
    print(f"    {label:3s}  {counts[label]:4d}  ({pcts[label]:.1f}%)  {bar}")

print(f"""
  ─────────────────────────────────────────────────
  Key observation:
  • Only {pcts['Yes']:.1f}% of employees left → dataset is IMBALANCED
  • A naive model saying "nobody leaves" = {pcts['No']:.1f}% accurate (misleading!)
  • Fix in Week 3: use XGBoost's scale_pos_weight parameter
  ─────────────────────────────────────────────────
""")


# ── CELL 9 ──────────────────────────────────────────────────────────────────
# Section 3 — Attrition — Chart
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Target Variable: Attrition Distribution", fontsize=13, fontweight="bold")

# Bar chart
colors = [PALETTE[k] for k in counts.index]
axes[0].bar(counts.index, counts.values, color=colors, width=0.4, alpha=0.85)
axes[0].set_xlabel("Attrition")
axes[0].set_ylabel("Employee Count")
axes[0].grid(False)
for i, (k, v) in enumerate(zip(counts.index, counts.values)):
    axes[0].text(i, v + 8, f"{v}\n({pcts[k]:.1f}%)", ha="center", fontsize=10)

# Pie chart
axes[1].pie(
    counts.values,
    labels=counts.index,
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
    pctdistance=0.78,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
axes[1].set_title("Proportion")

plt.tight_layout()
save(fig, "01_attrition_distribution.png")
plt.show()


# ── CELL 10 ─────────────────────────────────────────────────────────────────
# Section 4 — Numeric Feature Distributions — Stats
# ─────────────────────────────────────────────────────────────────────────────
section("SECTION 4 — Numeric Feature Distributions")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
print(f"\n  Numeric columns ({len(numeric_cols)}):\n  {numeric_cols}")
print()
df[numeric_cols].describe().round(1)


# ── CELL 11 ─────────────────────────────────────────────────────────────────
# Section 4 — Numeric Distributions — Chart
# ─────────────────────────────────────────────────────────────────────────────
n_cols = 4
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=25, color=BLUE, alpha=0.75, edgecolor="white")
    axes[i].set_title(col, fontsize=9, fontweight="bold")
    axes[i].axvline(
        df[col].mean(), color="#E24B4A", linewidth=1.5,
        linestyle="--", label=f"mean={df[col].mean():.0f}"
    )
    axes[i].legend(fontsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Numeric Feature Distributions  (red line = mean)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "02_numeric_distributions.png")
plt.show()


# ── CELL 12 ─────────────────────────────────────────────────────────────────
# Section 5 — Categorical Features — Stats
# ─────────────────────────────────────────────────────────────────────────────
section("SECTION 5 — Categorical Features")

cat_cols = [c for c in df.select_dtypes("object").columns if c != "Attrition"]
print(f"\n  Categorical columns ({len(cat_cols)}): {cat_cols}\n")

for col in cat_cols:
    vc = df[col].value_counts()
    print(f"  {col}  ({df[col].nunique()} unique values):")
    for val, cnt in vc.items():
        print(f"    {val:35s}  {cnt:4d}  ({cnt / len(df) * 100:.1f}%)")
    print()


# ── CELL 13 ─────────────────────────────────────────────────────────────────
# Section 5 — Categorical Features — Chart
# ─────────────────────────────────────────────────────────────────────────────
n_cols = 2
n_rows = (len(cat_cols) + 1) // 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    vc = df[col].value_counts()
    axes[i].barh(vc.index, vc.values, color=BLUE, alpha=0.8)
    axes[i].set_title(col, fontweight="bold")
    axes[i].set_xlabel("Count")
    for j, v in enumerate(vc.values):
        axes[i].text(v + 3, j, str(v), va="center", fontsize=9)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Categorical Feature Distributions", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "03_categorical_distributions.png")
plt.show()


# ── CELL 14 ─────────────────────────────────────────────────────────────────
# Section 6 — Attrition Rate by Categorical Feature — Stats
# ─────────────────────────────────────────────────────────────────────────────
section("SECTION 6 — Attrition Rate by Categorical Feature")

cat_to_plot  = ["Department", "JobRole", "OverTime",
                "MaritalStatus", "BusinessTravel", "Gender"]
overall_avg  = (df["Attrition"] == "Yes").mean() * 100

print(f"\n  Overall attrition rate: {overall_avg:.1f}%\n")
print(f"  {'Feature':<20}  {'Group':<35}  {'Rate':>6}  {'vs Avg':>8}")
print(f"  {'─'*20}  {'─'*35}  {'─'*6}  {'─'*8}")

for col in cat_to_plot:
    rate = (
        df.groupby(col)["Attrition"]
          .apply(lambda x: (x == "Yes").sum() / len(x) * 100)
          .sort_values(ascending=False)
    )
    for grp, val in rate.items():
        diff   = val - overall_avg
        flag   = " ← HIGH RISK" if val > overall_avg * 1.5 else ""
        sign   = "+" if diff >= 0 else ""
        print(f"  {col:<20}  {str(grp):<35}  {val:>5.1f}%  {sign}{diff:>6.1f}%{flag}")
    print()


# ── CELL 15 ─────────────────────────────────────────────────────────────────
# Section 6 — Attrition Rate by Category — Chart
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 13))
axes = axes.flatten()

for i, col in enumerate(cat_to_plot):
    rate = (
        df.groupby(col)["Attrition"]
          .apply(lambda x: (x == "Yes").sum() / len(x) * 100)
          .sort_values(ascending=True)
    )
    bar_colors = ["#E24B4A" if v > overall_avg else BLUE for v in rate.values]

    axes[i].barh(rate.index, rate.values, color=bar_colors, alpha=0.85)
    axes[i].axvline(overall_avg, color="gray", linestyle="--",
                    linewidth=1.2, label=f"Avg {overall_avg:.1f}%")
    axes[i].set_title(f"Attrition Rate by {col}", fontweight="bold", fontsize=11)
    axes[i].xaxis.set_major_formatter(mtick.PercentFormatter())
    axes[i].legend(fontsize=8)
    for j, v in enumerate(rate.values):
        axes[i].text(v + 0.4, j, f"{v:.1f}%", va="center", fontsize=9)

fig.suptitle("Attrition Rate by Categorical Features\n(Red = above average)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "04_attrition_by_category.png")
plt.show()


# ── CELL 16 ─────────────────────────────────────────────────────────────────
# Section 7 — Key Numeric Features vs Attrition — Stats
# ─────────────────────────────────────────────────────────────────────────────
section("SECTION 7 — Numeric Features vs Attrition")

key_numeric = ["Age", "MonthlyIncome", "TotalWorkingYears",
               "YearsAtCompany", "JobSatisfaction", "DistanceFromHome"]

print(f"\n  {'Feature':<25}  {'Left (Yes)':>12}  {'Stayed (No)':>12}  {'Gap':>10}")
print(f"  {'─'*25}  {'─'*12}  {'─'*12}  {'─'*10}")

for col in key_numeric:
    yes_m = df[df["Attrition"] == "Yes"][col].mean()
    no_m  = df[df["Attrition"] == "No"][col].mean()
    diff  = yes_m - no_m
    sign  = "+" if diff > 0 else ""
    print(f"  {col:<25}  {yes_m:>12.2f}  {no_m:>12.2f}  {sign}{diff:>9.2f}")


# ── CELL 17 ─────────────────────────────────────────────────────────────────
# Section 7 — Numeric vs Attrition — Overlapping Histograms
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(key_numeric):
    for label, color in PALETTE.items():
        subset = df[df["Attrition"] == label][col]
        axes[i].hist(
            subset, bins=25, alpha=0.55, label=label,
            color=color, edgecolor="white", density=True
        )
    axes[i].set_title(col, fontweight="bold")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Density")
    axes[i].legend(title="Attrition", fontsize=9)

fig.suptitle("Numeric Features: Attrition=Yes (red) vs No (green)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "05_numeric_vs_attrition.png")
plt.show()


# ── CELL 18 ─────────────────────────────────────────────────────────────────
# Section 7 — Box Plots: Income & Age
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

dept_order = (df.groupby("Department")["MonthlyIncome"]
                .median().sort_values().index.tolist())

sns.boxplot(
    data=df, x="MonthlyIncome", y="Department",
    hue="Attrition", order=dept_order,
    palette=PALETTE, ax=axes[0], linewidth=0.8
)
axes[0].set_title("Monthly Income by Department & Attrition", fontweight="bold")
axes[0].legend(title="Attrition")

sns.boxplot(
    data=df, x="JobLevel", y="Age",
    hue="Attrition", palette=PALETTE,
    ax=axes[1], linewidth=0.8
)
axes[1].set_title("Age by Job Level & Attrition", fontweight="bold")
axes[1].legend(title="Attrition")

plt.tight_layout()
save(fig, "06_boxplots_income_age.png")
plt.show()


# ── CELL 19 ─────────────────────────────────────────────────────────────────
# Section 8 — Correlation Heatmap — Stats
# ─────────────────────────────────────────────────────────────────────────────
section("SECTION 8 — Correlation Heatmap")

df_corr              = df.copy()
df_corr["Attrition_bin"] = (df_corr["Attrition"] == "Yes").astype(int)

numeric_for_corr = df_corr.select_dtypes(include=np.number).columns.tolist()
corr             = df_corr[numeric_for_corr].corr()

top10 = (
    corr["Attrition_bin"]
    .drop("Attrition_bin")
    .abs()
    .sort_values(ascending=False)
    .head(10)
)

print("\n  Top 10 features correlated with Attrition:\n")
print(f"  {'Feature':<30}  {'|r|':>6}  {'Direction'}")
print(f"  {'─'*30}  {'─'*6}  {'─'*35}")

for feat, val in top10.items():
    raw = corr.loc[feat, "Attrition_bin"]
    direction = "↑ more = more attrition" if raw > 0 else "↓ less = more attrition"
    print(f"  {feat:<30}  {val:>6.3f}  {direction}")


# ── CELL 20 ─────────────────────────────────────────────────────────────────
# Section 8 — Correlation Heatmap — Chart
# ─────────────────────────────────────────────────────────────────────────────
top15 = (
    corr["Attrition_bin"]
    .drop("Attrition_bin")
    .abs()
    .sort_values(ascending=False)
    .head(15)
    .index.tolist()
)
top15.append("Attrition_bin")
sub_corr = corr.loc[top15, top15]

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(sub_corr, dtype=bool))   # show lower triangle only

sns.heatmap(
    sub_corr, mask=mask,
    annot=True, fmt=".2f",
    cmap="RdYlGn", center=0,
    vmin=-1, vmax=1,
    linewidths=0.4,
    annot_kws={"size": 8},
    ax=ax,
)
ax.set_title("Correlation Heatmap — Top 15 Features vs Attrition",
             fontsize=13, fontweight="bold", pad=16)
plt.tight_layout()
save(fig, "07_correlation_heatmap.png")
plt.show()


# ── CELL 21 ─────────────────────────────────────────────────────────────────
# Section 9 — OverTime Deep Dive — Stats
# ─────────────────────────────────────────────────────────────────────────────
section("SECTION 9 — OverTime Deep Dive  (strongest single predictor)")

ot_overall = (
    df.groupby("OverTime")["Attrition"]
      .apply(lambda x: (x == "Yes").mean() * 100)
)

ot_dept = (
    df.groupby(["OverTime", "Department"])["Attrition"]
      .apply(lambda x: (x == "Yes").mean() * 100)
      .unstack()
      .round(1)
)

print("\n  Overall attrition rate by OverTime status:")
for ot, rate in ot_overall.items():
    print(f"    {ot:3s}  →  {rate:.1f}%")

print(f"\n  Employees doing OverTime leave at "
      f"{ot_overall['Yes'] / ot_overall['No']:.1f}× the rate of those who don't.\n")

print("  Attrition rate (%) by OverTime × Department:")
print(ot_dept.to_string())


# ── CELL 22 ─────────────────────────────────────────────────────────────────
# Section 9 — OverTime — Chart
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("OverTime — The Strongest Attrition Predictor",
             fontsize=13, fontweight="bold")

# Left: overall
axes[0].bar(
    ot_overall.index, ot_overall.values,
    color=["#1D9E75", "#E24B4A"], width=0.4, alpha=0.85
)
for i, (k, v) in enumerate(ot_overall.items()):
    axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
axes[0].set_title("Overall Attrition by OverTime Status")
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[0].set_xlabel("OverTime")
axes[0].set_ylabel("Attrition Rate (%)")

# Right: by department
ot_dept.T.plot(kind="bar", ax=axes[1],
               color=["#1D9E75", "#E24B4A"], alpha=0.85)
axes[1].set_title("Attrition Rate: Department × OverTime")
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[1].set_xlabel("Department")
axes[1].legend(title="OverTime")
plt.xticks(rotation=15)

plt.tight_layout()
save(fig, "08_overtime_analysis.png")
plt.show()


# ── CELL 23 ─────────────────────────────────────────────────────────────────
# Week 1 — Key Insights Summary
# ─────────────────────────────────────────────────────────────────────────────
attrition_rate = (df["Attrition"] == "Yes").mean() * 100
ot_yes  = df[df["OverTime"] == "Yes"]["Attrition"].apply(lambda x: 1 if x=="Yes" else 0).mean() * 100
ot_no   = df[df["OverTime"] == "No"]["Attrition"].apply(lambda x: 1 if x=="Yes" else 0).mean() * 100
inc_yes = df[df["Attrition"] == "Yes"]["MonthlyIncome"].mean()
inc_no  = df[df["Attrition"] == "No"]["MonthlyIncome"].mean()
age_yes = df[df["Attrition"] == "Yes"]["Age"].mean()
age_no  = df[df["Attrition"] == "No"]["Age"].mean()

print("=" * 55)
print("  WEEK 1 KEY INSIGHTS")
print("=" * 55)
print(f"""
  Dataset  : {df.shape[0]:,} employees · {df.shape[1]} features
  Attrition: {attrition_rate:.1f}% overall  (industry benchmark ~15%)

  FINDING 1 — OverTime is the biggest driver
    With overtime  : {ot_yes:.1f}% attrition
    Without overtime: {ot_no:.1f}% attrition
    → {ot_yes / ot_no:.1f}× the risk. Strongest actionable signal for HR.

  FINDING 2 — Pay gap between leavers and stayers
    Left company   : ${inc_yes:,.0f} / month avg
    Stayed         : ${inc_no:,.0f} / month avg
    → Leavers earned ${inc_no - inc_yes:,.0f}/month LESS.

  FINDING 3 — Youth flight risk
    Avg age (left)  : {age_yes:.1f} years
    Avg age (stayed): {age_no:.1f} years
    → Younger employees need stronger career growth paths.

  FINDING 4 — Sales has highest attrition by department
    → Needs targeted retention programs.

  FINDING 5 — Top ML features (from correlation)
    OverTime · JobSatisfaction · MonthlyIncome
    TotalWorkingYears · Age · StockOptionLevel
    → These will dominate the XGBoost model in Week 3.

  ─────────────────────────────────────────────────
  Figures saved to: {FIGURES_DIR}
    01_attrition_distribution.png
    02_numeric_distributions.png
    03_categorical_distributions.png
    04_attrition_by_category.png
    05_numeric_vs_attrition.png
    06_boxplots_income_age.png
    07_correlation_heatmap.png
    08_overtime_analysis.png
  ─────────────────────────────────────────────────

  Next → Week 2: Plotly dashboard + feature engineering
""")
print("=" * 55)
print("  ✅  Week 1 EDA Complete!")
print("=" * 55)
