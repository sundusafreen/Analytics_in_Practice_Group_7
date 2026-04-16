# -*- coding: utf-8 -*-
"""ML1_EDA_LogisticRegression_FIXED.ipynb

ML1 Role: EDA + Logistic Regression (Baseline Model)
FIXED VERSION — includes country, gender, preferred_category, age
Dataset: E-Commerce Customer Insights and Churn
"""

# ============================================================
# CELL 1 — Imports
# ============================================================
# !pip install imbalanced-learn  # uncomment if needed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

print("All imports OK")


# ============================================================
# CELL 2 — Load & Data Engineering (with FIX)
# ============================================================
df = pd.read_csv("E Commerce Customer Insights and Churn Dataset.csv")

# date conversion
for col in ["signup_date", "last_purchase_date", "order_date"]:
    df[col] = pd.to_datetime(df[col])

# clean
df = df.drop_duplicates()
for col in ["country", "gender", "subscription_status", "preferred_category"]:
    df[col] = df[col].str.strip().str.lower()

# feature engineering
reference_date = df["last_purchase_date"].max()
df["customer_tenure_days"]     = (reference_date - df["signup_date"]).dt.days
df["days_since_last_purchase"] = (reference_date - df["last_purchase_date"]).dt.days

# ── FIX: include categorical columns in aggregation ──
customer_df = df.groupby("customer_id").agg({
    "customer_tenure_days":     "max",
    "days_since_last_purchase": "min",
    "quantity":                 "sum",
    "unit_price":               "mean",
    "cancellations_count":      "sum",
    "purchase_frequency":       "mean",
    "subscription_status":      "last",
    "country":                  "last",   # added
    "gender":                   "last",   # added
    "preferred_category":       "last",   # added
    "age":                      "mean",   # added
}).reset_index()

customer_df["total_spent"] = df.groupby("customer_id").apply(
    lambda x: (x["quantity"] * x["unit_price"]).sum()
).values

# target variable
customer_df["churn"] = customer_df["subscription_status"].apply(
    lambda x: 1 if x == "cancelled" else 0
)
customer_df = customer_df.drop(columns=["customer_id", "subscription_status"])

# ── FIX: one-hot encode categorical columns ──
# drop_first=True removes one reference category per group,
# preventing multicollinearity in the logistic regression
customer_df = pd.get_dummies(
    customer_df,
    columns=["country", "gender", "preferred_category"],
    drop_first=True
)

print(f"Dataset shape: {customer_df.shape}")
print(f"Features: {customer_df.shape[1] - 1}  (was 7, now {customer_df.shape[1] - 1})")
print(f"\nAll columns:\n{list(customer_df.columns)}")


# ============================================================
# CELL 3 — EDA: Basic Stats
# ============================================================
print("=== Missing Values ===")
print(customer_df.isnull().sum()[customer_df.isnull().sum() > 0])
print("(none = no missing values)")

print("\n=== Summary Statistics (numerical) ===")
num_cols = ["customer_tenure_days", "days_since_last_purchase", "quantity",
            "unit_price", "cancellations_count", "purchase_frequency", "age", "total_spent"]
print(customer_df[num_cols].describe().round(2))


# ============================================================
# CELL 4 — EDA: Churn Distribution (Chart 1)
# ============================================================
churn_counts = customer_df["churn"].value_counts()
labels = ["Active / Paused (0)", "Churned (1)"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(labels, churn_counts.values,
            color=["#4C9BE8", "#E87C4C"], edgecolor="white", width=0.5)
axes[0].set_title("Churn Class Distribution", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Number of Customers")
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 10, str(v), ha="center", fontsize=12)

axes[1].pie(churn_counts.values, labels=labels, autopct="%1.1f%%",
            colors=["#4C9BE8", "#E87C4C"], startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Churn Rate", fontsize=14, fontweight="bold")

plt.suptitle("Target Variable: Churn Overview", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("chart1_churn_distribution.png", bbox_inches="tight")
plt.show()
print(f"Overall churn rate: {customer_df['churn'].mean():.1%}")


# ============================================================
# CELL 5 — EDA: Churn Rate by Country (Chart 2 — Comparison)
# ============================================================
country_cols = [c for c in customer_df.columns if c.startswith("country_")]
country_churn = {}
for col in country_cols:
    name = col.replace("country_", "").title()
    rate = customer_df[customer_df[col] == 1]["churn"].mean()
    country_churn[name] = rate

country_series = pd.Series(country_churn).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(country_series.index, country_series.values * 100,
              color="#4C9BE8", edgecolor="white", width=0.6)
ax.axhline(customer_df["churn"].mean() * 100, color="#E87C4C",
           linestyle="--", linewidth=1.5, label=f"Overall avg ({customer_df['churn'].mean():.1%})")
ax.set_xlabel("Country", fontsize=12)
ax.set_ylabel("Churn Rate (%)", fontsize=12)
ax.set_title("Churn Rate by Country", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
            f"{h:.1f}%", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("chart2_churn_by_country.png", bbox_inches="tight")
plt.show()


# ============================================================
# CELL 6 — EDA: Churn Rate by Preferred Category (Chart 3)
# ============================================================
cat_cols = [c for c in customer_df.columns if c.startswith("preferred_category_")]
cat_churn = {}
for col in cat_cols:
    name = col.replace("preferred_category_", "").title()
    rate = customer_df[customer_df[col] == 1]["churn"].mean()
    cat_churn[name] = rate

cat_series = pd.Series(cat_churn).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(cat_series.index, cat_series.values * 100,
              color="#A97DE8", edgecolor="white", width=0.6)
ax.axhline(customer_df["churn"].mean() * 100, color="#E87C4C",
           linestyle="--", linewidth=1.5, label=f"Overall avg ({customer_df['churn'].mean():.1%})")
ax.set_xlabel("Preferred Category", fontsize=12)
ax.set_ylabel("Churn Rate (%)", fontsize=12)
ax.set_title("Churn Rate by Preferred Category", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
            f"{h:.1f}%", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("chart3_churn_by_category.png", bbox_inches="tight")
plt.show()


# ============================================================
# CELL 7 — EDA: Numerical Feature Distributions (Chart 4)
# ============================================================
num_features = ["customer_tenure_days", "days_since_last_purchase",
                "cancellations_count", "purchase_frequency",
                "total_spent", "age", "quantity", "unit_price"]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for i, feat in enumerate(num_features):
    churned     = customer_df[customer_df["churn"] == 1][feat]
    not_churned = customer_df[customer_df["churn"] == 0][feat]
    axes[i].hist(not_churned, bins=30, alpha=0.6, color="#4C9BE8", label="Not Churned")
    axes[i].hist(churned,     bins=30, alpha=0.6, color="#E87C4C", label="Churned")
    axes[i].set_title(feat.replace("_", " ").title(), fontsize=11)
    axes[i].legend(fontsize=8)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")

plt.suptitle("Feature Distributions: Churned vs Not Churned", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("chart4_feature_distributions.png", bbox_inches="tight")
plt.show()


# ============================================================
# CELL 8 — EDA: Correlation Heatmap (numerical only) (Chart 5)
# ============================================================
corr_cols = num_features + ["churn"]
corr = customer_df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 10})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("chart5_correlation_heatmap.png", bbox_inches="tight")
plt.show()

print("\n=== Correlation with churn (all features, sorted) ===")
full_corr = customer_df.corr(numeric_only=True)["churn"].drop("churn").sort_values(ascending=False)
print(full_corr.round(4).to_string())


# ============================================================
# CELL 9 — Train / Test Split + SMOTE
# ============================================================
X = customer_df.drop("churn", axis=1)
y = customer_df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Before SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train_res).value_counts())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled  = scaler.transform(X_test)

print(f"\nX_train_scaled: {X_train_scaled.shape}")
print(f"X_test_scaled:  {X_test_scaled.shape}")
print(f"Features used:  {list(X.columns)}")


# ============================================================
# CELL 10 — Logistic Regression: Train
# ============================================================
lr = LogisticRegression(
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    class_weight={0: 1, 1: 3},   # penalise missing a churner 3x more than a false alarm
    random_state=42
)

lr.fit(X_train_scaled, y_train_res)
print("Logistic Regression trained.")


# ============================================================
# CELL 11 — Evaluate on Test Set
# ============================================================
y_pred      = lr.predict(X_test_scaled)
y_pred_prob = lr.predict_proba(X_test_scaled)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_prob)

print("========================================")
print("  LOGISTIC REGRESSION — TEST RESULTS")
print("  (FIXED: includes categorical features)")
print("========================================")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1 Score : {f1:.4f}")
print(f"  ROC-AUC  : {roc_auc:.4f}")
print("========================================")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Churned", "Churned"]))


# ============================================================
# CELL 12 — Confusion Matrix (Chart 6)
# ============================================================
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted: Not Churned", "Predicted: Churned"],
            yticklabels=["Actual: Not Churned",    "Actual: Churned"],
            linewidths=0.5, ax=ax)
ax.set_title("Confusion Matrix — Logistic Regression (Fixed)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("chart6_confusion_matrix_lr.png", bbox_inches="tight")
plt.show()


# ============================================================
# CELL 13 — ROC Curve (Chart 7)
# ============================================================
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#E87C4C", lw=2,
        label=f"Logistic Regression (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Baseline")
ax.fill_between(fpr, tpr, alpha=0.08, color="#E87C4C")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — Logistic Regression (Fixed)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("chart7_roc_curve_lr.png", bbox_inches="tight")
plt.show()


# ============================================================
# CELL 14 — Feature Importance: Top 15 Coefficients (Chart 8)
# ============================================================
coef_df = pd.DataFrame({
    "Feature":     X.columns,
    "Coefficient": lr.coef_[0]
}).sort_values("Coefficient", key=abs, ascending=False).head(15).sort_values("Coefficient")

colors_bar = ["#E87C4C" if c > 0 else "#4C9BE8" for c in coef_df["Coefficient"]]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(coef_df["Feature"], coef_df["Coefficient"],
               color=colors_bar, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Coefficient Value", fontsize=12)
ax.set_title("Top 15 Feature Importance — Logistic Regression\n"
             "(Orange = increases churn risk  |  Blue = reduces churn risk)",
             fontsize=12, fontweight="bold")
for bar, val in zip(bars, coef_df["Coefficient"]):
    ax.text(val + (0.005 if val >= 0 else -0.005),
            bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=9)
plt.tight_layout()
plt.savefig("chart8_feature_importance_lr.png", bbox_inches="tight")
plt.show()


# ============================================================
# CELL 15 — 5-Fold Cross Validation
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_train_cv = scaler.transform(X_train)

cv_scores = {
    "Accuracy": cross_val_score(lr, X_train_cv, y_train, cv=cv, scoring="accuracy"),
    "F1":       cross_val_score(lr, X_train_cv, y_train, cv=cv, scoring="f1"),
    "ROC-AUC":  cross_val_score(lr, X_train_cv, y_train, cv=cv, scoring="roc_auc"),
}

print("========================================")
print("  5-FOLD CROSS-VALIDATION RESULTS")
print("  (Logistic Regression, no SMOTE)")
print("========================================")
for metric, scores in cv_scores.items():
    print(f"  {metric:<12}: {scores.mean():.4f} ± {scores.std():.4f}  {np.round(scores, 3)}")
print("========================================")


# ============================================================
# CELL 16 — Save Summary for Partner (ML2) and LLM Group
# ============================================================
summary = pd.DataFrame([{
    "Model":               "Logistic Regression (Baseline - Fixed)",
    "Accuracy":            round(accuracy, 4),
    "Precision":           round(precision, 4),
    "Recall":              round(recall, 4),
    "F1":                  round(f1, 4),
    "ROC-AUC":             round(roc_auc, 4),
    "CV ROC-AUC (mean)":   round(cv_scores["ROC-AUC"].mean(), 4),
    "CV ROC-AUC (std)":    round(cv_scores["ROC-AUC"].std(), 4),
    "Note":                "Fixed version — includes country, gender, preferred_category, age"
}])

print("=== SUMMARY — send to ML2 partner ===")
print(summary.T.to_string())

# full feature importance (all features, for LLM group Prompt 2)
coef_all = pd.DataFrame({
    "Feature":     X.columns,
    "Coefficient": lr.coef_[0]
}).sort_values("Coefficient", key=abs, ascending=False).reset_index(drop=True)

summary.to_csv("lr_results_summary_v2.csv", index=False)
coef_all.to_csv("lr_feature_importance_v2.csv", index=False)

print("\nFiles saved:")
print("  lr_results_summary_v2.csv        → send to ML2 partner")
print("  lr_feature_importance_v2.csv     → send to LLM group (input for Prompt 2)")
print("  chart8_feature_importance_lr.png → send to LLM group + Viz group")
