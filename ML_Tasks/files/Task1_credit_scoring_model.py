"""
=============================================================
TASK 1: CREDIT SCORING MODEL — CodeAlpha ML Internship
=============================================================
Objective : Predict an individual's creditworthiness using
            past financial data.
Algorithms : Logistic Regression, Decision Tree, Random Forest,
             XGBoost, SVM
Metrics    : Accuracy, Precision, Recall, F1-Score, ROC-AUC
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import os

# ── Output directory ────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)


# ── 1. Synthetic Credit Dataset ──────────────────────────────
def generate_credit_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic credit dataset."""
    rng = np.random.RandomState(seed)

    age = rng.randint(18, 75, n_samples)
    income = rng.lognormal(10.5, 0.6, n_samples).astype(int)
    employment_years = np.clip(rng.normal(8, 5, n_samples), 0, 45).astype(int)
    num_credit_lines = rng.randint(1, 15, n_samples)
    credit_utilisation = np.clip(rng.beta(2, 5, n_samples), 0, 1)
    num_late_payments = rng.poisson(1.5, n_samples)
    num_defaults = rng.poisson(0.3, n_samples)
    total_debt = (income * rng.uniform(0.1, 4, n_samples)).astype(int)
    monthly_expenses = (income * rng.uniform(0.3, 0.9, n_samples) / 12).astype(int)
    savings = np.clip(rng.normal(15000, 20000, n_samples), 0, None).astype(int)
    loan_amount = rng.randint(1000, 100000, n_samples)
    loan_duration_months = rng.choice([12, 24, 36, 48, 60, 84], n_samples)
    education = rng.choice(
        ["High School", "Bachelor's", "Master's", "PhD", "None"], n_samples,
        p=[0.25, 0.40, 0.20, 0.05, 0.10]
    )
    employment_status = rng.choice(
        ["Employed", "Self-Employed", "Unemployed", "Retired"], n_samples,
        p=[0.60, 0.20, 0.10, 0.10]
    )
    marital_status = rng.choice(
        ["Married", "Single", "Divorced", "Widowed"], n_samples,
        p=[0.50, 0.30, 0.15, 0.05]
    )

    # Credit score heuristic (cleaner, more predictive)
    credit_score_raw = (
        (income / income.max()) * 300
        + (employment_years / 45) * 100
        + (savings / savings.max()) * 150
        - (total_debt / total_debt.max()) * 200
        - num_late_payments * 25
        - num_defaults * 80
        - credit_utilisation * 120
        + (num_credit_lines / 14) * 50
    )
    credit_score = np.clip(
        credit_score_raw * 2 + 500 + rng.normal(0, 15, n_samples), 300, 850
    ).astype(int)

    # Binary target: 1 = creditworthy (use median split for balance)
    prob_credit = 1 / (1 + np.exp(-(credit_score - np.median(credit_score)) / 40))
    target = (rng.uniform(0, 1, n_samples) < prob_credit).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "employment_years": employment_years,
        "num_credit_lines": num_credit_lines,
        "credit_utilisation": credit_utilisation,
        "num_late_payments": num_late_payments,
        "num_defaults": num_defaults,
        "total_debt": total_debt,
        "monthly_expenses": monthly_expenses,
        "savings": savings,
        "loan_amount": loan_amount,
        "loan_duration_months": loan_duration_months,
        "credit_score": credit_score,
        "education": education,
        "employment_status": employment_status,
        "marital_status": marital_status,
        "creditworthy": target,
    })
    return df


# ── 2. Feature Engineering ───────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["debt_to_income"] = df["total_debt"] / (df["income"] + 1)
    df["expense_to_income"] = df["monthly_expenses"] / (df["income"] / 12 + 1)
    df["savings_to_debt"] = df["savings"] / (df["total_debt"] + 1)
    df["loan_to_income"] = df["loan_amount"] / (df["income"] + 1)
    df["monthly_loan_payment"] = df["loan_amount"] / df["loan_duration_months"]

    edu_map = {"None": 0, "High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
    df["education_num"] = df["education"].map(edu_map)

    emp_map = {"Unemployed": 0, "Retired": 1, "Employed": 2, "Self-Employed": 3}
    df["employment_num"] = df["employment_status"].map(emp_map)

    mar_map = {"Single": 0, "Divorced": 1, "Widowed": 2, "Married": 3}
    df["marital_num"] = df["marital_status"].map(mar_map)

    drop_cols = ["education", "employment_status", "marital_status"]
    df.drop(columns=drop_cols, inplace=True)
    return df


# ── 3. EDA Plots ─────────────────────────────────────────────
def plot_eda(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Credit Scoring — Exploratory Data Analysis", fontsize=16, fontweight="bold")

    # Target distribution
    counts = df["creditworthy"].value_counts()
    axes[0, 0].pie(counts, labels=["Not Creditworthy", "Creditworthy"],
                   autopct="%1.1f%%", colors=["#e74c3c", "#2ecc71"], startangle=90)
    axes[0, 0].set_title("Target Distribution")

    # Credit score by target
    for val, colour, lbl in zip([0, 1], ["#e74c3c", "#2ecc71"], ["Not Creditworthy", "Creditworthy"]):
        axes[0, 1].hist(df[df["creditworthy"] == val]["credit_score"],
                        bins=30, alpha=0.7, color=colour, label=lbl)
    axes[0, 1].set_title("Credit Score by Class")
    axes[0, 1].set_xlabel("Credit Score")
    axes[0, 1].legend()

    # Income distribution
    axes[0, 2].hist(df["income"], bins=40, color="#3498db", edgecolor="white")
    axes[0, 2].set_title("Income Distribution")
    axes[0, 2].set_xlabel("Annual Income (USD)")

    # Debt-to-income
    axes[1, 0].boxplot(
        [df[df["creditworthy"] == 0]["debt_to_income"].clip(0, 5),
         df[df["creditworthy"] == 1]["debt_to_income"].clip(0, 5)],
        labels=["Not Creditworthy", "Creditworthy"],
        patch_artist=True,
        boxprops=dict(facecolor="#3498db", color="white"),
    )
    axes[1, 0].set_title("Debt-to-Income Ratio by Class")

    # Late payments
    late_pivot = df.groupby(["num_late_payments", "creditworthy"]).size().unstack(fill_value=0)
    late_pivot = late_pivot[late_pivot.index <= 8]
    late_pivot.plot(kind="bar", ax=axes[1, 1], color=["#e74c3c", "#2ecc71"])
    axes[1, 1].set_title("Late Payments vs Creditworthiness")
    axes[1, 1].set_xlabel("Num Late Payments")
    axes[1, 1].legend(["Not Creditworthy", "Creditworthy"])
    axes[1, 1].tick_params(axis="x", rotation=0)

    # Correlation heatmap (numeric only)
    num_cols = df.select_dtypes(include=[np.number]).columns[:8]
    corr = df[num_cols].corr()
    sns.heatmap(corr, ax=axes[1, 2], annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, annot_kws={"size": 7})
    axes[1, 2].set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.savefig("outputs/eda_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [✔] EDA plot saved → outputs/eda_plots.png")


# ── 4. Model Training & Evaluation ───────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10,
                                                 random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                       use_label_encoder=False, eval_metric="logloss",
                                       random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                                         learning_rate=0.05, random_state=42),
    }

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    results = {}
    roc_data = {}

    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_prob = (model.predict_proba(X_test_sc)[:, 1]
                  if hasattr(model, "predict_proba") else
                  model.decision_function(X_test_sc))

        results[name] = {
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall":    recall_score(y_test, y_pred),
            "F1-Score":  f1_score(y_test, y_pred),
            "ROC-AUC":   roc_auc_score(y_test, y_prob),
        }
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr)

    return results, roc_data, scaler, models


# ── 5. Plotting Results ───────────────────────────────────────
def plot_results(results, roc_data, X_train, y_train, best_model_name, best_model):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Credit Scoring Model — Results", fontsize=15, fontweight="bold")

    # Metrics comparison bar chart
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    df_res = pd.DataFrame(results).T[metrics]
    df_res.plot(kind="bar", ax=axes[0], colormap="tab10", edgecolor="white")
    axes[0].set_title("Model Performance Comparison")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].tick_params(axis="x", rotation=30)

    # ROC Curves
    colours = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for (name, (fpr, tpr)), colour in zip(roc_data.items(), colours):
        auc = results[name]["ROC-AUC"]
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colour, lw=2)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_title("ROC Curves — All Models")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(fontsize=8)

    # Feature importance of best model
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_names = X_train.columns
        fi = pd.Series(importances, index=feature_names).sort_values(ascending=True)
        top15 = fi.tail(15)
        top15.plot(kind="barh", ax=axes[2], color="#3498db")
        axes[2].set_title(f"Feature Importances — {best_model_name}")
        axes[2].set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig("outputs/model_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [✔] Results plot saved → outputs/model_results.png")


def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Creditworthy", "Creditworthy"],
                yticklabels=["Not Creditworthy", "Creditworthy"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [✔] Confusion matrix saved → outputs/confusion_matrix.png")


# ── 6. Main Pipeline ──────────────────────────────────────────
def main():
    print("=" * 60)
    print("  TASK 1: CREDIT SCORING MODEL — CodeAlpha")
    print("=" * 60)

    # Data
    print("\n[1] Generating dataset...")
    df_raw = generate_credit_dataset(n_samples=6000)
    print(f"    Dataset shape: {df_raw.shape}")
    print(f"    Class balance:\n{df_raw['creditworthy'].value_counts()}")
    df_raw.to_csv("outputs/credit_dataset.csv", index=False)

    # EDA
    print("\n[2] Running EDA...")
    df = feature_engineering(df_raw)
    plot_eda(df)

    # Split
    print("\n[3] Splitting data...")
    X = df.drop(columns=["creditworthy"])
    y = df["creditworthy"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # SMOTE for class imbalance
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"    After SMOTE: {X_train_res.shape}")

    # Train
    print("\n[4] Training models...")
    results, roc_data, scaler, models = train_and_evaluate(
        X_train_res, X_test, y_train_res, y_test
    )

    # Print results table
    print("\n[5] Results:")
    df_results = pd.DataFrame(results).T.round(4)
    print(df_results.to_string())
    df_results.to_csv("outputs/model_metrics.csv")

    # Best model
    best_name = df_results["ROC-AUC"].idxmax()
    best_model = models[best_name]
    print(f"\n    🏆 Best Model: {best_name}")
    print(f"    ROC-AUC: {results[best_name]['ROC-AUC']:.4f}")

    # Classification report
    scaler_fit = StandardScaler()
    X_tr_sc = scaler_fit.fit_transform(X_train_res)
    X_te_sc = scaler_fit.transform(X_test)
    best_model.fit(X_tr_sc, y_train_res)
    y_pred_best = best_model.predict(X_te_sc)
    print("\n    Classification Report:")
    print(classification_report(y_test, y_pred_best,
                                target_names=["Not Creditworthy", "Creditworthy"]))

    # Plots
    print("\n[6] Saving plots...")
    plot_results(results, roc_data, X_train_res, y_train_res, best_name, best_model)
    plot_confusion_matrix(y_test, y_pred_best, best_name)

    print("\n✅ Task 1 Complete! All outputs saved in outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
