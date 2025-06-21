import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from preprocess import preprocess_data

# ---------------------------
# Load and preprocess data
# ---------------------------
df = pd.read_excel("Balanced_Risk_Levels.xlsx", engine="openpyxl")
X, y = preprocess_data(df, is_train=True, encoder_path="insomnia_encoder.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Train Random Forest
# ---------------------------
# rf_model = RandomForestClassifier(
#     n_estimators=100, random_state=42, class_weight='balanced'
# )

rf_model = RandomForestClassifier(
    class_weight='balanced',  # <- key addition
    n_estimators=100,
    max_depth=None,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("=== Random Forest Classification Report ===")
print(classification_report(y_test, rf_pred, zero_division=0))

# ---------------------------
# Train XGBoost
# ---------------------------
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

print("=== XGBoost Classification Report ===")
print(classification_report(y_test, xgb_pred, zero_division=0))

# ---------------------------
# Evaluation Plots (Random Forest)
# ---------------------------
def evaluate_model(y_test, y_pred, model_name="Model"):
    labels = ['No Risk', 'At Risk', 'High Risk']
    label_map = {0: 'No Risk', 1: 'At Risk', 2: 'High Risk'}

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # Pie Chart of Predictions
    unique, counts = np.unique(y_pred, return_counts=True)
    pred_distribution = dict(zip(unique, counts))
    pie_labels = [label_map[i] for i in pred_distribution.keys()]
    sizes = list(pred_distribution.values())

    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=pie_labels, autopct='%1.1f%%',
            startangle=140, colors=['#66b3ff', '#ff9999', '#99ff99'])
    plt.axis('equal')
    plt.title(f'{model_name} - Prediction Distribution')
    plt.tight_layout()
    plt.show()

    # Precision, Recall, F1 by Class
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, zero_division=0)
    x = np.arange(len(label_map))
    bar_width = 0.25

    plt.figure(figsize=(8, 6))
    plt.bar(x - bar_width, precision, bar_width,
            label='Precision', color='skyblue')
    plt.bar(x, recall, bar_width, label='Recall', color='lightgreen')
    plt.bar(x + bar_width, f1, bar_width, label='F1-Score', color='salmon')

    plt.xticks(x, [label_map[i] for i in range(len(label_map))])
    plt.ylim(0, 1.1)
    plt.xlabel("Risk Level")
    plt.ylabel("Score")
    plt.title(f"{model_name} - Performance by Class")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Evaluate both models
evaluate_model(y_test, rf_pred, "Random Forest")
evaluate_model(y_test, xgb_pred, "XGBoost")

# ---------------------------
# Save models and features
# ---------------------------
joblib.dump(rf_model, "insomnia_rf_model.pkl")
joblib.dump(xgb_model, "insomnia_xgb_model.pkl")

with open("feature_columns.json", "w") as f:
    json.dump(list(X.columns), f)

# ---------------------------
# Log Unique Labels
# ---------------------------
print("=== Label Summary ===")
print("Unique y_train:", y_train.unique())
print("Unique RF Predictions :", pd.Series(rf_pred).unique())
print("Unique XGB Predictions:", pd.Series(xgb_pred).unique())