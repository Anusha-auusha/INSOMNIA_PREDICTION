import pandas as pd
import json
import joblib
from preprocess import preprocess_data

with open("feature_columns.json", "r") as f:
        feature_cols = json.load(f)


rf_model = joblib.load("insomnia_rf_model.pkl")
xgb_model = joblib.load("insomnia_xgb_model.pkl")

df = pd.DataFrame([[
            23, "Female", "03:00", 3, "Yes", "No",
            "Multiple times daily", 1, "Always", "Yes", "No",
            "Yes", "Yes", "Yes", "Chronic Insomnia", "High Stress"
        ]], columns=[
            "Age", "Gender", "What time do you usually sleep?",
            "How many hours do you sleep per night?",
            "Do you wake up frequently at night?",
            "Do you wake up feeling refreshed?",
            "How often do you take daytime naps?", "Sleep_Quality",
            "How often do you use your phone/laptop before sleep?",
            "Do you drink caffeine (coffee, tea, energy drinks) in the evening?",
            "Do you exercise regularly?",
            "Do you have trouble falling asleep within 30 minutes?",
            "Do you wake up too early and cannot fall asleep again?",
            "Long_Term_Issues", "Stress_Level", "Has_Disorder"
        ])

# Preprocess
# Add custom feature
df["Bad_Habits_Count"] = (
    (df["Do you drink caffeine (coffee, tea, energy drinks) in the evening?"] == "Yes").astype(int) +
    (df["How often do you use your phone/laptop before sleep?"] == "Always").astype(int) +
    (df["Do you exercise regularly?"] == "No").astype(int) +
    (df["Do you wake up frequently at night?"] == "Yes").astype(int) +
    (df["Do you have trouble falling asleep within 30 minutes?"] == "Yes").astype(int) +
    (df["Do you wake up feeling refreshed?"] == "No").astype(int)
)

# Add the missing column
df["If Any Disorder Please mention"] = "Narcolepsy"

# Modify Long_Term_Issues to be more severe
df["Long_Term_Issues"] = "Chronic Insomnia"

X_input, _ = preprocess_data(df.copy(), is_train=False)

# Ensure columns are in the same order as training
X_input = X_input[feature_cols]

# Predict
prediction_rf = rf_model.predict(X_input)[0]
prediction_xgb = xgb_model.predict(X_input)[0]
# Decode prediction

risk_mapping = {
    0: "No Risk",
    1: "At Risk",
    2: "High Risk"
}

label_rf = risk_mapping.get(prediction_rf, "Unknown")
label_xgb = risk_mapping.get(prediction_xgb, "Unknown")

# --- Custom override logic for extreme cases ---
def is_extreme_case(row):
    return (
        (row["How many hours do you sleep per night?"] <= 3) or
        (row["Sleep_Quality"] <= 1) or
        (row["Long_Term_Issues"] in ["Chronic Insomnia", "Narcolepsy"]) or
        (row["Stress_Level"] in ["High Stress", "Extreme Stress"]) or
        (row["If Any Disorder Please mention"] in ["Narcolepsy", "Sleep Apnea"])
    )

if label_rf == "At Risk" and is_extreme_case(df.iloc[0]):
    label_rf = "High Risk"
if label_xgb == "At Risk" and is_extreme_case(df.iloc[0]):
    label_xgb = "High Risk"

# Show result
print("🧠 Prediction Result")
print(f"The predicted insomnia risk level is: **{label_rf}**")
print("Random Forest probabilities:", rf_model.predict_proba(X_input))
print("--------------------------------")
print(f"The predicted insomnia risk level is: **{label_xgb}**")
print("XGBoost probabilities:", xgb_model.predict_proba(X_input))

# Print feature importance
print("\n📊 Feature Importance (Random Forest):")
rf_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(rf_importance)

print("\n📊 Feature Importance (XGBoost):")
xgb_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(xgb_importance)
