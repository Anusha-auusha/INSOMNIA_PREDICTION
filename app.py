import streamlit as st
import pandas as pd
import joblib
import json
from preprocess import preprocess_data

# ---------------------
# App Config & Title
# ---------------------
st.set_page_config(page_title="Insomnia Risk Predictor", layout="centered")
st.title("🛌 Insomnia Risk Level Predictor")

# ---------------------
# Load models and encoders
# ---------------------
try:
    rf_model = joblib.load("insomnia_rf_model.pkl")
    xgb_model = joblib.load("insomnia_xgb_model.pkl")
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.stop()

try:
    with open("feature_columns.json", "r") as f:
        feature_cols = json.load(f)
except Exception as e:
    st.error(f"❌ Error loading feature columns: {str(e)}")
    st.stop()

# ---------------------
# User Inputs
# ---------------------
st.markdown("### Enter your sleep and lifestyle details:")

with st.form("insomnia_form"):
    model_choice = st.selectbox("Choose Model", ["Random Forest", "XGBoost"])

    age = st.slider("Age", 19, 25, 21)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    sleep_time = st.selectbox("What time do you usually sleep?", ["10:00", "11:00", "12:00", "01:00", "02:00"])
    sleep_hours = st.slider("How many hours do you sleep per night?", 0, 12, 6)
    wake_freq = st.selectbox("Do you wake up frequently at night?", ["Yes", "No", "Sometimes"])
    refreshed = st.selectbox("Do you wake up feeling refreshed?", ["Yes", "No", "Sometimes"])
    nap_freq = st.selectbox("How often do you take daytime naps?", ["Never", "Rare", "Sometimes", "Daily"])
    sleep_quality = st.slider("Sleep Quality (1-Poor to 5-Great)", 1, 5, 3)
    device_use = st.selectbox("How often do you use your phone/laptop before sleep?", ["Never", "Rarely", "Occasionally", "Always"])
    caffeine = st.selectbox("Do you drink caffeine (coffee, tea, energy drinks) in the evening?", ["Yes", "No", "Sometimes"])
    exercise = st.selectbox("Do you exercise regularly?", ["Yes", "No"])
    trouble_sleep = st.selectbox("Do you have trouble falling asleep within 30 minutes?", ["Yes", "No", "Sometimes"])
    early_wakeup = st.selectbox("Do you wake up too early and cannot fall asleep again?", ["Yes", "No", "Sometimes"])
    long_term = st.selectbox("Long term sleep issues?", ["Yes", "No"])
    stress = st.selectbox("Stress Level", ["No Stress", "Mild Stress", "Moderate Stress", "High Stress"])
    disorder = st.selectbox("Has been diagnosed with a disorder?", ["Yes", "No"])

    submit = st.form_submit_button("Predict Risk")


# --- Custom override logic for extreme cases ---
def is_extreme_case(row):
    # Direct severe conditions
    severe_sleep_duration = row["How many hours do you sleep per night?"] <= 3
    severe_sleep_quality = row["Sleep_Quality"] <= 1
    severe_stress = row["Stress_Level"] in ["High Stress", "Extreme Stress"]
    severe_disorder = row["If Any Disorder Please mention"] in ["Narcolepsy", "Sleep Apnea"]
    severe_long_term = row["Long_Term_Issues"] in ["Chronic Insomnia", "Narcolepsy"]
    
    # Moderate conditions
    moderate_sleep_duration = 3 < row["How many hours do you sleep per night?"] <= 5
    moderate_sleep_quality = 1 < row["Sleep_Quality"] <= 3
    moderate_stress = row["Stress_Level"] == "Moderate Stress"
    moderate_disorder = row["If Any Disorder Please mention"] in ["Restless Leg Syndrome", "Circadian Rhythm Disorder"]
    moderate_long_term = row["Long_Term_Issues"] in ["Occasional Insomnia", "Sleep Deprivation"]
    
    # Additional risk factors
    has_medication = row["If Any Disorder Please mention"] == "Sleep Medication"
    has_anxiety = row["If Any Disorder Please mention"] == "Anxiety"
    has_depression = row["If Any Disorder Please mention"] == "Depression"
    
    # Complex combinations that indicate high risk
    return (
        # Direct severe conditions
        severe_sleep_duration or
        severe_sleep_quality or
        severe_disorder or
        severe_long_term or
        
        # Multiple moderate conditions
        (moderate_sleep_duration and moderate_sleep_quality and moderate_stress) or
        (moderate_sleep_duration and moderate_disorder) or
        (moderate_sleep_quality and moderate_long_term) or
        
        # Severe stress with any other condition
        (severe_stress and (moderate_sleep_duration or moderate_sleep_quality)) or
        
        # Medication with sleep issues
        (has_medication and (moderate_sleep_duration or moderate_sleep_quality)) or
        
        # Mental health conditions with sleep issues
        ((has_anxiety or has_depression) and 
         (moderate_sleep_duration or moderate_sleep_quality or moderate_stress)) or
        
        # Multiple moderate conditions with any additional risk factor
        ((moderate_sleep_duration or moderate_sleep_quality) and 
         (moderate_stress or moderate_disorder or moderate_long_term))
    )

# ---------------------
# Prediction Logic
# ---------------------
if submit:
    try:
        df = pd.DataFrame([[
            age, gender, sleep_time, sleep_hours, wake_freq, refreshed,
            nap_freq, sleep_quality, device_use, caffeine, exercise,
            trouble_sleep, early_wakeup, long_term, stress, disorder
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
        if model_choice == "Random Forest":
            prediction = rf_model.predict(X_input)[0]
        else:
            prediction = xgb_model.predict(X_input)[0]

        # Decode prediction
        risk_mapping = {
            0: "No Risk",
            1: "At Risk",
            2: "High Risk"
        }
        label = risk_mapping.get(prediction, "Unknown")

        if label == "At Risk" and is_extreme_case(X_input.iloc[0]):
            label = "High Risk"

        # Show result
        st.subheader("🧠 Prediction Result")
        st.success(f"The predicted insomnia risk level is: **{label}**")

    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")