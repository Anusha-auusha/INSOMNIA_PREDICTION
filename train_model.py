# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import LabelEncoder
# import joblib
# import json

# # Load dataset
# df = pd.read_excel(r"C:\Users\HP\OneDrive\Desktop\Anu_project\insomnia_rf_streamlit_app_final\insomnia_rf_streamlit_app\Balanced_Risk_Levels.xlsx", engine='openpyxl')

# # Drop irrelevant columns
# df = df.drop(columns=["If Any Disorder Please mention"], errors='ignore')

# # Ensure all object columns are strings
# for col in df.select_dtypes(include=['object', 'datetime', 'category']).columns:
#     df[col] = df[col].astype(str)

# # Label encode categorical columns
# encoders = {}
# categorical_cols = df.select_dtypes(include='object').columns.drop("Risk_Level", errors='ignore')
# for col in categorical_cols:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     encoders[col] = le

# # Encode target
# le_target = LabelEncoder()
# df["Risk_Level"] = le_target.fit_transform(df["Risk_Level"])
# encoders["Risk_Level"] = le_target

# # Save encoders
# joblib.dump(encoders, "insomnia_encoder.pkl")

# # Split data
# X = df.drop("Risk_Level", axis=1)
# y = df["Risk_Level"]
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # Train model
# model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
# model.fit(X_train, y_train)

# # Evaluate model
# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))

# # Save model
# joblib.dump(model, "insomnia_model.pkl")

# # Save feature columns
# with open("feature_columns.json", "w") as f:
#     json.dump(list(X.columns), f)

#----------------------------------------------------------------------------

# import pandas as pd
# import json
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.utils.class_weight import compute_sample_weight
# from imblearn.over_sampling import SMOTE
# from preprocess import preprocess_data

# # Load the dataset
# df = pd.read_excel("Balanced_Risk_Levels.xlsx")

# # Preprocess the data
# X, y = preprocess_data(df, is_train=True, encoder_path="insomnia_encoder.pkl")

# # Strip column names (critical for consistency)
# X.columns = X.columns.str.strip()

# # Save the clean feature columns
# with open("feature_columns.json", "w") as f:
#     json.dump(list(X.columns), f)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # Apply SMOTE to training data
# sm = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

# # Train Random Forest with class_weight='balanced'
# rf_model = RandomForestClassifier(
#     class_weight="balanced", random_state=42, n_estimators=100
# )
# rf_model.fit(X_train_resampled, y_train_resampled)

# # Train XGBoost with sample weights
# sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_resampled)
# xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
# xgb_model.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights)

# # Save models
# joblib.dump(rf_model, "insomnia_rf_model.pkl")
# joblib.dump(xgb_model, "insomnia_xgb_model.pkl")

# # Evaluate models
# print("📊 Random Forest Classification Report:")
# print(classification_report(y_test, rf_model.predict(X_test)))
# print("Confusion Matrix:\n", confusion_matrix(y_test, rf_model.predict(X_test)))

# print("\n📊 XGBoost Classification Report:")
# print(classification_report(y_test, xgb_model.predict(X_test)))
# print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_model.predict(X_test)))



#----------------------------------------------------------------------------

import pandas as pd
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from preprocess import preprocess_data

# Load dataset
df = pd.read_excel("Balanced_Risk_Levels.xlsx")

# --------------------------
# FEATURE ENGINEERING
# --------------------------
# Ensure text columns are string
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip()

df.columns = df.columns.str.strip()  # ✅ Clean up column names


# Add custom feature
df["Bad_Habits_Count"] = (
    (df["Do you drink caffeine (coffee, tea, energy drinks) in the evening?"] == "Yes").astype(int) +
    (df["How often do you use your phone/laptop before sleep?"] == "Always").astype(int) +
    (df["Do you exercise regularly?"] == "No").astype(int) +
    (df["Do you wake up frequently at night?"] == "Yes").astype(int) +
    (df["Do you have trouble falling asleep within 30 minutes?"] == "Yes").astype(int) +
    (df["Do you wake up feeling refreshed?"] == "No").astype(int)
)

# Preprocess
X, y = preprocess_data(df, is_train=True, encoder_path="insomnia_encoder.pkl")

# Clean column names
X.columns = X.columns.str.strip()

# Save feature column names
with open("feature_columns.json", "w") as f:
    json.dump(list(X.columns), f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# --------------------------
# Train Random Forest
# --------------------------
rf_model = RandomForestClassifier(
    class_weight="balanced", n_estimators=100, random_state=42
)
rf_model.fit(X_train_res, y_train_res)

# --------------------------
# Train XGBoost
# --------------------------
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_res)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train_res, y_train_res, sample_weight=sample_weights)

# Save models
joblib.dump(rf_model, "insomnia_rf_model.pkl")
joblib.dump(xgb_model, "insomnia_xgb_model.pkl")

# --------------------------
# Evaluate
# --------------------------
print("📊 Random Forest Classification Report:")
print(classification_report(y_test, rf_model.predict(X_test)))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_model.predict(X_test)))
print("\nClass Distribution in Test Set:")
print(pd.Series(y_test).value_counts())

print("\n📊 XGBoost Classification Report:")
print(classification_report(y_test, xgb_model.predict(X_test)))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_model.predict(X_test)))
print("\nClass Distribution in Test Set:")
print(pd.Series(y_test).value_counts())

# Print feature importance
print("\n📊 Feature Importance (Random Forest):")
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(rf_importance)

print("\n📊 Feature Importance (XGBoost):")
xgb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(xgb_importance)