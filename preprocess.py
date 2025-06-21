import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_data(df, is_train=True, encoder_path="insomnia_encoder.pkl"):
    df = df.copy()

    # Drop target and irrelevant column
    target_col = "Risk_Level"
    drop_cols = ["If Any Disorder Please mention "]
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # Convert all object/date columns to string to ensure consistency
    for col in df.select_dtypes(include=['object', 'datetime', 'category']).columns:
        df[col] = df[col].astype(str)

    # Special case for time
    if 'What time do you usually sleep? ' in df.columns:
        df['What time do you usually sleep? '] = df['What time do you usually sleep? '].astype(str)

    if is_train:
        encoders = {}
        categorical_cols = df.select_dtypes(include='object').columns.drop(target_col, errors='ignore')

        # Encode input features
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        # Encode target separately
        if target_col in df.columns:
            le_target = LabelEncoder()
            df[target_col] = le_target.fit_transform(df[target_col])
            encoders[target_col] = le_target

        # Save all encoders
        joblib.dump(encoders, encoder_path)

    else:
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder file '{encoder_path}' not found. Ensure training is completed.")

        encoders = joblib.load(encoder_path)
        categorical_cols = df.select_dtypes(include='object').columns.drop(target_col, errors='ignore')

        for col in categorical_cols:
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            else:
                df[col] = -1  # Unknown column

        if target_col in df.columns and target_col in encoders:
            le_target = encoders[target_col]
            df[target_col] = df[target_col].apply(lambda x: le_target.transform([x])[0] if x in le_target.classes_ else -1)

    # Split features and label
    if target_col in df.columns:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
    else:
        X = df
        y = None

    # print(len(X.columns))
    return X, y