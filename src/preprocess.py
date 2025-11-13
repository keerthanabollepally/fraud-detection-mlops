import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/Synthetic_Financial_datasets_log.csv"
OUT_DIR = "data/processed"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def preprocess():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("🔍 Loading dataset...")
    df = pd.read_csv(RAW_PATH)
    print("Shape:", df.shape)

    # Label column
    label_col = "isFraud"

    # ========================
    # Basic Cleaning
    # ========================
    df = df.drop_duplicates()

    # Replace missing values
    df = df.fillna(0)

    # ========================
    # Feature Engineering
    # ========================

    # 1. Log transform amount
    df["amount_log"] = np.log1p(df["amount"])

    # 2. Convert categorical column "type" to one-hot encoding
    df = pd.get_dummies(df, columns=["type"], drop_first=True)

    # 3. Drop string-based IDs (model cannot use them)
    df = df.drop(columns=["nameOrig", "nameDest"])

    # ========================
    # Train-test split
    # ========================
    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # important for imbalance
    )

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train.to_csv(f"{OUT_DIR}/train.csv", index=False)
    test.to_csv(f"{OUT_DIR}/test.csv", index=False)

    print("✅ Preprocessing complete!")
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

if __name__ == "__main__":
    preprocess()
