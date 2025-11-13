import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/Synthetic_Financial_datasets_log.csv"
OUT_DIR = "data/processed"
TEST_SIZE = 0.2
RANDOM_STATE = 42

LABEL_CANDIDATES = ["isFraud", "label", "Class", "fraud", "target", "is_fraud"]

def detect_label(df):
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    # if no label found, raise error
    raise ValueError("Label column not found. Check the dataset.")

def preprocess():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    print("Loaded:", df.shape)

    label_col = detect_label(df)
    print("Detected label:", label_col)

    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))

    # Split
    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train.to_csv(f"{OUT_DIR}/train.csv", index=False)
    test.to_csv(f"{OUT_DIR}/test.csv", index=False)

    print("Saved processed dataset!")

if __name__ == "__main__":
    preprocess()
