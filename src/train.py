import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import os

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
ARTIFACT_DIR = "artifacts"


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    X_train = train.drop(columns=["isFraud"])
    y_train = train["isFraud"]

    X_test = test.drop(columns=["isFraud"])
    y_test = test["isFraud"]

    return X_train, y_train, X_test, y_test


def train_model(n_estimators=200, max_depth=10):
    X_train, y_train, X_test, y_test = load_data()

    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name="rf-run"):

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",   # important for imbalanced fraud
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # Save model
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
        joblib.dump(model, model_path)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        print("🎉 Training complete!")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {auc:.4f}")



if __name__ == "__main__":
    train_model()
