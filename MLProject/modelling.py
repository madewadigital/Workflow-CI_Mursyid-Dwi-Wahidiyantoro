"""
Machine Learning Training Module for MLflow Project CI Workflow.
Author: Mursyid Dwi Wahidiyantoro (mursyiddwiw)
Course: Membangun Sistem Machine Learning (Dicoding)
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn


def main():
    parser = argparse.ArgumentParser(description="Train Heart Disease Classifier via MLflow Project")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=5, help="Max depth of trees")
    args = parser.parse_args()

    # 1. Konfigurasi MLflow
    # Jika dijalankan via mlflow run, run sudah otomatis dibuat oleh MLflow runner
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment("Heart_Disease_CI_Pipeline")

    # 2. Muat dataset
    data_path = os.path.join(os.path.dirname(__file__), "heart_preprocessing.csv")
    if not os.path.exists(data_path):
        data_path = "heart_preprocessing.csv"

    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Training Model
    with mlflow.start_run():
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("model_type", "RandomForestClassifier")

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="HeartDisease_CI_Model"
        )

        print("=== Retraining Model CI Selesai ===")
        print(f"n_estimators: {args.n_estimators}, max_depth: {args.max_depth}")
        print(f"Accuracy    : {acc:.4f}")
        print(f"Precision   : {prec:.4f}")
        print(f"Recall      : {rec:.4f}")
        print(f"F1-Score    : {f1:.4f}")


if __name__ == "__main__":
    main()
