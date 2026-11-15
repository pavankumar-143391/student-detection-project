"""Train a dropout-risk model using the UCI student dataset.

Expected file: data/student_dropout.csv
Expected target column: Target (Dropout, Graduate, Enrolled)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app import FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_dropout.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.json"


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(
            "Dataset not found. Download the UCI Student Dropout dataset and save the CSV as "
            "data/student_dropout.csv. See README.md for instructions."
        )

    data = pd.read_csv(DATA_PATH, sep=None, engine="python")
    missing = [column for column in FEATURE_COLUMNS + ["Target"] if column not in data.columns]
    if missing:
        raise SystemExit("Missing expected dataset columns: " + ", ".join(missing))

    # The original dataset has three final outcomes. This baseline groups Graduate/Enrolled
    # together as non-dropout and Dropout as the positive class. Document this choice.
    y = (data["Target"].astype(str).str.strip().str.lower() == "dropout").astype(int)
    X = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if y.nunique() < 2:
        raise SystemExit("The Target column must contain both Dropout and non-Dropout records.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(1)]

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics = {
        "rows": int(len(data)),
        "features": len(FEATURE_COLUMNS),
        "positive_class": "Dropout",
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "note": "Random stratified holdout evaluation; not a causal finding or a guarantee of future outcomes.",
    }
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Model saved to", MODEL_PATH)
    print(json.dumps(metrics, indent=2))
    print("\nReview recall and false negatives carefully before using this model for student support.")


if __name__ == "__main__":
    main()
