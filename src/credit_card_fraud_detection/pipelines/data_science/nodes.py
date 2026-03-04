import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, roc_auc_score


def prepare_features(X_train: pd.DataFrame, X_test: pd.DataFrame):

    # ----- FEATURE ENGINEERING -----
    for df in [X_train, X_test]:
        df["year"] = df["trans_date_trans_time"].dt.year
        df["month"] = df["trans_date_trans_time"].dt.month
        df["day"] = df["trans_date_trans_time"].dt.day
        df["hour"] = df["trans_date_trans_time"].dt.hour

    # Convert DOB to age if exists
    if "dob" in X_train.columns:
        for df in [X_train, X_test]:
            df["age"] = 2026 - df["dob"].dt.year

    # Drop raw date columns
    drop_cols = ["trans_date_trans_time", "dob",
                 "first", "last", "street", "city"]  # drop high-cardinality text

    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

    # ----- DEFINE COLUMNS PROPERLY -----
    categorical_cols = ["merchant", "category", "gender", "job", "state"]

    numeric_cols = [
        col for col in X_train.columns
        if col not in categorical_cols
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            ),
            (
                "num",
                "passthrough",
                numeric_cols,
            ),
        ]
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    return X_train_encoded, X_test_encoded



def train_model(X_train, y_train):

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    return report