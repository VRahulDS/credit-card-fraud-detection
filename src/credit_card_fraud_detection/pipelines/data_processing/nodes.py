import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop(columns=["first", "last", "street", "trans_num"], errors="ignore")

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["year"] = df["trans_date_trans_time"].dt.year

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = 2025 - df["dob"].dt.year

    df["amt_log"] = df["amt"].apply(lambda x: 0 if x <= 0 else np.log(x))

    return df


def split_data(df: pd.DataFrame):

    df = df.sort_values("trans_date_trans_time")

    train_size = int(len(df) * 0.8)

    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    X_train = train.drop(columns=["is_fraud"])
    y_train = train["is_fraud"]

    X_test = test.drop(columns=["is_fraud"])
    y_test = test["is_fraud"]

    return X_train, X_test, y_train, y_test