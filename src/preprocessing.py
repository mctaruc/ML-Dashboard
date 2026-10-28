"""Data loading, feature definitions, and model pipelines.

Used by BOTH the training script and the Streamlit app, so new user inputs
go through exactly the same preprocessing that was fitted during training.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cleaned_underemployment_dataset.csv"

RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["Education_Tier", "Month"]
NUMERICAL_FEATURES = ["Underemployment_Rate", "Lag1_Rate", "Lag2_Rate", "Lag3_Rate"]
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET = "Target"

MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]

MODEL_INFO = {
    "Linear Regression": {
        "description": "A linear model that fits a straight line relationship between the predictors and next month's rate.",
        "settings": "Default settings",
        "interpretable": "High",
    },
    "Ridge Regression": {
        "description": "A linear model with an L2 penalty that keeps the weights small, which can help on small datasets.",
        "settings": "alpha = 5.0",
        "interpretable": "High",
    },
    "Lasso Regression": {
        "description": "A linear model with an L1 penalty that can shrink weak predictors to zero.",
        "settings": "alpha = 0.05",
        "interpretable": "High",
    },
    "Random Forest": {
        "description": "An average of many decision trees. Can capture curved patterns but is harder to explain.",
        "settings": "200 trees, max depth 5, min samples split 10, min samples leaf 5",
        "interpretable": "Lower",
    },
}


def load_data(path=DATA_PATH):
    """Load the prepared dataset and sort it by date and education tier."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Date", "Education_Tier"]).reset_index(drop=True)


def split_by_date(df, train_share=0.80):
    """Chronological split: earlier months for training, latest months for testing."""
    dates = sorted(df["Date"].unique())
    cutoff_date = dates[int(len(dates) * train_share)]
    train_df = df[df["Date"] < cutoff_date].copy()
    test_df = df[df["Date"] >= cutoff_date].copy()
    return train_df, test_df


def build_preprocessor():
    """One hot encode the two category columns and pass the rates through."""
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ]
    )


def build_models():
    """Return fresh, unfitted pipelines (preprocessing + model) for every candidate."""
    def pipe(model):
        return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])

    return {
        "Linear Regression": pipe(LinearRegression()),
        "Ridge Regression": pipe(Ridge(alpha=5.0)),
        "Lasso Regression": pipe(Lasso(alpha=0.05, max_iter=10000, random_state=RANDOM_STATE)),
        "Random Forest": pipe(RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_split=10,
            min_samples_leaf=5, random_state=RANDOM_STATE)),
    }


def build_latest_inputs(df):
    """Rebuild the newest month's inputs from the last rows of the file.

    The prepared file stops one month before the last month of raw data, because every
    row needs a next month as its target. The Target of the last rows is the newest
    month's rate, so that month's inputs can be rebuilt (and used to forecast ahead).
    """
    last = df[df["Date"] == df["Date"].max()].copy()
    latest = last.copy()
    latest["Date"] = last["Date"] + pd.DateOffset(months=1)
    latest["Month"] = latest["Date"].dt.month_name()
    latest["Lag3_Rate"] = last["Lag2_Rate"]
    latest["Lag2_Rate"] = last["Lag1_Rate"]
    latest["Lag1_Rate"] = last["Underemployment_Rate"]
    latest["Underemployment_Rate"] = last["Target"]
    return latest.reset_index(drop=True)
