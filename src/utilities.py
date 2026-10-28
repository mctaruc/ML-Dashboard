"""Small helper functions: metrics and simple baselines."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    """MAE and RMSE are in percentage points of underemployment. R2 is unitless."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R²": r2_score(y_true, y_pred),
    }


def persistence_baseline(frame):
    """Predict that next month's rate equals this month's rate (Lag 1 Persistence)."""
    return frame["Underemployment_Rate"]


def tier_average_baseline(train_frame, target_frame):
    """Predict each education tier's average next month rate from the training data."""
    means = train_frame.groupby("Education_Tier")["Target"].mean()
    return target_frame["Education_Tier"].map(means)


def slugify(name):
    return name.lower().replace(" ", "_")
