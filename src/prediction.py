"""Loading saved models, making predictions, checking inputs, and explaining results."""
import json

import joblib
import pandas as pd

from src.preprocessing import FEATURES, MODEL_INFO, ROOT
from src.utilities import slugify

MODELS_DIR = ROOT / "models"
MAX_RATE = 60.0


def load_artifacts():
    """Load metadata and every fitted pipeline. Raises an error if something is missing."""
    meta = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
    models = {name: joblib.load(MODELS_DIR / f"{slugify(name)}.joblib") for name in MODEL_INFO}
    return {"meta": meta, "models": models}


def predict(model, tier, month, rate, lag1, lag2, lag3):
    """The saved pipeline applies its own fitted preprocessing to the single input row."""
    row = pd.DataFrame([{
        "Education_Tier": tier, "Month": month,
        "Underemployment_Rate": rate, "Lag1_Rate": lag1,
        "Lag2_Rate": lag2, "Lag3_Rate": lag3,
    }])[FEATURES]
    return float(model.predict(row)[0])


def check_inputs(tier_stats, tier, rates):
    """Return (errors, warnings). Errors block the prediction, warnings do not."""
    errors, warnings = [], []
    labels = ["Rate this month", "Rate 1 month earlier", "Rate 2 months earlier", "Rate 3 months earlier"]
    for label, value in zip(labels, rates):
        if value is None or pd.isna(value):
            errors.append(f"{label} is missing.")
        elif value <= 0:
            errors.append(f"{label} must be greater than 0%.")
        elif value > MAX_RATE:
            errors.append(f"{label} must be {MAX_RATE:.0f}% or lower.")
    if not errors:
        s = tier_stats[tier]
        low, high = s["min"] - 3, s["max"] + 3
        outside = [l for l, v in zip(labels, rates) if v < low or v > high]
        if outside:
            warnings.append(
                f"{', '.join(outside)} is outside the range seen for {tier} workers "
                f"({s['min']:.1f}% to {s['max']:.1f}%). The forecast is less reliable here."
            )
    return errors, warnings


def interpret(meta, tier, model_name, predicted, current, forecast_label):
    """Short, data driven explanation of one forecast for the stakeholder."""
    s = meta["tier_stats"][tier]
    me = meta["model_errors"]
    errs = me[model_name]
    change = predicted - current

    # Result
    if abs(change) < 0.5:
        move = f"about the same as the latest rate ({current:.1f}%)"
    else:
        side = "above" if change > 0 else "below"
        move = f"{abs(change):.1f} points {side} the latest rate ({current:.1f}%)"
    result = (f"**{predicted:.1f}%** of employed {tier} workers are forecast to be underemployed "
              f"in {forecast_label}, {move}.")

    # Context: compare with the group's own history (half a standard deviation rule)
    z = (predicted - s["mean"]) / s["std"] if s["std"] > 0 else 0.0
    if z > 0.5:
        level, level_key = "above its usual level", "above"
    elif z < -0.5:
        level, level_key = "below its usual level", "below"
    else:
        level, level_key = "close to its usual level", "close"
    context = (f"The usual rate for {tier} is **{s['mean']:.1f}%** (range {s['min']:.1f}% to {s['max']:.1f}%). "
               f"This forecast is {level}.")

    # Possible next step (a consideration, not a decision)
    if level_key == "above":
        action = "Consider closer monitoring of this group and compare with the next PSA release."
    else:
        action = "Routine monitoring may be enough. Compare with the next PSA release."

    # Caution with the numbers behind it (no verdict words)
    ref = me["Tier Average Baseline"]["rolling_mae"]
    caution = (f"Typical miss: ±{errs['test_mae']:.1f} points on the test months and "
               f"±{errs['rolling_mae']:.1f} in validation. This group's rate changes by about "
               f"{s['avg_monthly_change']:.1f} points per month on average. For reference, forecasting each "
               f"group's usual level had a validation MAE of {ref:.2f}, against {errs['rolling_mae']:.2f} for this "
               f"model. This is an estimate for human review, not a way to judge individual workers.")
    return {"result": result, "context": context, "action": action, "caution": caution, "level": level}
