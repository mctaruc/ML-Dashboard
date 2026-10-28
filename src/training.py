"""Training, validation, and evaluation. Run with:  python train_models.py

Leakage safe design
1. The data is split by time FIRST: earlier months = training, latest months = test.
2. Models are compared with rolling validation on the TRAINING months only.
3. The recommended model is chosen from the rolling validation, NOT from the test set.
4. The test set is scored once, at the end, to report the final result.
5. Preprocessing lives inside each Pipeline, so it is fitted only on training rows.
6. Deployed models are refitted on all rows so the forecast uses the newest data.
"""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.preprocessing import (
    DATA_PATH, FEATURES, MODEL_INFO, ROOT, TARGET,
    build_latest_inputs, build_models, load_data, split_by_date,
)
from src.utilities import (
    persistence_baseline, regression_metrics, slugify, tier_average_baseline,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS_DIR = ROOT / "models"
EVAL_DIR = ROOT / "evaluation"
N_SPLITS = 3
PERSISTENCE = "Persistence Baseline"
TIER_AVERAGE = "Tier Average Baseline"
BASELINES = [PERSISTENCE, TIER_AVERAGE]


def month_label(ts):
    return pd.Timestamp(ts).strftime("%b %Y")


def rolling_validation(train_df):
    """Rolling validation inside the training months. All tiers of a month stay together."""
    dates = np.sort(train_df["Date"].unique())
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS).split(dates), start=1):
        fold_train = train_df[train_df["Date"].isin(dates[tr_idx])]
        fold_val = train_df[train_df["Date"].isin(dates[va_idx])]
        info = {
            "Fold": fold,
            "Training Months": f"{month_label(dates[tr_idx][0])} to {month_label(dates[tr_idx][-1])}",
            "Validation Months": f"{month_label(dates[va_idx][0])} to {month_label(dates[va_idx][-1])}",
        }
        for name, model in build_models().items():
            model.fit(fold_train[FEATURES], fold_train[TARGET])
            rows.append({**info, "Model": name,
                         **regression_metrics(fold_val[TARGET], model.predict(fold_val[FEATURES]))})
        rows.append({**info, "Model": PERSISTENCE,
                     **regression_metrics(fold_val[TARGET], persistence_baseline(fold_val))})
        rows.append({**info, "Model": TIER_AVERAGE,
                     **regression_metrics(fold_val[TARGET], tier_average_baseline(fold_train, fold_val))})
    return pd.DataFrame(rows)


def final_test(train_df, test_df):
    """Fit on all training months, then score the held out test months ONCE."""
    preds = {}
    for name, model in build_models().items():
        model.fit(train_df[FEATURES], train_df[TARGET])
        preds[name] = model.predict(test_df[FEATURES])
    preds[PERSISTENCE] = persistence_baseline(test_df).values
    preds[TIER_AVERAGE] = tier_average_baseline(train_df, test_df).values

    results = pd.DataFrame(
        [{"Model": n, **regression_metrics(test_df[TARGET], p)} for n, p in preds.items()]
    )
    predictions = test_df[["Date", "Education_Tier"]].reset_index(drop=True)
    predictions["Actual"] = test_df[TARGET].values
    for name, p in preds.items():
        predictions[name] = np.asarray(p)
    return results, predictions


def tier_statistics(df):
    """Per tier facts used by the app to explain a forecast."""
    stats = {}
    for tier, g in df.groupby("Education_Tier"):
        rates = pd.concat([g["Underemployment_Rate"], g[TARGET]])
        stats[tier] = {
            "mean": float(rates.mean()), "std": float(rates.std()),
            "min": float(rates.min()), "max": float(rates.max()),
            "avg_monthly_change": float((g["Underemployment_Rate"] - g["Lag1_Rate"]).abs().mean()),
        }
    median_change = float(np.median([s["avg_monthly_change"] for s in stats.values()]))
    for s in stats.values():
        s["noisy"] = bool(s["avg_monthly_change"] > 1.25 * median_change)
    return stats


def bar_figure(table, title, path, highlight):
    data = table.sort_values("MAE", ascending=False)
    colors = ["#9e9e9e" if m in BASELINES else "#1f77b4" for m in data["Model"]]
    labels = [f"{m} (selected)" if m == highlight else m for m in data["Model"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, data["MAE"], color=colors)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in data["MAE"]], padding=3)
    ax.set_xlabel("MAE (percentage points). Lower means a smaller average error.")
    ax.set_title(title)
    ax.set_xlim(0, data["MAE"].max() * 1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scatter_figure(predictions, model_name, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(predictions["Actual"], predictions[model_name], alpha=0.8)
    low = min(predictions["Actual"].min(), predictions[model_name].min()) - 1
    high = max(predictions["Actual"].max(), predictions[model_name].max()) + 1
    ax.plot([low, high], [low, high], linestyle="--", color="gray", label="Perfect forecast")
    ax.set_xlabel("Actual next month rate (%)")
    ax.set_ylabel("Predicted next month rate (%)")
    ax.set_title(f"Held out test months: {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def to_markdown(df):
    """Tiny markdown table writer (no extra packages needed)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def train_and_evaluate(data_path=DATA_PATH):
    """Run the full workflow and save every artifact the app and the report need."""
    MODELS_DIR.mkdir(exist_ok=True)
    EVAL_DIR.mkdir(exist_ok=True)

    df = load_data(data_path)
    train_df, test_df = split_by_date(df)

    # 1. Rolling validation on training months only, then choose the recommended model.
    by_fold = rolling_validation(train_df)
    summary = (by_fold.groupby("Model")[["MAE", "RMSE", "R²"]].mean()
               .sort_values("MAE").reset_index())
    ml_summary = summary[~summary["Model"].isin(BASELINES)]
    recommended = ml_summary.iloc[0]["Model"]

    # 2. Final held out test, scored once.
    test_results, test_predictions = final_test(train_df, test_df)
    test_results = test_results.sort_values("MAE").reset_index(drop=True)

    # 3. Error by tier for the recommended model on the test months.
    tier_rows = []
    for tier, g in test_predictions.groupby("Education_Tier"):
        tier_rows.append({
            "Education Tier": tier,
            f"MAE ({recommended})": float(np.abs(g["Actual"] - g[recommended]).mean()),
            f"MAE ({PERSISTENCE})": float(np.abs(g["Actual"] - g[PERSISTENCE]).mean()),
            f"MAE ({TIER_AVERAGE})": float(np.abs(g["Actual"] - g[TIER_AVERAGE]).mean()),
        })
    tier_errors = pd.DataFrame(tier_rows)

    # 4. Save evaluation tables and figures.
    by_fold.round(4).to_csv(EVAL_DIR / "rolling_validation_by_fold.csv", index=False)
    summary.round(4).to_csv(EVAL_DIR / "rolling_validation_summary.csv", index=False)
    test_results.round(4).to_csv(EVAL_DIR / "test_results.csv", index=False)
    test_predictions.round({c: 4 for c in test_predictions.columns if c not in ("Date", "Education_Tier")}).to_csv(
        EVAL_DIR / "test_predictions.csv", index=False)
    tier_errors.round(4).to_csv(EVAL_DIR / "test_error_by_tier.csv", index=False)
    bar_figure(summary, "Rolling validation (training months only): average MAE",
               EVAL_DIR / "fig_rolling_validation_mae.png", recommended)
    bar_figure(test_results, "Final held out test months: MAE",
               EVAL_DIR / "fig_test_mae.png", recommended)
    scatter_figure(test_predictions, recommended, EVAL_DIR / "fig_test_actual_vs_predicted.png")

    md = [
        "# Evaluation results (generated by train_models.py)", "",
        f"Training months: {month_label(train_df['Date'].min())} to {month_label(train_df['Date'].max())}",
        f"Test months: {month_label(test_df['Date'].min())} to {month_label(test_df['Date'].max())}",
        f"Rolling validation folds: {N_SPLITS} (training months only)", "",
        f"**Selected model (rule: lowest mean MAE across the rolling validation folds, training months only):** {recommended}", "",
        "## Rolling validation, average across folds", to_markdown(summary), "",
        "## Final held out test result", to_markdown(test_results), "",
        "## Test error by education tier", to_markdown(tier_errors), "",
        "## Rolling validation by fold", to_markdown(by_fold), "",
    ]
    (EVAL_DIR / "results_summary.md").write_text("\n".join(md), encoding="utf-8")

    # 5. Refit every model on ALL rows and save the pipelines (preprocessing included).
    for name, model in build_models().items():
        model.fit(df[FEATURES], df[TARGET])
        joblib.dump(model, MODELS_DIR / f"{slugify(name)}.joblib")

    latest = build_latest_inputs(df)
    latest_inputs = {
        r["Education_Tier"]: {
            "Month": r["Month"],
            "Underemployment_Rate": float(r["Underemployment_Rate"]),
            "Lag1_Rate": float(r["Lag1_Rate"]),
            "Lag2_Rate": float(r["Lag2_Rate"]),
            "Lag3_Rate": float(r["Lag3_Rate"]),
        } for _, r in latest.iterrows()
    }
    latest_date = latest["Date"].iloc[0]
    forecast_date = latest_date + pd.DateOffset(months=1)

    train_dates = np.sort(train_df["Date"].unique())
    validation_dates = set()
    for _, va_idx in TimeSeriesSplit(n_splits=N_SPLITS).split(train_dates):
        validation_dates.update(train_dates[va_idx])
    n_validation_months = len(validation_dates)

    summary_idx = summary.set_index("Model")
    test_idx = test_results.set_index("Model")
    metadata = {
        "recommended": recommended,
        "model_names": list(MODEL_INFO.keys()),
        "features": FEATURES,
        "n_rows": int(len(df)),
        "n_tiers": int(df["Education_Tier"].nunique()),
        "data_start": month_label(df["Date"].min()),
        "data_end": month_label(df["Date"].max()),
        "train_period": f"{month_label(train_df['Date'].min())} to {month_label(train_df['Date'].max())}",
        "test_period": f"{month_label(test_df['Date'].min())} to {month_label(test_df['Date'].max())}",
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "n_train_months": int(train_df["Date"].nunique()),
        "n_test_months": int(test_df["Date"].nunique()),
        "n_validation_months": n_validation_months,
        "n_splits": N_SPLITS,
        "latest_label": latest_date.strftime("%B %Y"),
        "forecast_label": forecast_date.strftime("%B %Y"),
        "latest_inputs": latest_inputs,
        "tier_stats": tier_statistics(df),
        "model_errors": {
            name: {"rolling_mae": float(summary_idx.loc[name, "MAE"]),
                   "test_mae": float(test_idx.loc[name, "MAE"])}
            for name in list(MODEL_INFO.keys()) + BASELINES
        },
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
