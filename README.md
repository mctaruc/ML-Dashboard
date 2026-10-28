# Underemployment Beyond Unemployment — Streamlit Forecasting 

This application implements the Machine Learning final project as a Streamlit + Plotly app. It forecasts next month's underemployment rate for each education level in the Philippines (UN SDG 8: Decent Work and Economic Growth), using four regression models compared against a simple baseline.

## Files
- `app.py` — Streamlit application (Home, Data Explorer, Models & Evaluation, Forecast, About & Limitations)
- `data/cleaned_underemployment_dataset.csv` — prepared PSA monthly data (6 education tiers × 32 months)
- `data/data_dictionary.md` — column meanings and which columns are used as predictors
- `models/` — saved model pipelines (`.joblib`) and `registry.json`
- `evaluation/` — saved metric tables and predictions used by the app
- `src/preprocessing.py` — features, encoding pipeline, chronological split
- `src/training.py` — trains, evaluates, selects and saves the models
- `src/prediction.py` — model loading, prediction and stakeholder interpretation
- `src/utilities.py` — paths, constants, helpers
- `notebooks/model_development.ipynb` — original experiments
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in a browser. Use the sidebar to move between pages. On the Forecast page, choose a model and an education group, check the values, and press **Generate forecast**.

To rebuild the models and evaluation tables from the data:

```bash
python -m src.training
```

If the model files are missing or incompatible with your scikit-learn version, the app retrains them automatically on first load.

## Analytical boundaries
- Forecasts are one-month-ahead statistical estimates for education groups, not guarantees and not decisions.
- The model finds patterns in past rates; it does not explain why underemployment changes.
- The data covers only 32 months (192 rows) and only 7 test months, so the top models are close to a tie.
- Lasso Regression is recommended because it had the lowest average RMSE across four rolling validation folds. Linear Regression scored best on the single test window but was the least stable across folds.
- The last validation fold overlaps the test months, so validation and test evidence are not fully independent.
- The ± range shown in the app is the model's typical miss (MAE), not a statistical confidence interval.
- No Grade Completed is the hardest group to forecast, and for Post-Secondary, repeating last month's rate beat the models on the test window.
- Forecasts describe groups, not individual workers, and must not be used to screen or rank people.
- Always confirm results against official PSA releases and keep a human in the decision.