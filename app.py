"""Streamlit app: Forecasting Underemployment Across Education Levels in the Philippines.

Run locally:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src import ui
from src.prediction import MAX_RATE, check_inputs, interpret, load_artifacts, predict
from src.preprocessing import DATA_PATH, MODEL_INFO, MONTHS, ROOT
from src.utilities import slugify

# ---- Information the project files do not contain. Please fill these in before submitting. ----
TEAM_MEMBERS = "[Add your name]"
DATA_SOURCE_DETAILS = "[Add the exact PSA table names or links used to build the dataset]"
# --------------------------------------------------------------------------------------------

EVAL = ROOT / "evaluation"
BASELINES = ["Persistence Baseline", "Tier Average Baseline"]
NAV = ["🏠 Home", "📊 Models & Evaluation", "🔮 Prediction", "ℹ️ About & Limitations"]

st.set_page_config(page_title="Underemployment Forecaster", page_icon="📊", layout="wide")
ui.inject_css(st)


# ---------------------------------------------------------------- helpers
def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True)


def show_table(df):
    try:
        st.dataframe(df, hide_index=True, width="stretch")
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True)


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def gap():
    html(ui.gap())


def go_to(page):
    st.session_state["nav"] = page


@st.cache_resource(show_spinner="Loading models...")
def get_artifacts():
    """Load saved models. If they are missing or unreadable, retrain from the data file."""
    try:
        return load_artifacts()
    except Exception:
        from src.training import train_and_evaluate
        train_and_evaluate()
        return load_artifacts()


@st.cache_data
def load_eval():
    return {
        "summary": pd.read_csv(EVAL / "rolling_validation_summary.csv"),
        "by_fold": pd.read_csv(EVAL / "rolling_validation_by_fold.csv"),
        "test": pd.read_csv(EVAL / "test_results.csv"),
        "tier_errors": pd.read_csv(EVAL / "test_error_by_tier.csv"),
        "preds": pd.read_csv(EVAL / "test_predictions.csv", parse_dates=["Date"]),
    }


@st.cache_data
def load_history():
    """Actual monthly rates per education group (the newest month is rebuilt from the Target column)."""
    if not DATA_PATH.exists():
        return {}
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"]).sort_values(["Education_Tier", "Date"])
    history = {}
    for tier, g in df.groupby("Education_Tier"):
        s = g.set_index("Date")["Underemployment_Rate"]
        s.loc[g["Date"].max() + pd.DateOffset(months=1)] = g["Target"].iloc[-1]
        history[tier] = s
    return history


def ranked(me, key):
    """'Name value' pairs for every model and baseline, lowest error first."""
    items = sorted(me.items(), key=lambda kv: kv[1][key])
    return ", ".join(f"{name} {vals[key]:.2f}" for name, vals in items)


def selection_points(meta, by_fold):
    """Technical selection rationale built only from the saved evaluation numbers."""
    rec, me = meta["recommended"], meta["model_errors"]
    ml = list(MODEL_INFO)
    by_val = sorted(ml, key=lambda m: me[m]["rolling_mae"])
    gap_top = me[by_val[1]]["rolling_mae"] - me[by_val[0]]["rolling_mae"]
    folds = by_fold[by_fold["Model"] == rec]["MAE"]
    spread = folds.max() - folds.min()
    best_test = min(ml, key=lambda m: me[m]["test_mae"])

    points = [
        ("Selection rule",
         f"**{rec}** has the lowest mean MAE across {meta['n_splits']} rolling validation folds. The folds use the "
         f"training months only ({meta['n_train_months']} months, with {meta['n_validation_months']} validation "
         f"months in total). The {meta['n_test_months']} test months were not used to choose. "
         f"Model: {MODEL_INFO[rec]['description']} Settings: {MODEL_INFO[rec]['settings']}."),
        ("Validation MAE (mean of folds)", ranked(me, "rolling_mae") + "."),
        ("Size of the gap",
         f"The gap between the two lowest of the four models is {gap_top:.2f} points. For {rec}, MAE ranges from "
         f"{folds.min():.2f} to {folds.max():.2f} across folds. "
         + ("The gap is smaller than this fold to fold range, so the order of the closest models can change with the period."
            if gap_top < spread else "The gap is larger than this fold to fold range.")),
        ("Final test MAE (scored once)", ranked(me, "test_mae") + "."),
    ]
    if best_test != rec:
        worst_val = by_val[-1]
        if best_test == worst_val:
            first = (f"{best_test} has the lowest test MAE ({me[best_test]['test_mae']:.2f}) and the highest "
                     f"validation MAE ({me[best_test]['rolling_mae']:.2f}) of the four models. ")
        else:
            first = (f"{best_test} has the lowest test MAE ({me[best_test]['test_mae']:.2f}) and {worst_val} has "
                     f"the highest validation MAE ({me[worst_val]['rolling_mae']:.2f}) of the four models. ")
        second = (f"The selection follows validation because it covers {meta['n_validation_months']} months in "
                  f"{meta['n_splits']} folds, while the test covers {meta['n_test_months']} months.")
        points.append(("Validation and test rankings differ", first + second))
    return points


# ---------------------------------------------------------------- Home
def page_home(art):
    meta, rec = art["meta"], art["meta"]["recommended"]
    html(ui.hero(
        "Forecasting Underemployment Across Education Levels in the Philippines",
        "Machine Learning Final Project &nbsp;·&nbsp; UN SDG 8: Decent Work and Economic Growth",
        ["Next month forecast", f"{len(MODEL_INFO)} models compared", "Philippine Statistics Authority data"]))

    c = st.columns(4)
    cards = [
        ui.kpi("Education levels", meta["n_tiers"], "College, Elementary, Junior High and more"),
        ui.kpi("Months of data per level", meta["n_rows"] // meta["n_tiers"],
               f"{meta['data_start']} to {meta['data_end']} (prepared rows)"),
        ui.kpi("Models compared", len(MODEL_INFO), "plus 2 reference forecasts"),
        ui.kpi("Selected model", rec, "by the validation rule on the Models page"),
    ]
    for col, card in zip(c, cards):
        col.markdown(card, unsafe_allow_html=True)

    gap()
    a, b, d = st.columns(3)
    a.markdown(ui.info_card(
        "The problem and who it helps",
        "Having a job does not always mean having enough work. **Underemployed** workers have jobs but want "
        "more hours or better fitting work. **Labor planners, schools, and career guidance offices** need an "
        "early signal of which education groups may be under pressure.", "lg fill"), unsafe_allow_html=True)
    b.markdown(ui.info_card(
        "What the app predicts",
        "The **underemployment rate (%) of one education level in the next month**: underemployed workers divided "
        "by employed workers in that group. Inputs: education level, month, this month's rate, and the rates "
        "of the previous three months.", "lg fill"), unsafe_allow_html=True)
    d.markdown(ui.info_card(
        "The data",
        f"Monthly **Philippine Statistics Authority** data for {meta['n_tiers']} education levels. These are "
        f"group totals, not records of individual workers. {meta['n_rows']} prepared rows: the first "
        f"{meta['n_train_months']} months train the models and the last {meta['n_test_months']} months are kept "
        f"for the final test.", "lg fill"), unsafe_allow_html=True)

    gap()
    st.subheader("Where to go next")
    s1, s2, s3 = st.columns(3)
    next_steps = [
        (s1, "1. See the evidence", "How the four models were compared and how one was selected.",
         "Open Models & Evaluation", NAV[1], "go_models", "secondary"),
        (s2, "2. Make a forecast", "Pick an education level and a model, enter recent rates, and read what the forecast means.",
         "Open Prediction", NAV[2], "go_predict", "primary"),
        (s3, "3. Know the limits", "When the forecast helps, and when it should not be relied on.",
         "Open About & Limitations", NAV[3], "go_about", "secondary"),
    ]
    for col, title, text, label, target, key, kind in next_steps:
        with col:
            with st.container(border=True, height=190):
                st.markdown(f"**{title}**")
                st.write(text)
                try:
                    st.button(label, on_click=go_to, args=(target,), key=key, type=kind, width="stretch")
                except TypeError:  # older Streamlit versions
                    st.button(label, on_click=go_to, args=(target,), key=key, type=kind, use_container_width=True)

    gap()
    st.warning("**Responsible use.** This is a forecasting aid, not a decision maker. Forecasts are typically off by "
               "about two percentage points and should be reviewed by a person together with official PSA releases.")


# ---------------------------------------------------------------- Models & Evaluation
def page_models(art):
    meta, rec = art["meta"], art["meta"]["recommended"]
    me = meta["model_errors"]
    ev = load_eval()

    st.title("Models & Evaluation")
    html(ui.banner(f"Selected model: {rec}",
                   f"Selection rule: lowest mean MAE across {meta['n_splits']} rolling validation folds, using the "
                   f"training months only. The test months were not used to choose."))

    k = st.columns(4)
    cards = [
        ui.kpi("Validation MAE", f"{me[rec]['rolling_mae']:.2f}", f"{rec}, mean of folds"),
        ui.kpi("Final test MAE", f"{me[rec]['test_mae']:.2f}", f"{rec}, {meta['test_period']}"),
        ui.kpi("Tier Average Baseline MAE", f"{me['Tier Average Baseline']['rolling_mae']:.2f} | "
                                            f"{me['Tier Average Baseline']['test_mae']:.2f}",
               "validation | final test"),
        ui.kpi("Persistence Baseline MAE", f"{me['Persistence Baseline']['rolling_mae']:.2f} | "
                                           f"{me['Persistence Baseline']['test_mae']:.2f}",
               "validation | final test"),
    ]
    for col, card in zip(k, cards):
        col.markdown(card, unsafe_allow_html=True)
    gap()

    t1, t2, t3, t4 = st.tabs(["Candidate models", "Validation evidence", "Final test result",
                              "Selection & trade offs"])

    with t1:
        st.write("All four models solve the same task with the same inputs and the same preprocessing: education "
                 "level and month are one hot encoded, and the four rates are used as they are.")
        grid = st.columns(2)
        for i, (n, v) in enumerate(MODEL_INFO.items()):
            grid[i % 2].markdown(ui.model_card(n, v["description"], v["settings"], v["interpretable"], n == rec),
                                 unsafe_allow_html=True)
            if i < 2:
                grid[i % 2].markdown(ui.gap(), unsafe_allow_html=True)
        gap()
        c1, c2 = st.columns(2)
        c1.markdown(ui.info_card(
            "Two reference forecasts",
            "**Persistence Baseline:** next month equals this month. **Tier Average Baseline:** next month equals the "
            "group's average rate in the training months. Their errors are shown next to the models for comparison.",
            "tall fill"), unsafe_allow_html=True)
        c2.markdown(ui.info_card(
            "Same design for every model",
            f"**Target:** next month's rate. **Predictors:** education level, month, this month's rate, and three "
            f"earlier monthly rates. **Split by time:** training {meta['train_period']} ({meta['n_train']} rows), "
            f"test {meta['test_period']} ({meta['n_test']} rows). **Random seed:** 42.", "tall fill"),
            unsafe_allow_html=True)

    with t2:
        st.write(f"**Validation evidence** uses rolling validation with {meta['n_splits']} folds inside the training "
                 f"period ({meta['train_period']}). Each fold trains on earlier months and is checked on the next "
                 f"months, so a model never sees the future. The test months are not used here.")
        metric = st.radio("Metric", ["MAE", "RMSE", "R²"], horizontal=True, key="val_metric",
                          help="MAE and RMSE are in percentage points (lower means a smaller error). Higher R² means more of the variation is explained.")
        show_chart(ui.metric_bar(ev["summary"], metric, rec, f"Mean {metric} across validation folds"))
        models_shown = st.multiselect("Models shown in the fold chart", list(ev["by_fold"]["Model"].unique()),
                                      default=[rec] + BASELINES, key="fold_models")
        if models_shown:
            show_chart(ui.fold_lines(ev["by_fold"], metric, models_shown, rec))
        else:
            st.info("Pick at least one model to draw the chart.")
        with st.expander("Tables"):
            st.caption("Mean across folds")
            show_table(ev["summary"].round(3))
            st.caption("Every fold")
            show_table(ev["by_fold"].round(3))

    with t3:
        st.write(f"**Final test result.** The test months ({meta['test_period']}) were kept untouched until the end "
                 f"and scored once. This is separate from the validation above. Models here were trained on the "
                 f"training months only.")
        metric2 = st.radio("Metric", ["MAE", "RMSE", "R²"], horizontal=True, key="test_metric")
        c1, c2 = st.columns(2)
        with c1:
            show_chart(ui.metric_bar(ev["test"], metric2, rec, f"Final test {metric2}"))
        with c2:
            show_chart(ui.tier_error_bars(ev["tier_errors"]))
        st.subheader("Look at the test months")
        f1, f2 = st.columns([1, 2])
        tier = f1.selectbox("Education level", sorted(ev["preds"]["Education_Tier"].unique()), key="test_tier")
        chosen = f2.multiselect("Models to draw", list(MODEL_INFO), default=[rec], key="test_models")
        first = (ev["preds"]["Date"].min() + pd.DateOffset(months=1)).strftime("%b %Y")
        last = (ev["preds"]["Date"].max() + pd.DateOffset(months=1)).strftime("%b %Y")
        st.caption(f"The test rows are {meta['test_period']} (the month each forecast is made from). The chart shows the "
                   f"month being forecast, {first} to {last}.")
        show_chart(ui.test_timeseries(ev["preds"], tier, chosen, rec))
        scatter_model = st.selectbox("Model for the actual vs predicted chart", list(MODEL_INFO),
                                     index=list(MODEL_INFO).index(rec), key="scatter_model")
        show_chart(ui.test_scatter(ev["preds"], scatter_model, sorted(ev["preds"]["Education_Tier"].unique())))
        st.caption("Click a level in the legend to hide or show it. Points closer to the dashed line mean a smaller error.")
        with st.expander("Tables"):
            show_table(ev["test"].round(3))
            show_table(ev["tier_errors"].round(3))

    with t4:
        st.subheader("How the model was selected")
        for title, body in selection_points(meta, ev["by_fold"]):
            html(ui.insight(title, body, "#1F6FEB"))
        st.subheader("Trade offs")
        c1, c2 = st.columns(2)
        c1.markdown(ui.info_card(
            "Cost of forecast errors",
            "A forecast that is far **too low** could hide a group that needs support. A forecast that is far "
            "**too high** could send attention to a group that does not need it. MAE shows the typical miss in "
            "percentage points, and RMSE is also shown because large misses weigh more in it.", "tall fill"),
            unsafe_allow_html=True)
        c2.markdown(ui.info_card(
            "Error versus explainability",
            "Linear, Ridge, and Lasso are linear models whose coefficients can be inspected. Random Forest averages "
            "many trees and is harder to explain. When two models have similar errors, the simpler one is easier to "
            "justify to a stakeholder.", "tall fill"), unsafe_allow_html=True)
        gap()
        st.caption("Evidence on this page comes from models trained on the training months. The models used on the "
                   "Prediction page are the same four models refitted on all rows, so forecasts use the newest data.")


# ---------------------------------------------------------------- Prediction
def page_predict(art):
    meta, models = art["meta"], art["models"]
    rec = meta["recommended"]
    tiers = sorted(meta["tier_stats"])

    st.title("Prediction")
    st.write("Choose an education level and a model, enter the recent rates, then press **Predict**.")

    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1.3])
        tier = c1.selectbox("Education level", tiers, key="tier")
        labels = [f"{n} (selected)" if n == rec else n for n in MODEL_INFO]
        model_label = c2.selectbox("Model used for the prediction", labels,
                                   index=list(MODEL_INFO).index(rec), key="model_choice",
                                   help="'Selected' marks the model chosen by the validation rule.")
        model_name = model_label.replace(" (selected)", "")
        mode = c3.radio("Input values", [f"Latest data ({meta['latest_label']})", "My own values"], key="mode")
        use_latest = mode.startswith("Latest")

        latest = meta["latest_inputs"][tier]
        stats = meta["tier_stats"][tier]
        if use_latest:
            month = latest["Month"]
            defaults = [latest[k] for k in ["Underemployment_Rate", "Lag1_Rate", "Lag2_Rate", "Lag3_Rate"]]
            st.caption(f"Pre filled with the latest rates for {tier} workers ({meta['latest_label']}). You can still edit them.")
        else:
            month = st.selectbox("Month of the latest rate", MONTHS, index=MONTHS.index(latest["Month"]), key="month")
            defaults = [round(stats["mean"], 1)] * 4

        key = f"{tier}_{use_latest}"
        cols = st.columns(4)
        names = ["Rate this month (%)", "Rate 1 month earlier (%)", "Rate 2 months earlier (%)", "Rate 3 months earlier (%)"]
        rates = [
            cols[i].number_input(names[i], min_value=0.0, max_value=MAX_RATE, value=float(round(defaults[i], 2)),
                                 step=0.1, format="%.2f", key=f"{key}_{i}",
                                 help="Underemployed workers as a percentage of employed workers in this group.")
            for i in range(4)
        ]

    forecast_label = meta["forecast_label"] if use_latest else f"the month after {month}"
    pressed = st.button("🔮 Predict", type="primary")

    if pressed:
        errors, warnings = check_inputs(meta["tier_stats"], tier, rates)
        if errors:
            for e in errors:
                st.error(e)
            st.session_state.pop("result", None)
        else:
            all_preds = {n: predict(m, tier, month, *rates) for n, m in models.items()}
            st.session_state["result"] = {
                "tier": tier, "model": model_name, "month": month, "rates": rates, "use_latest": use_latest,
                "value": all_preds[model_name], "current": rates[0],
                "forecast_label": forecast_label, "warnings": warnings, "all": all_preds,
            }

    res = st.session_state.get("result")
    if not res:
        st.info("Press **Predict** to see the forecast and what it means. Nothing is predicted until you do.")
        return

    st.divider()
    st.header("Prediction result")
    for w in res["warnings"]:
        st.warning(w)

    s = meta["tier_stats"][res["tier"]]
    errs = meta["model_errors"][res["model"]]
    change = res["value"] - res["current"]
    if abs(change) < 0.5:
        chip = "About the same as the latest rate"
    else:
        chip = f"{'↑' if change > 0 else '↓'} {abs(change):.1f} points vs latest rate"

    k = st.columns(4)
    cards = [
        ui.kpi(f"Forecast for {res['forecast_label']}", f"{res['value']:.1f}%", res["tier"], chip, "blue"),
        ui.kpi("Model used", res["model"], f"file: models/{slugify(res['model'])}.joblib",
               "Selected by validation rule" if res["model"] == rec else "", "blue"),
        ui.kpi("Usual level for this group", f"{s['mean']:.1f}%", f"range {s['min']:.1f}% to {s['max']:.1f}%"),
        ui.kpi("Typical miss of this model", f"±{errs['test_mae']:.1f} pts", "average error on the final test months"),
    ]
    for col, card in zip(k, cards):
        col.markdown(card, unsafe_allow_html=True)
    gap()

    text = interpret(meta, res["tier"], res["model"], res["value"], res["current"], res["forecast_label"])
    left, right = st.columns([1.15, 1])
    history = load_history().get(res["tier"])
    with left:
        if history is not None and res["use_latest"]:
            show_chart(ui.history_chart(history, res["tier"], pd.to_datetime(meta["forecast_label"]),
                                        res["value"], errs["test_mae"], s["mean"], res["model"]))
            st.caption("The orange diamond is the forecast and its bar is the model's typical miss "
                       "(a rough guide, not a confidence interval).")
        else:
            show_chart(ui.compare_chart(res["all"], res["model"], rec))
    with right:
        st.subheader("What this means for the stakeholder")
        html(ui.insight("Result", text["result"], "#1F6FEB"))
        html(ui.insight("Context", text["context"], "#1F6FEB"))
        html(ui.insight("Possible next step", text["action"], "#1F6FEB"))
        html(ui.insight("Caution", text["caution"], "#1F6FEB"))
        st.caption("Above or below the usual level means more than half a standard deviation from the group's "
                   "average over the data period.")

    with st.expander("Compare all models for these inputs"):
        if history is not None and res["use_latest"]:
            show_chart(ui.compare_chart(res["all"], res["model"], rec))
        comp = pd.DataFrame({"Model": list(res["all"]), "Forecast (%)": [round(v, 2) for v in res["all"].values()]})
        show_table(comp)
        st.caption("Models can give different forecasts for the same inputs. The evidence for each model is on the Models page.")

    export = pd.DataFrame({
        "Education level": res["tier"], "Latest month": res["month"],
        "Rate this month (%)": res["rates"][0], "Rate 1 month earlier (%)": res["rates"][1],
        "Rate 2 months earlier (%)": res["rates"][2], "Rate 3 months earlier (%)": res["rates"][3],
        "Model": list(res["all"]), "Forecast (%)": [round(v, 2) for v in res["all"].values()],
        "Selected for this result": [n == res["model"] for n in res["all"]],
        "Forecast month": res["forecast_label"],
    })
    st.download_button("⬇️ Download this forecast (CSV)", export.to_csv(index=False).encode("utf-8"),
                       file_name="underemployment_forecast.csv", mime="text/csv")


# ---------------------------------------------------------------- About & Limitations
def page_about(art):
    meta = art["meta"]
    me = meta["model_errors"]
    st.title("About & Limitations")
    t1, t2, t3, t4, t5 = st.tabs(["Limitations", "Intended use", "Fairness & data caution",
                                  "Data dictionary", "Project information"])
    with t1:
        items = [
            ("Small dataset", f"Only {meta['n_rows']} prepared rows ({meta['n_tiers']} groups with about "
                              f"{meta['n_rows'] // meta['n_tiers']} months each). Results can change from one period to another."),
            ("Survey based numbers", "PSA numbers come from a survey. Month to month changes in smaller groups, "
                                     "such as No Grade Completed, are large and are not fully captured by the models."),
            ("One month ahead only", "The models forecast next month. They are not built for forecasts several months away."),
            ("Sudden changes", "Sharp drops or jumps are often missed. Forecasts tend to stay closer to a group's usual level."),
            ("Limited inputs", "Only recent rates, education level, and month are used. Events such as disasters or "
                               "policy changes are not known to the model."),
            ("Reference forecasts", f"In validation, forecasting each group's usual level had a mean MAE of "
                                    f"{me['Tier Average Baseline']['rolling_mae']:.2f}, compared with "
                                    f"{me[meta['recommended']]['rolling_mae']:.2f} for the selected model. Compare "
                                    f"a forecast with the group's usual level."),
        ]
        c = st.columns(2)
        for i, (t, b) in enumerate(items):
            c[i % 2].markdown(ui.info_card(t, b, "tall"), unsafe_allow_html=True)
            c[i % 2].markdown(ui.gap(), unsafe_allow_html=True)
    with t2:
        c1, c2 = st.columns(2)
        c1.markdown(ui.info_card("Intended use",
                                 "Early, rough monitoring of which education groups may see higher underemployment next "
                                 "month, as one input for labor planners, schools, and career guidance offices.", "xl fill"),
                    unsafe_allow_html=True)
        c2.markdown(ui.info_card("Do not use it for",
                                 "Decisions about individual workers, employers, or applicants. Claims about what "
                                 "**causes** underemployment. Forecasts when the input rates are far outside the range "
                                 "in the data or after a major shock. Replacing the official PSA release or professional judgment.",
                                 "xl fill"), unsafe_allow_html=True)
    with t3:
        c1, c2 = st.columns(2)
        c1.markdown(ui.info_card("Macro level data",
                                 "The data are monthly totals for education groups, not records of individual workers. "
                                 "A forecast describes a **group**, never a person, and the dataset cannot show who is "
                                 "underemployed or why.", "xl fill"), unsafe_allow_html=True)
        c2.markdown(ui.info_card("Fairness and bias",
                                 "Groups are defined only by education level. The data does not show differences by sex, "
                                 "age, region, or income, and groups with fewer workers are measured less precisely. "
                                 "A forecast should never be used to treat people differently because of their education level.",
                                 "xl fill"), unsafe_allow_html=True)
    with t4:
        dd = ROOT / "data" / "data_dictionary.csv"
        if dd.exists():
            show_table(pd.read_csv(dd))
        else:
            st.info("data/data_dictionary.csv was not found.")
    with t5:
        st.markdown(f"**Project:** Machine Learning Final Project: Underemployment Beyond Unemployment  \n"
                    f"**UN SDG:** SDG 8, Decent Work and Economic Growth  \n"
                    f"**Prepared by:** Mary Claire Raeona T. Taruc  \n"
                    f"**Data source:** Philippine Statistics Authority (PSA).  \n"
                    f"**Training period:** {meta['train_period']}  \n**Final test period:** {meta['test_period']}")


# ---------------------------------------------------------------- Navigation
if not DATA_PATH.exists() and not (ROOT / "models" / "metadata.json").exists():
    st.error("Data file not found. Put cleaned_underemployment_dataset.csv in the data folder and run "
             "`python train_models.py`.")
    st.stop()

artifacts = get_artifacts()
st.session_state.setdefault("nav", NAV[0])

with st.sidebar:
    st.markdown("## 📊 Underemployment Forecaster")
    st.caption("Education levels · Philippines · SDG 8")
    st.radio("Navigate", NAV, key="nav", label_visibility="collapsed")
    st.divider()
    st.markdown("**Project snapshot**")
    st.caption(f"Selected model: **{artifacts['meta']['recommended']}**  \n"
               f"Target: next month's rate  \n"
               f"Data: {artifacts['meta']['data_start']} to {artifacts['meta']['data_end']}")
    summary_file = EVAL / "results_summary.md"
    if summary_file.exists():
        st.download_button("⬇️ Evaluation summary", summary_file.read_bytes(),
                           file_name="results_summary.md", mime="text/markdown")
    st.caption("Forecasts are estimates and need human review.")

{NAV[0]: page_home, NAV[1]: page_models, NAV[2]: page_predict, NAV[3]: page_about}[st.session_state["nav"]](artifacts)
