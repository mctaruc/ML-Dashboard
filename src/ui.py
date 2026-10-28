"""Look and feel: styling, small HTML components, and interactive Plotly charts."""
import re

import pandas as pd
import plotly.graph_objects as go

NAVY = "#0F1E36"
BLUE = "#1F6FEB"
GREEN = "#1B7F4B"
RED = "#C23B3B"
MUTED = "#5B6B82"

MODEL_COLORS = {
    "Linear Regression": "#2563EB",
    "Ridge Regression": "#7C3AED",
    "Lasso Regression": "#16A34A",
    "Random Forest": "#F59E0B",
    "Persistence Baseline": "#94A3B8",
    "Tier Average Baseline": "#64748B",
}

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1280px;}
h2, h3 {color: #0F1E36;}
.hero {background: linear-gradient(135deg, #0F1E36 0%, #1D4B8F 100%); color: #fff;
       padding: 26px 32px; border-radius: 16px; margin-bottom: 18px;}
.hero h1 {color: #fff; margin: 0 0 6px 0; font-size: 1.9rem; line-height: 1.25; padding: 0;}
.hero p {color: #D6E2F5; margin: 0 0 12px 0; font-size: 1rem;}
.badge {display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .78rem;
        background: rgba(255,255,255,.16); color: #fff; margin: 0 6px 4px 0;}
.card {background: #fff; border: 1px solid #E3E8F0; border-radius: 14px; padding: 16px 18px 20px;
       box-shadow: 0 1px 3px rgba(15,30,54,.05); min-height: 132px;}
.card.tall {min-height: 206px;}
.card.lg {min-height: 224px;}
.card.xl {min-height: 236px;}
.gap {height: 14px;}
.card.fill {height: 100%; box-sizing: border-box;}
div[data-testid="stColumn"] div:has(.card.fill) {height: 100%;}
.card.model {min-height: 196px;}
.kpi-value.sm {font-size: 1.3rem; padding-top: 4px;}
.card h4 {margin: 0 0 6px 0; color: #0F1E36; font-size: 1.02rem;}
.card p {margin: 0; color: #3B4A60; font-size: .93rem; line-height: 1.5;}
.kpi-label {font-size: .82rem; color: #5B6B82; margin-bottom: 2px;}
.kpi-value {font-size: 1.75rem; font-weight: 700; color: #0F1E36; line-height: 1.25;}
.kpi-sub {font-size: .8rem; color: #7A8AA0; margin-top: 4px;}
.chip {display: inline-block; font-size: .78rem; font-weight: 600; padding: 2px 9px;
       border-radius: 999px; margin-top: 6px;}
.chip-bad {background: #FDECEC; color: #C23B3B;}
.chip-good {background: #E6F6EC; color: #1B7F4B;}
.chip-neutral {background: #EEF2F7; color: #52627A;}
.chip-blue {background: #E7F0FE; color: #1F57B5;}
.insight {border-left: 5px solid #1F6FEB; border-radius: 10px; padding: 12px 16px;
          background: #F7F9FC; margin-bottom: 10px;}
.insight .t {font-weight: 700; color: #0F1E36; margin-bottom: 3px; font-size: .95rem;}
.insight .b {color: #2B3A52; font-size: .92rem; line-height: 1.5;}
.banner {border-radius: 14px; padding: 16px 20px; margin-bottom: 14px;
         background: #EEF4FF; border: 1px solid #C9DAF8;}
.banner .t {font-weight: 700; color: #1F3F7A; font-size: 1.1rem;}
.banner .b {color: #2B4670; font-size: .93rem; margin-top: 3px;}
.note {font-size: .85rem; color: #5B6B82;}
div[data-testid="stSidebar"] .stRadio label p {font-size: .98rem;}
</style>
"""


def inject_css(st):
    st.markdown(CSS, unsafe_allow_html=True)


def bold_to_html(text):
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def hero(title, subtitle, badges):
    chips = "".join(f'<span class="badge">{b}</span>' for b in badges)
    return f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{chips}</div>'


def kpi(label, value, sub="", chip="", chip_kind="neutral"):
    chip_html = f'<div class="chip chip-{chip_kind}">{chip}</div>' if chip else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    size = " sm" if len(str(value)) > 11 else ""
    return (f'<div class="card fill"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value{size}">{value}</div>{sub_html}{chip_html}</div>')


def info_card(title, body, size=""):
    return f'<div class="card {size}"><h4>{title}</h4><p>{bold_to_html(body)}</p></div>'


def model_card(name, description, settings, explain, recommended):
    tag = ' <span class="chip chip-blue" style="margin:0 0 0 6px">selected</span>' if recommended else ""
    chip_kind = "neutral"
    border = "border: 2px solid #1F6FEB;" if recommended else ""
    return (f'<div class="card model" style="{border}"><h4>{name}{tag}</h4>'
            f'<p>{description}</p>'
            f'<p class="note" style="margin-top:8px">Settings: {settings}</p>'
            f'<div class="chip chip-{chip_kind}">Easy to explain: {explain}</div></div>')


def insight(title, body, color=BLUE):
    return (f'<div class="insight" style="border-left-color:{color}"><div class="t">{title}</div>'
            f'<div class="b">{bold_to_html(body)}</div></div>')


def gap():
    return '<div class="gap"></div>'


def banner(title, body):
    return f'<div class="banner"><div class="t">{title}</div><div class="b">{bold_to_html(body)}</div></div>'


# ------------------------------------------------------------------ charts
def _style(fig, height=360, title=None, legend_y=-0.18, bottom=60):
    fig.update_layout(
        template="plotly_white", height=height, title=title,
        margin=dict(l=10, r=10, t=55 if title else 20, b=bottom),
        font=dict(color="#1B2A41"),
        legend=dict(orientation="h", yanchor="top", y=legend_y, xanchor="left", x=0),
    )
    return fig


def _label(name, rec):
    return f"{name} (selected)" if name == rec else name


def metric_bar(table, metric, rec, title):
    """Horizontal bars, best model at the top. Lower is better for MAE and RMSE."""
    d = table.sort_values(metric, ascending=(metric == "R²"))
    fig = go.Figure(go.Bar(
        x=d[metric], y=[_label(m, rec) for m in d["Model"]], orientation="h",
        marker_color=[MODEL_COLORS.get(m, "#999") for m in d["Model"]],
        text=[f"{v:.2f}" for v in d[metric]], textposition="outside",
        hovertemplate="%{y}<br>" + metric + ": %{x:.3f}<extra></extra>",
    ))
    note = "higher is better" if metric == "R²" else "lower is better"
    fig.update_xaxes(title=f"{metric} ({note})", rangemode="tozero")
    return _style(fig, 340, title)


def fold_lines(by_fold, metric, models, rec):
    fig = go.Figure()
    folds = by_fold[["Fold", "Validation Months"]].drop_duplicates().sort_values("Fold")
    labels = [f"Fold {f}<br>{v}" for f, v in zip(folds["Fold"], folds["Validation Months"])]
    for m in models:
        d = by_fold[by_fold["Model"] == m].sort_values("Fold")
        fig.add_trace(go.Scatter(
            x=labels, y=d[metric], mode="lines+markers", name=_label(m, rec),
            line=dict(color=MODEL_COLORS.get(m), width=4 if m == rec else 2,
                      dash="dot" if "Baseline" in m else "solid"),
            hovertemplate="%{x}<br>" + m + ": %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title=metric)
    return _style(fig, 360, f"{metric} in each validation fold")


def test_timeseries(preds, tier, models, rec):
    d = preds[preds["Education_Tier"] == tier].sort_values("Date")
    x = d["Date"] + pd.DateOffset(months=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=d["Actual"], mode="lines+markers", name="Actual",
                             line=dict(color="#0F1E36", width=4)))
    for m in models:
        fig.add_trace(go.Scatter(x=x, y=d[m], mode="lines+markers", name=_label(m, rec),
                                 line=dict(color=MODEL_COLORS.get(m), width=2, dash="dash")))
    fig.update_yaxes(title="Underemployment rate (%)", rangemode="tozero")
    fig.update_xaxes(title="Month being forecast", tickformat="%b %Y")
    return _style(fig, 380, f"Held out test months: {tier}")


def test_scatter(preds, model, tiers):
    d = preds[preds["Education_Tier"].isin(tiers)]
    fig = go.Figure()
    for t, g in d.groupby("Education_Tier"):
        fig.add_trace(go.Scatter(x=g["Actual"], y=g[model], mode="markers", name=t,
                                 marker=dict(size=9, opacity=.85),
                                 hovertemplate=t + "<br>Actual %{x:.1f}%<br>Predicted %{y:.1f}%<extra></extra>"))
    lo = float(min(d["Actual"].min(), d[model].min())) - 1
    hi = float(max(d["Actual"].max(), d[model].max())) + 1
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect forecast",
                             line=dict(color="gray", dash="dash")))
    fig.update_xaxes(title="Actual next month rate (%)")
    fig.update_yaxes(title="Predicted next month rate (%)")
    return _style(fig, 420, f"Actual vs predicted: {model}", legend_y=-0.22, bottom=90)


def tier_error_bars(tier_errors):
    fig = go.Figure()
    colors = ["#16A34A", "#94A3B8", "#64748B"]
    for col, color in zip([c for c in tier_errors.columns if c.startswith("MAE")], colors):
        fig.add_trace(go.Bar(x=tier_errors["Education Tier"], y=tier_errors[col],
                             name=col.replace("MAE (", "").rstrip(")"), marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="MAE (percentage points)")
    fig.update_xaxes(tickangle=-25)
    return _style(fig, 380, "Test error by education level", legend_y=-0.38, bottom=110)


def history_chart(rates_by_date, tier, forecast_date, forecast_value, typical_miss, usual_level, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rates_by_date.index, y=rates_by_date.values, mode="lines+markers",
                             name="Actual rate", line=dict(color=BLUE, width=3)))
    fig.add_hline(y=usual_level, line_dash="dot", line_color="#94A3B8",
                  annotation_text=f"Usual level {usual_level:.1f}%", annotation_position="top left")
    last_date, last_value = rates_by_date.index[-1], rates_by_date.values[-1]
    fig.add_trace(go.Scatter(x=[last_date, forecast_date], y=[last_value, forecast_value],
                             mode="lines", showlegend=False, line=dict(color="#F59E0B", dash="dash")))
    fig.add_trace(go.Scatter(
        x=[forecast_date], y=[forecast_value], mode="markers", name=f"Forecast ({model_name})",
        marker=dict(color="#F59E0B", size=13, symbol="diamond"),
        error_y=dict(type="constant", value=typical_miss, color="#F59E0B", thickness=2),
        hovertemplate=f"Forecast %{{y:.1f}}% (bar = typical miss ±{typical_miss:.1f})<extra></extra>"))
    fig.update_yaxes(title="Underemployment rate (%)", rangemode="tozero")
    fig.update_xaxes(tickformat="%b %Y")
    return _style(fig, 340, f"{tier}: actual rates and the forecast")


def compare_chart(all_preds, selected, rec):
    names = list(all_preds)
    fig = go.Figure(go.Bar(
        x=[all_preds[n] for n in names], y=[_label(n, rec) for n in names], orientation="h",
        marker_color=[MODEL_COLORS[n] if n == selected else "#CBD5E1" for n in names],
        text=[f"{all_preds[n]:.2f}%" for n in names], textposition="outside"))
    fig.update_xaxes(title="Forecast (%)", rangemode="tozero")
    return _style(fig, 260, "Same inputs, every model (selected model in color)")
