"""
Streamlit front end for the TCN forecasters (source-only vs CORAL-adapted, seed 0).

  streamlit run app/app.py

Models are loaded lazily on first use and cached; nothing is loaded at import time.
"""
import io
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

INK, INPUT, FORECAST = "#FFFF00", "#a3a29c", "#2a78d6"
MODELS = {"Source-only": "source_only", "CORAL-adapted": "coral"}
NORMS = {"First 70% of the uploaded series (default)": "upload", "Checkpoint's source stats": "source"}


@st.cache_data
def manifest():
    return core.load_manifest()


@st.cache_resource(show_spinner="Loading model…")
def model_for(path):
    return core.load_model(path)


@st.cache_data
def read_csv(data):
    return pd.read_csv(io.BytesIO(data))


@st.cache_data
def column_defaults(data, domain):
    return core.default_columns(read_csv(data), manifest()["domains"][domain])


def line_chart(df, y_title):
    """df columns: time, value, series."""
    colors = alt.Scale(domain=["input window", "actual", "forecast"], range=[INPUT, INK, FORECAST])
    return (alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("time:T", title=None), y=alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=colors, legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("time:T"), "series:N", alt.Tooltip("value:Q", format=".5g")],
    ).properties(height=320).interactive(bind_y=False))


def metric_row(metrics, label):
    c1, c2 = st.columns(2)
    c1.metric(f"MAE ({label})", f"{metrics['MAE']:.5g}")
    c2.metric(f"RMSE ({label})", f"{metrics['RMSE']:.5g}")


def main():
    st.set_page_config(page_title="Domain-adapted forecasting", layout="wide")
    man = manifest()
    domains = list(man["domains"])

    with st.sidebar:
        st.header("Settings")
        domain = st.selectbox("Domain", domains, format_func=lambda d: man["domains"][d]["label"])
        model_label = st.radio("Model", list(MODELS))
        norm_label = st.radio("Normalisation", list(NORMS))
    meta = man["domains"][domain]
    model, ckpt = model_for(str(core.checkpoint_path(man, domain, MODELS[model_label])))
    L, H = ckpt["input_len"], ckpt["horizon"]

    st.title("Source → target forecasting: source-only vs CORAL")
    if domain == "finance":
        st.warning("Raw log-returns: no learnable signal (negative control). No model beats the train-mean.")
    elif domain == "finance_vol":
        st.info("Negative control: on source validation no backbone beat the train-mean baseline.")

    st.subheader("Data")
    src = st.radio("Input", ["Bundled sample", "Upload CSV"], horizontal=True)
    if src == "Bundled sample":
        df = pd.read_csv(core.SAMPLE_DIR / meta["sample"])
        date_col, value_col = df.columns[0], df.columns[1]
        st.caption(f"Sample `{meta['sample']}` (from `{meta['sample_from']}`), {len(df)} rows.")
    else:
        up = st.file_uploader("CSV with a date column and a value column", type=["csv"])
        if up is None:
            st.info("Upload a CSV, or switch to the bundled sample.")
            st.stop()
        try:
            df = read_csv(up.getvalue())
        except Exception as e:  # unreadable file
            st.error(f"Could not read the CSV: {e}")
            st.stop()
        if df.shape[1] < 2:
            st.error(f"The CSV needs at least two columns (date, value); found {df.shape[1]}.")
            st.stop()
        cols, date_opts = list(df.columns), core.date_options(df)
        date_def, value_def = column_defaults(up.getvalue(), domain)
        c1, c2 = st.columns(2)
        date_col = c1.selectbox("Date column", date_opts, index=date_opts.index(date_def))
        value_col = c2.selectbox("Value column" + (" (closing price)" if domain.startswith("finance") else ""),
                                 cols, index=cols.index(value_def))

    try:
        series, notes = core.parse_series(df, date_col, value_col, meta, L, H)
    except core.InputError as e:
        st.error(str(e))
        st.stop()
    stats = core.pick_stats(series, ckpt, NORMS[norm_label])

    with st.sidebar:
        st.divider()
        st.markdown(f"**Active:** {model_label} · {domain}")
        st.markdown(f"**Normalisation:** {'first 70% of upload' if NORMS[norm_label] == 'upload' else 'checkpoint source stats'}"
                    f" (mean {stats['mean']:.5g}, std {stats['std']:.5g})")
        st.caption(f"Checkpoint `{core.checkpoint_path(man, domain, MODELS[model_label]).name}` · TCN · "
                   f"input_len {L} · horizon {H} · CPU")
    for n in notes:
        st.caption(n)
    st.caption(f"Model-ready series: {len(series)} points, {series.index[0]} → {series.index[-1]} ({meta['units']}).")

    tab_f, tab_b = st.tabs(["Forecast", "Backtest"])
    with tab_f:
        holdout = st.checkbox(f"Hold out the final {H} points as actuals", value=True)
        f = core.forecast_last(model, series, stats, L, H, meta["frequency"], holdout)
        parts = [pd.DataFrame({"time": f["input"].index, "value": f["input"].values, "series": "input window"}),
                 pd.DataFrame({"time": f["table"]["time"], "value": f["table"]["forecast"], "series": "forecast"})]
        if f["metrics"]:
            parts.append(pd.DataFrame({"time": f["table"]["time"], "value": f["table"]["actual"], "series": "actual"}))
        st.altair_chart(line_chart(pd.concat(parts), meta["units"]), width="stretch")
        if f["metrics"]:
            metric_row(f["metrics"], f"{H}-step forecast")
        else:
            st.caption("Forecast beyond the end of the series; no actuals to score against.")
        st.dataframe(f["table"], hide_index=True, width="stretch")

    with tab_b:
        b = core.backtest(model, series, stats, L, H)
        st.caption(f"{b['n_windows']} sliding windows (stride 1) over the whole series. Plot: first forecast step.")
        fs = b["first_step"].melt(id_vars="time", value_vars=["actual", "forecast"], var_name="series")
        st.altair_chart(line_chart(fs, meta["units"]), width="stretch")
        metric_row(b["metrics_first_step"], "step 1")
        metric_row(b["metrics_all_steps"], f"all {H} steps")
        st.markdown("**Train-mean baseline** (mean of the first 70% of this series, "
                    f"{b['train_mean']:.5g}, predicted for every step)")
        metric_row(b["baseline_train_mean_all_steps"], f"baseline, all {H} steps")
        if domain.startswith("finance"):
            st.caption("For finance, compare against this baseline: matching it is the expected outcome.")
        st.caption("The first-70% stats and baseline mean overlap the backtest windows, so this is a diagnostic, "
                   "not a held-out test. The bundled samples are the full target series, whose first 70% "
                   "CORAL saw as unlabelled inputs during training.")


try:
    main()
except Exception as e:  # last resort: a plain message instead of a traceback
    if type(e).__name__ in ("StopException", "RerunException"):
        raise
    st.error(f"Something went wrong: {type(e).__name__}: {e}")
