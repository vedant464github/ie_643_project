"""
App logic without Streamlit, so it can be tested directly. Preprocessing, normalisation, windowing, prediction and
metrics are the project's own src functions; this module only validates uploads and wires them together.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.prepare import chronological_split, normalize, apply_norm  # noqa: E402
from src.data.preprocess_climate import fill_daily  # noqa: E402
from src.data.preprocess_energy import fill_hourly  # noqa: E402
from src.data.preprocess_finance import close_to_log_return  # noqa: E402
from src.data.windowing import make_windows, SERIES_TRANSFORMS  # noqa: E402
from src.evaluate import load_model_checkpoint, predict, denorm, naive_mean_metrics  # noqa: E402
from src.utils.metrics import evaluate_all  # noqa: E402

APP = Path(__file__).resolve().parent
CKPT_DIR = APP / "checkpoints"
SAMPLE_DIR = APP / "samples"
PREPROCESS = {"fill_daily": fill_daily, "fill_hourly": fill_hourly}
STEP = {"D": pd.Timedelta(days=1), "B": pd.Timedelta(days=1), "h": pd.Timedelta(hours=1)}
FREQ_NAME = {"D": "daily", "B": "daily (trading days)", "h": "hourly"}
DATE_NAMES = ("date", "day", "time", "timestamp", "datetime")
VALUE_NAMES = {"temp": ("temp",), "load_mw": ("load", "mw", "demand"), "Close": ("close", "price")}  # by value_name
MIN_PARSE = 0.95  # share of rows that must parse for a column to be a default


class InputError(ValueError):
    """A problem with the user's data, shown as a plain message."""


def load_manifest():
    return json.loads((CKPT_DIR / "manifest.json").read_text(encoding="utf-8"))


def checkpoint_path(manifest, domain, kind):
    return CKPT_DIR / manifest["domains"][domain]["checkpoints"][kind]["file"]


def load_model(path):
    """(model, ckpt) on CPU; any checkpoint here carries input_len, horizon and norm stats."""
    return load_model_checkpoint(path)


def source_stats(ckpt):
    """Source-train stats: CORAL checkpoints store {source_train, target_train}; source-only ones store one dict."""
    ns = ckpt["norm_stats"]
    return ns["source_train"] if "source_train" in ns else ns


def _constant_text(col):
    return not pd.api.types.is_numeric_dtype(col) and col.nunique(dropna=True) <= 1


def _date_fraction(col):
    """Share of all rows that parse as dates; numeric columns never count (pandas would read them as epochs)."""
    if len(col) == 0 or pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # format-inference warnings on non-date text
        return float(pd.to_datetime(col, errors="coerce").notna().mean())


def _numeric_fraction(col):
    """Share of non-empty cells that are numeric."""
    filled = col.notna() & (col.astype(str).str.strip() != "")
    return float(pd.to_numeric(col[filled], errors="coerce").notna().mean()) if filled.any() else 0.0


def _name_rank(name, prefs):
    n = str(name).strip().lower()
    return 0 if n in prefs else 1 if any(p in n for p in prefs) else 2


def date_options(df):
    """Columns offered as the date: all except constant text columns (unless nothing else is left)."""
    return [c for c in df.columns if not _constant_text(df[c])] or list(df.columns)


def default_columns(df, meta):
    """
    (date_col, value_col) defaults for an upload. Date: the first offered column whose rows parse as dates
    (>= MIN_PARSE), preferring DATE_NAMES. Value: the first mostly numeric other column, preferring the domain's
    typical name. Falls back to the first candidates so parse_series still reports a clear error.
    """
    opts = date_options(df)
    dates = [c for c in opts if _date_fraction(df[c]) >= MIN_PARSE]
    date_col = min(dates, key=lambda c: _name_rank(c, DATE_NAMES)) if dates else opts[0]
    rest = [c for c in df.columns if c != date_col]
    nums = [c for c in rest if _numeric_fraction(df[c]) >= MIN_PARSE]
    prefs = tuple(p.lower() for p in (meta["value_name"],) + VALUE_NAMES.get(meta["value_name"], ()))
    value_col = min(nums, key=lambda c: _name_rank(c, prefs)) if nums else (rest[0] if rest else date_col)
    return date_col, value_col


def _examples(values, k=3):
    return ", ".join(repr(str(v)) for v in list(values)[:k])


def parse_series(df, date_col, value_col, meta, input_len, horizon):
    """
    Validates an upload-format frame and returns the model-ready series (pd.Series, DatetimeIndex) plus notes.
    Same preprocessing as training: daily calendar fill (climate), hourly interpolation (energy), or
    Close -> log-return (+ abs for finance_vol) on trading days (finance).
    """
    for c, what in ((date_col, "date"), (value_col, "value")):
        if c is None or c not in df.columns:
            raise InputError(f"Missing {what} column {c!r}. Columns in the file: {', '.join(map(str, df.columns))}.")
    if date_col == value_col:
        raise InputError("Pick different columns for the date and the value.")
    if len(df) < 3:
        raise InputError(f"Series too short: {len(df)} row(s). Need at least {input_len + horizon} after preprocessing.")

    raw_dates = df[date_col]
    dates = pd.to_datetime(raw_dates, errors="coerce")
    bad = dates.isna()
    if bad.any():
        raise InputError(f"Unparsable dates in {date_col!r}: {int(bad.sum())} row(s), e.g. {_examples(raw_dates[bad])}.")
    raw_vals = df[value_col]
    vals = pd.to_numeric(raw_vals, errors="coerce")
    non_num = vals.isna() & raw_vals.notna() & (raw_vals.astype(str).str.strip() != "")
    if non_num.any():
        raise InputError(f"Non-numeric values in {value_col!r}: {int(non_num.sum())} row(s), e.g. "
                         f"{_examples(raw_vals[non_num])}.")
    if vals.notna().sum() < 3:
        raise InputError(f"Column {value_col!r} has fewer than 3 numeric values.")
    dup = dates.duplicated(keep=False)
    if dup.any():
        raise InputError(f"Duplicate dates: {int(dates[dup].nunique())} date(s) repeated, e.g. "
                         f"{_examples(dates[dup].dt.strftime('%Y-%m-%d %H:%M').unique())}.")
    if not dates.is_monotonic_increasing:
        i = int(np.argmax(dates.diff().dt.total_seconds().to_numpy()[1:] < 0)) + 1
        raise InputError(f"Dates are not sorted ascending (row {i + 1}: {dates.iloc[i]} comes after "
                         f"{dates.iloc[i - 1]}). Sort the file by date first.")

    freq, step = meta["frequency"], STEP[meta["frequency"]]
    diffs = dates.diff().dropna()
    if (diffs < step).any() or ((diffs % step) != pd.Timedelta(0)).any():
        raise InputError(f"This domain expects {FREQ_NAME[freq]} data, but some timestamps are closer together than "
                         f"one step or not aligned to whole {'hours' if freq == 'h' else 'days'}.")
    if diffs.median() != step:
        raise InputError(f"This domain expects {FREQ_NAME[freq]} data; the median spacing in the file is "
                         f"{diffs.median()}.")

    frame = pd.DataFrame({meta["value_name"]: vals.to_numpy(dtype=float)}, index=pd.DatetimeIndex(dates))
    notes = []
    if meta["preprocess"] == "close_to_log_return":
        if (frame["Close"].dropna() <= 0).any():
            raise InputError("Closing prices must be positive to compute log-returns.")
        out = close_to_log_return(frame)["log_return"]
        if meta.get("transform"):
            out = pd.Series(SERIES_TRANSFORMS[meta["transform"]](out.to_numpy()), index=out.index)
        notes.append(f"{len(frame)} closing prices → {len(out)} log-returns"
                     + (" (absolute value, as in training)" if meta.get("transform") else "") + ".")
    else:
        n_before = int(frame.iloc[:, 0].notna().sum())
        out = PREPROCESS[meta["preprocess"]](frame, meta["value_name"])[meta["value_name"]]
        if len(out) != n_before:
            notes.append(f"{len(out) - n_before} missing {FREQ_NAME[freq]} step(s) filled as in training preprocessing.")
    if len(out) < input_len + horizon:
        raise InputError(f"Series below input_len + horizon: {len(out)} point(s) after preprocessing, need "
                         f"{input_len} + {horizon} = {input_len + horizon}.")
    return out.astype(float), notes


def upload_stats(series):
    """Stats of the first 70% of the series (chronological), mirroring target-train stats."""
    train, val, test = chronological_split(series.to_numpy())
    return normalize(train, val, test)[3]


def pick_stats(series, ckpt, mode):
    return upload_stats(series) if mode == "upload" else source_stats(ckpt)


def future_index(series, freq, horizon):
    last = series.index[-1]
    if freq == "B":  # next trading days, weekends skipped (exchange holidays not known)
        return pd.bdate_range(last + pd.Timedelta(days=1), periods=horizon)
    return pd.date_range(last + STEP[freq], periods=horizon, freq=STEP[freq])


def forecast_last(model, series, stats, input_len, horizon, freq, holdout=True):
    """Forecast from the last input_len points (or the input_len points before the held-out final horizon)."""
    v = series.to_numpy()
    if holdout:
        inp, actual = v[-(input_len + horizon):-horizon], v[-horizon:]
        inp_idx, out_idx = series.index[-(input_len + horizon):-horizon], series.index[-horizon:]
    else:
        inp, actual = v[-input_len:], None
        inp_idx, out_idx = series.index[-input_len:], future_index(series, freq, horizon)
    pred = denorm(predict(model, apply_norm(inp[None, :].astype(np.float32), stats)), stats)[0]
    table = pd.DataFrame({"step": np.arange(1, horizon + 1), "time": out_idx, "forecast": pred})
    metrics = None
    if actual is not None:
        table["actual"] = actual
        table["error"] = pred - actual
        metrics = {k: evaluate_all(actual, pred)[k] for k in ("MAE", "RMSE")}
    return {"input": pd.Series(inp, index=inp_idx), "table": table, "metrics": metrics}


def backtest(model, series, stats, input_len, horizon):
    """Every window of the series (stride 1): forecasts vs actuals, MAE/RMSE, and the upload's train-mean baseline."""
    v = series.to_numpy().astype(np.float32)
    X, Y = make_windows(apply_norm(v, stats), input_len, horizon)
    pred, Y_true = denorm(predict(model, X), stats), denorm(Y, stats)
    first = pd.DataFrame({"time": series.index[input_len:input_len + len(X)], "actual": Y_true[:, 0],
                          "forecast": pred[:, 0]})
    pick = lambda m: {k: m[k] for k in ("MAE", "RMSE")}
    s70 = upload_stats(series)
    _, Y70 = make_windows(apply_norm(v, s70), input_len, horizon)
    return {
        "first_step": first, "n_windows": int(len(X)),
        "metrics_first_step": pick(evaluate_all(Y_true[:, 0], pred[:, 0])),
        "metrics_all_steps": pick(evaluate_all(Y_true, pred)),
        "baseline_train_mean_all_steps": pick(naive_mean_metrics(Y70, s70)),
        "baseline_train_mean_first_step": pick(naive_mean_metrics(Y70[:, :1], s70)),
        "train_mean": s70["mean"],
    }
