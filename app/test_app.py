"""
App checks with Streamlit's AppTest (no browser, no training):  python app/test_app.py
Every domain x model x normalisation on its bundled sample, an upload happy path, forecast without holdout, and
bad uploads that must produce an st.error without an exception. Prints PASS/FAIL per case; exit code 1 on failure.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
import core  # noqa: E402

APP = str(APP_DIR / "app.py")
TIMEOUT = 300
results = []


def new_app():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    return at


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def set_sidebar(at, domain, model="Source-only", norm=None):
    at.sidebar.selectbox[0].select(domain).run()
    at.sidebar.radio[0].set_value(model).run()
    if norm:
        at.sidebar.radio[1].set_value(norm).run()
    return at


def upload(at, data, name="upload.csv"):
    at.radio[0].set_value("Upload CSV").run()
    at.file_uploader[0].set_value((name, data, "text/csv")).run()
    return at


def errors(at):
    return [e.value for e in at.error]


def demo_uploads():
    """Upload-format CSVs (bytes) built from data/processed: renamed columns plus an irrelevant constant text
    column placed first, and a 20-day file that is below input_len + horizon."""
    proc = APP_DIR.parent / "data" / "processed"
    iso = lambda s, fmt="%Y-%m-%d": pd.to_datetime(s).dt.strftime(fmt)
    c = pd.read_csv(proc / "climate" / "shimla_clean.csv")
    e = pd.read_csv(proc / "energy" / "aep_clean.csv").tail(1500)
    f = pd.read_csv(proc / "finance" / "nifty_it_clean.csv").tail(400)
    frames = {
        "demo_climate_upload.csv": pd.DataFrame({"station": "Shimla", "day": iso(c["time"].tail(300)),
                                                 "temperature_c": c["temp"].tail(300).values}),
        "demo_energy_upload.csv": pd.DataFrame({"region": "AEP", "timestamp": iso(e["datetime"], "%Y-%m-%d %H:%M:%S"),
                                                "demand_mw": e["load_mw"].values}),
        "demo_finance_upload.csv": pd.DataFrame({"Note": "NIFTY IT", "Date": iso(f["Date"]),
                                                 "Index_Close": f["Close"].values}),
        "demo_bad_too_short.csv": pd.DataFrame({"day": iso(c["time"].tail(20)), "temperature_c": c["temp"].tail(20).values}),
    }
    return {name: df.to_csv(index=False).encode() for name, df in frames.items()}


def picker(at, label):
    return next(s for s in at.selectbox if s.label.startswith(label))


def check_defaults(at, want, expect_error=None, not_offered=None):
    """Default (date, value) picks equal `want`; then either 8 metrics, or an st.error containing expect_error."""
    date_box, value_box = picker(at, "Date column"), picker(at, "Value column")
    picked, errs, exc = (date_box.value, value_box.value), errors(at), [e.value for e in at.exception]
    ok = picked == want and not exc
    if not_offered:
        ok = ok and not_offered not in date_box.options
    ok = ok and (bool(errs) and expect_error in errs[0] if expect_error else not errs and len(at.metric) == 8)
    detail = f"defaults {picked}" + (f" | {errs[0][:90]}" if errs else f", {len(at.metric)} metrics")
    return ok, detail + (f" | exception: {exc}" if exc else "")


def check_sample(domain, model, norm):
    t0 = time.perf_counter()
    at = new_app()
    set_sidebar(at, domain, model, norm)
    errs, exc = errors(at), [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    ok = not errs and not exc and len(at.metric) == 8 and len(at.dataframe) == 1
    active = " ".join(m.value for m in at.sidebar.markdown)
    ok = ok and model in active and domain in active
    return ok, f"{len(at.metric)} metrics, {time.perf_counter() - t0:.0f}s" + (f" | {errs or exc}" if errs or exc else "")


def main():
    man = core.load_manifest()
    norms = ["First 70% of the uploaded series (default)", "Checkpoint's source stats"]
    for d in ["climate", "energy", "finance_vol", "finance"]:
        for model in ["Source-only", "CORAL-adapted"]:
            for norm in norms:
                try:
                    ok, detail = check_sample(d, model, norm)
                except Exception as e:
                    ok, detail = False, f"{type(e).__name__}: {e}"
                record(f"sample | {d} | {model} | {norm.split(' (')[0]}", ok, detail)

    # upload-format stats on the full Delhi sample must equal the target-train stats used in training
    ck = torch.load(core.CKPT_DIR / man["domains"]["climate"]["checkpoints"]["coral"]["file"], map_location="cpu")
    s, _ = core.parse_series(pd.read_csv(core.SAMPLE_DIR / "climate_sample.csv"), "date", "temp",
                             man["domains"]["climate"], ck["input_len"], ck["horizon"])
    got, want = core.upload_stats(s), ck["norm_stats"]["target_train"]
    record("climate sample first-70% stats == training target-train stats",
           abs(got["mean"] - want["mean"]) < 1e-9 and abs(got["std"] - want["std"]) < 1e-9, f"{got} vs {want}")

    # upload happy path: extra column, renamed columns, picked via selectboxes
    df = pd.read_csv(core.SAMPLE_DIR / "climate_sample.csv").rename(columns={"date": "when", "temp": "t_mean"})
    df.insert(1, "station", "delhi")
    at = upload(set_sidebar(new_app(), "climate", "CORAL-adapted"), df.to_csv(index=False).encode())
    at.selectbox[0].select("when").run()
    at.selectbox[1].select("t_mean").run()
    record("upload | climate | picked columns", not errors(at) and not at.exception and len(at.metric) == 8,
           str(errors(at) or [e.value for e in at.exception] or f"{len(at.metric)} metrics"))

    # default column picks: every demo upload must work with no clicks; constant text columns are never offered as date
    demo = demo_uploads()
    demo_cases = [("demo_climate_upload.csv", "climate", ("day", "temperature_c"), None),
                  ("demo_energy_upload.csv", "energy", ("timestamp", "demand_mw"), None),
                  ("demo_finance_upload.csv", "finance_vol", ("Date", "Index_Close"), None),
                  ("demo_finance_upload.csv", "finance", ("Date", "Index_Close"), None),
                  ("demo_bad_too_short.csv", "climate", ("day", "temperature_c"), "below input_len + horizon")]
    for name, domain, want, expect in demo_cases:
        try:
            at = upload(set_sidebar(new_app(), domain), demo[name], name)
            record(f"demo upload, no clicks | {name} | {domain}", *check_defaults(at, want, expect))
        except Exception as e:
            record(f"demo upload, no clicks | {name} | {domain}", False, f"{type(e).__name__}: {e}")

    df = pd.DataFrame({"comment": [f"note {i}" for i in range(100)], "station": "x",
                       "date": pd.date_range("2020-01-01", periods=100, freq="D").strftime("%Y-%m-%d"),
                       "temp": [20 + (i % 7) for i in range(100)]})
    try:
        at = upload(set_sidebar(new_app(), "climate"), df.to_csv(index=False).encode(), "text_first.csv")
        record("upload, no clicks | text columns first", *check_defaults(at, ("date", "temp"), None, "station"))
    except Exception as e:
        record("upload, no clicks | text columns first", False, f"{type(e).__name__}: {e}")

    at = set_sidebar(new_app(), "energy")
    at.checkbox[0].uncheck().run()
    tbl = at.dataframe[0].value
    record("forecast beyond end (no holdout) | energy", "actual" not in tbl.columns and len(tbl) == 24
           and not errors(at), f"columns {list(tbl.columns)}, {len(tbl)} rows")

    day = lambda n, start="2020-01-01": pd.date_range(start, periods=n, freq="D").strftime("%Y-%m-%d")
    good = pd.DataFrame({"date": day(100), "temp": range(100)})
    bad_cases = {
        "too short (2 rows)": ("climate", b"date,temp\n2020-01-01,1\n2020-01-02,2\n", "too short"),
        "single column": ("climate", b"date\n2020-01-01\n2020-01-02\n", "at least two columns"),
        "non-numeric values": ("climate", good.assign(temp=good["temp"].astype(str).where(good.index != 5, "warm"))
                               .to_csv(index=False).encode(), "Non-numeric"),
        "unparsable dates": ("climate", good.assign(date=good["date"].where(good.index != 7, "32/13/2020"))
                             .to_csv(index=False).encode(), "Unparsable dates"),
        "duplicate dates": ("climate", pd.concat([good, good.iloc[[10]]]).sort_values("date")
                            .to_csv(index=False).encode(), "Duplicate dates"),
        "unsorted dates": ("climate", good.iloc[::-1].to_csv(index=False).encode(), "not sorted"),
        "below input_len + horizon (20 days)": ("climate", good.head(20).to_csv(index=False).encode(),
                                                "below input_len + horizon"),
        "energy given daily data": ("energy", good.to_csv(index=False).encode(), "expects hourly"),
        "climate given hourly data": ("climate", pd.DataFrame({"date": pd.date_range("2020-01-01", periods=500,
                                      freq="h"), "temp": 1.0}).to_csv(index=False).encode(), "expects daily"),
        "finance non-positive price": ("finance", good.assign(temp=good["temp"] - 50).rename(columns={"temp": "Close"})
                                       .to_csv(index=False).encode(), "positive"),
        "not a CSV (binary)": ("climate", bytes(range(256)) * 4, ""),
    }
    for name, (domain, data, expect) in bad_cases.items():
        try:
            at = upload(set_sidebar(new_app(), domain), data, "bad.csv")
            errs, exc = errors(at), [e.value for e in at.exception]
            ok = bool(errs) and not exc and all(expect in e for e in errs[:1]) and "Traceback" not in " ".join(errs)
            detail = (errs[0][:110] if errs else "no st.error") + (f" | exception: {exc}" if exc else "")
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        record(f"bad upload | {name}", ok, detail)

    n_fail = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - n_fail}/{len(results)} passed")
    return n_fail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
