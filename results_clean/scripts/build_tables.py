"""
Regenerates every results_clean/ table from the experiment JSONs. No training, no fitting.
Baselines (naive train-mean, persistence / seasonal-naive, mean of last 5) are recomputed from the
processed CSVs with the same prepare_pair() windows the TCN runs used, and checked against the JSONs.

  set OMP_NUM_THREADS=2 && venv\\Scripts\\python -m results_clean.scripts.build_tables   (from repo root)
Writes only to results_clean/.
"""
import csv
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np

from src.evaluate import denorm, seasonal_naive
from src.experiment import prepare_pair
from src.run_experiments import DOMAIN_CONFIGS
from src.utils.metrics import evaluate_all

OUT = Path("results_clean")
EXP = Path("experiments")
FINAL = EXP / "final_3seed" / "tcn"
DOMAINS = ["energy", "climate", "finance", "finance_vol"]
PAIR = {"energy": "AEP->COMED", "climate": "Shimla->Delhi", "finance": "Nifty IT->Nifty Pharma (log-return)",
        "finance_vol": "Nifty IT->Nifty Pharma (|log-return|)"}
UNIT = {"energy": "MW", "climate": "degC", "finance": "log-return units", "finance_vol": "|log-return| units"}
SEEDS = [0, 1, 2]
RES = {"energy": 0.0, "climate": 0.0, "finance": 2e-5, "finance_vol": 2e-5}  # 2 rounding steps of 5-decimal JSON values

# ---- setting labels (canonical order) ----
NAIVE = "naive train-mean (target-train stats)"
SO_S = "source-only, source stats"
SO_T = "source-only, target-train stats"
C10 = "CORAL lambda=10"
C1 = "CORAL lambda=1 (sensitivity)"
ADA = "AdaRNN-style lambda=10"
TO = "target-only (labels, reference)"
PERSIST = {  # (label, kind, arg); kind "lag" uses src.evaluate.seasonal_naive, "meanlast" the mean of the last k inputs
    "energy": [("seasonal-naive lag 24 (same hour yesterday)", "lag", 24),
               ("seasonal-naive lag 168 (same hour last week)", "lag", 168)],
    "climate": [("persistence (last observed value repeated)", "lag", 1)],
    "finance": [("persistence (last observed value repeated)", "lag", 1),
                ("mean of last 5 inputs (repeated)", "meanlast", 5)],
    "finance_vol": [("persistence (last observed value repeated)", "lag", 1),
                    ("mean of last 5 inputs (repeated)", "meanlast", 5)],
}
PERSIST_DEF = (
    "For each target-test window with last observed step t and horizon step j=1..H: "
    "seasonal-naive with lag p predicts y_hat[t+j] = y[t + j - p*ceil(j/p)] (src.evaluate.seasonal_naive). "
    "p=1 is persistence: the last observed value y[t] repeated for all H steps. "
    "Energy (H=24): p=24 gives y[t+j-24] = same hour yesterday; p=168 gives y[t+j-168] = same hour last week. "
    "Mean of last 5 inputs: y_hat[t+j] = mean(y[t-4..t]) for all j. "
    "Windows: the same target-test windows the TCN runs score on (prepare_pair -> prepare_domain: chronological "
    "70/15/15 split of the target series, windows built inside the test split only; inputs and targets both "
    "inside the test split). Lagged values never reach past t, and with these lags they stay inside the test split.")


def load(p):
    return json.load(open(p))


def runs_for(domain):
    """Per-seed target-test metrics for every model setting, read from the JSONs. {setting: {seed: {MAE,RMSE}}}"""
    out = {s: {} for s in [SO_S, SO_T, C10, C1, ADA, TO]}
    meta = {}
    for s in SEEDS:
        f = FINAL / "runs" / f"{domain}_seed{s}.json"
        if f.exists():
            j = load(f)
            out[SO_S][s] = j["source_only_on_target_test"]["source_stats"]
            out[SO_T][s] = j["source_only_on_target_test"]["target_stats"]
            out[TO][s] = j["target_only_on_target_test"]["target_stats"]
            meta.setdefault("final_3seed/tcn", []).append((f, j))
        for lab, d, key in [(C10, "coral", "coral_on_target_test"), (C1, "coral_lambda1", "coral_on_target_test"),
                            (ADA, "adarnn", "adarnn_on_target_test")]:
            f = EXP / d / domain / f"seed{s}.json"
            if f.exists():
                j = load(f)
                out[lab][s] = j[key]["target_stats"]
                meta.setdefault(d, []).append((f, j))
    return out, meta


def mstd(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


# =================================================================== baselines + alignment
def baselines(domain, summary, log):
    cfg = DOMAIN_CONFIGS[domain]
    src, tgt_src, tgt_tgt, series = prepare_pair(cfg)
    st = tgt_tgt["norm_stats"]
    Y_true = denorm(tgt_tgt["Y_test"].astype(np.float64), st)
    n_json = summary["domains"][domain]["baselines"]["n_target_test_windows"]
    rows = {}
    # naive train-mean, target-train stats: predict the target-train mean
    rows[NAIVE] = evaluate_all(Y_true, np.full_like(Y_true, st["mean"]))
    log.append(f"[{domain}] model test windows (prepare_pair, target-train stats): {len(Y_true)} x H={Y_true.shape[1]}; "
               f"JSON n_target_test_windows = {n_json}; equal: {len(Y_true) == n_json}")
    for lab, kind, arg in PERSIST[domain]:
        if kind == "lag":
            sn_true, sn_pred = seasonal_naive(series, cfg["input_len"], cfg["horizon"], arg)
        else:
            X = denorm(tgt_tgt["X_test"].astype(np.float64), st)
            sn_true = Y_true
            sn_pred = np.repeat(X[:, -arg:].mean(axis=1, keepdims=True), Y_true.shape[1], axis=1)
        same_n = sn_true.shape == Y_true.shape
        maxdiff = float(np.max(np.abs(sn_true - Y_true))) if same_n else float("nan")
        ok = same_n and np.allclose(sn_true, Y_true, rtol=1e-4, atol=1e-3)
        log.append(f"[{domain}] {lab}: windows {sn_true.shape[0]} x {sn_true.shape[1]} vs model {Y_true.shape[0]} x "
                   f"{Y_true.shape[1]}; max |y_true(baseline) - y_true(model windows)| = {maxdiff:.2e}; "
                   f"ALIGNED: {ok}")
        assert ok, f"{domain} {lab} misaligned"
        rows[lab] = evaluate_all(sn_true, sn_pred)
    # cross-check against the numbers already in summary.json where they exist
    b = summary["domains"][domain]["baselines"]
    chk = [(NAIVE, b["naive_train_mean"]["target_stats"])]
    for lab, kind, arg in PERSIST[domain]:
        if kind == "lag" and f"period_{arg}" in b.get("seasonal_naive", {}):
            chk.append((lab, b["seasonal_naive"][f"period_{arg}"]))
        if kind == "meanlast" and f"mean_of_last_{arg}_inputs" in b:
            chk.append((lab, b[f"mean_of_last_{arg}_inputs"]["target_stats"]))
    for lab, ref in chk:
        same = all(abs(rows[lab][m] - ref[m]) <= 1e-5 * max(1, abs(ref[m])) for m in ("MAE", "RMSE"))
        log.append(f"[{domain}] recomputed {lab}: MAE {rows[lab]['MAE']} RMSE {rows[lab]['RMSE']} | "
                   f"summary.json MAE {ref['MAE']} RMSE {ref['RMSE']} | match: {same}")
    for lab, kind, arg in PERSIST[domain]:
        if not any(lab == c[0] for c in chk):
            log.append(f"[{domain}] {lab}: not in summary.json (new baseline, computed here)")
    return rows


# =================================================================== provenance
def provenance(all_meta):
    groups = {}
    for domain, meta in all_meta.items():
        for g, lst in meta.items():
            for f, j in lst:
                di = j.get("device_info", {})
                groups.setdefault(g, []).append({
                    "domain": domain, "seed": j.get("seed"), "device": di.get("device"), "threads": di.get("cpu_threads"),
                    "torch": di.get("torch"),
                    "mtime": dt.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M"),
                    "file": str(f).replace("\\", "/")})
    L = ["# Provenance of every result row group", "",
         "Generated by `results_clean/scripts/build_tables.py` from the run JSONs. No JSON carries a run date or",
         "timestamp field, so **date = file modification time** of the JSON (local time). `threads` is",
         "`device_info.cpu_threads` = `torch.get_num_threads()` recorded at run time.", ""]
    flags = []
    order = ["final_3seed/tcn", "coral", "coral_lambda1", "adarnn"]
    for g in order:
        rs = groups.get(g, [])
        L += [f"## {g}", "", "| domain | seeds present | device | torch | threads | date (JSON mtime) |", "|---|---|---|---|---|---|"]
        for d in DOMAINS:
            rr = [r for r in rs if r["domain"] == d]
            if not rr:
                L.append(f"| {d} | none | | | | |")
                continue
            seeds = sorted(r["seed"] for r in rr)
            th = sorted({r["threads"] for r in rr})
            dates = sorted(r["mtime"] for r in rr)
            L.append(f"| {d} | {seeds} | {','.join(sorted({r['device'] for r in rr}))} | "
                     f"{','.join(sorted({r['torch'] for r in rr}))} | {','.join(map(str, th))} | "
                     f"{dates[0]} .. {dates[-1]} |")
        th_all = sorted({r["threads"] for r in rs})
        if len(th_all) > 1:
            per = {d: sorted({r['threads'] for r in rs if r['domain'] == d}) for d in DOMAINS}
            msg = (f"**FLAG: {g} mixes torch thread counts {th_all}** "
                   f"({'; '.join(f'{d}: {v}' for d, v in per.items())}). Thread count can change floating-point "
                   f"summation order (CLAUDE.md amendment 2026-10-25 (d)); no experiment setting differs.")
            flags.append(msg)
            L += ["", msg]
        L.append("")
    all_th = sorted({r["threads"] for g in groups for r in groups[g]})
    L += ["## Cross-group thread summary", ""]
    for g in order:
        L.append(f"- {g}: threads {sorted({r['threads'] for r in groups.get(g, [])})}")
    L += ["", f"Thread counts across all groups: {all_th}. The baseline rows (naive, persistence) are deterministic "
          "numpy computations, recomputed here with OMP_NUM_THREADS=2; thread count does not affect them.",
          "", "Devices: every run JSON records device = cpu (no CUDA).", ""]
    L += ["## Per-file listing", "", "| group | domain | seed | threads | date | file |", "|---|---|---|---|---|---|"]
    for g in order:
        for r in sorted(groups.get(g, []), key=lambda r: (DOMAINS.index(r["domain"]), r["seed"])):
            L.append(f"| {g} | {r['domain']} | {r['seed']} | {r['threads']} | {r['mtime']} | `{r['file']}` |")
    (OUT / "PROVENANCE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return flags


# =================================================================== canonical
def fmt(x, domain, kind="val"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if kind == "pct":
        return f"{x:.1f}"
    if domain.startswith("finance"):
        return f"{x:.6f}"
    return f"{x:.2f}" if domain == "energy" else f"{x:.3f}"


def canonical(per_domain_rows):
    cols = ["domain", "pair", "setting", "n_seeds", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std", "unit",
            "MAE_pct_naive", "RMSE_pct_naive", "MAE_pct_best_baseline", "RMSE_pct_best_baseline",
            "best_baseline_MAE_row", "best_baseline_RMSE_row", "MAE_reduction_vs_SO_src_pct", "RMSE_reduction_vs_SO_src_pct",
            "per_seed_MAE", "per_seed_RMSE", "source"]
    table = []
    for d in DOMAINS:
        rows = per_domain_rows[d]
        base_labels = [NAIVE] + [p[0] for p in PERSIST[d]]
        best = {m: min(base_labels, key=lambda l: rows[l]["MAE" if m == "MAE" else "RMSE"]["mean"]) for m in ("MAE", "RMSE")}
        for lab, r in rows.items():
            rec = {"domain": d, "pair": PAIR[d], "setting": lab, "n_seeds": r["n"], "unit": UNIT[d],
                   "source": r["source"]}
            for m in ("MAE", "RMSE"):
                rec[f"{m}_mean"], rec[f"{m}_std"] = r[m]["mean"], r[m]["std"]
                rec[f"{m}_pct_naive"] = 100 * r[m]["mean"] / rows[NAIVE][m]["mean"]
                rec[f"{m}_pct_best_baseline"] = 100 * r[m]["mean"] / rows[best[m]][m]["mean"]
                rec[f"best_baseline_{m}_row"] = best[m]
                rec[f"{m}_reduction_vs_SO_src_pct"] = 100 * (rows[SO_S][m]["mean"] - r[m]["mean"]) / rows[SO_S][m]["mean"]
                rec[f"per_seed_{m}"] = ";".join(str(v) for v in r[m]["per_seed"])
            table.append(rec)
    with open(OUT / "canonical.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rec in table:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in rec.items()})
    return table


def canonical_md(table, align_log, missing):
    L = ["# Canonical results (one source of truth)", "",
         "All numbers are on the **target test set**, original scale, regenerated from the run JSONs by",
         "`results_clean/scripts/build_tables.py` (machine-readable copy: `canonical.csv`, full precision).",
         "Mean and std are over seeds; std is the sample std (ddof=1). Per-seed metrics in the JSONs are rounded",
         "to 5 decimals (`src/utils/metrics.py`), so finance values carry only ~3 significant digits.", "",
         "## Definitions", "",
         f"- **{NAIVE}**: predict the target-train mean for every horizon step (= 0 in target-normalised space).",
         f"- **Persistence / seasonal-naive** (new for the final TCN runs, CLAUDE.md amendment 2026-11-15 (g)(b)): {PERSIST_DEF}",
         f"- **{SO_S}**: TCN trained on source, target inputs normalised with source-train stats.",
         f"- **{SO_T}**: same model, target inputs normalised with target-train stats (no target labels).",
         f"- **{C10}**: TCN + CORAL loss on get_features(), λ=10 (pre-registered λ choice), target-train stats. `experiments/coral/`.",
         f"- **{C1}**: same, λ=1, sensitivity check only. `experiments/coral_lambda1/`.",
         f"- **{ADA}**: AdaRNN-style (TDC periods + pairwise CORAL), λ=10, target-train stats. `experiments/adarnn/`.",
         f"- **{TO}**: TCN trained on target-train labels for a fixed epoch count (= source best epoch). Upper reference; uses target labels.",
         "- **% naive** = 100·mean / mean(naive train-mean row). **% best baseline** = 100·mean / min over the naive and",
         "  persistence/seasonal rows of the same metric (the row used is named in the CSV). **reduction vs SO-src** =",
         "  100·(mean(SO-src) − mean) / mean(SO-src); positive = lower error.",
         "- Units: MW (energy), °C (climate), log-return units (finance; a daily log-return, 0.01 ≈ 1%), |log-return| units (finance_vol).",
         "  MAPE/sMAPE are not reported (meaningless for log-returns; kept out everywhere for consistency).", ""]
    if missing:
        L += ["## Missing", ""] + [f"- {m}" for m in missing] + [""]
    else:
        L += ["MAE is present in every finance JSON, so MAE and RMSE are both reported for finance.", ""]
    for d in DOMAINS:
        rr = [t for t in table if t["domain"] == d]
        L += [f"## {d}: {PAIR[d]} (unit: {UNIT[d]})", "",
              "| setting | n | MAE mean ± std | RMSE mean ± std | MAE % naive | RMSE % naive | MAE % best baseline | RMSE % best baseline | MAE red. vs SO-src % | RMSE red. vs SO-src % |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for t in rr:
            def pm(m):
                s = fmt(t[f"{m}_std"], d)
                return fmt(t[f"{m}_mean"], d) + (f" ± {s}" if s else "")
            L.append(f"| {t['setting']} | {t['n_seeds']} | {pm('MAE')} | {pm('RMSE')} | {fmt(t['MAE_pct_naive'], d, 'pct')} | "
                     f"{fmt(t['RMSE_pct_naive'], d, 'pct')} | {fmt(t['MAE_pct_best_baseline'], d, 'pct')} | "
                     f"{fmt(t['RMSE_pct_best_baseline'], d, 'pct')} | {fmt(t['MAE_reduction_vs_SO_src_pct'], d, 'pct')} | "
                     f"{fmt(t['RMSE_reduction_vs_SO_src_pct'], d, 'pct')} |")
        L += ["", f"Best baseline: MAE = {rr[0]['best_baseline_MAE_row']}; RMSE = {rr[0]['best_baseline_RMSE_row']}.", ""]
    L += ["## Window alignment check (baselines vs model test windows)", "", "```"] + align_log + ["```", ""]
    (OUT / "canonical.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# =================================================================== headline / decomposition
def headline(rows_all):
    L = ["# Headline: source-only vs adapted", "",
         "'Adapted' = target-train-stat normalisation + CORAL λ=10 (the loss chosen by the pre-registered rule;",
         "CLAUDE.md amendment 2026-11-15 (g)(a): this framing is a post-results presentation choice). Target test set,",
         "mean over seeds 0-2, original scale. All numbers from `canonical.csv`.", "",
         "Notation: A = source-only, source stats; B = source-only, target-train stats; C = adapted method",
         "(CORAL λ=10 or AdaRNN-style λ=10, both with target-train stats); T = target-only (uses target labels).", "",
         "- total reduction % = 100·(A − C)/A",
         "- normalisation part = A − B; method part = B − C; total = A − C = (A − B) + (B − C)",
         "- share of total drop: norm = (A − B)/(A − C), method = (B − C)/(A − C)",
         "- gap to target-only = A − T; share of gap closed: norm = (A − B)/(A − T), method = (B − C)/(A − T), total = (A − C)/(A − T)",
         "- Shares are only meaningful when there is a drop (A − C > 0) or a gap (A − T > 0) clearly larger than noise. A share is",
         "  marked **n/m** (not meaningful) when its denominator is ≤ 0, or ≤ the larger seed std of the two rows involved, or",
         "  below 2 rounding steps of the JSON values (2e-5 for finance/finance_vol, whose per-seed metrics are stored to 5 decimals).", ""]
    summary = []
    for meth in [C10, ADA]:
        L += [f"## Method: {meth}", "",
              "| domain | metric | A: SO-src | B: SO-tgt | C: method | T: target-only | total red. % = 100(A−C)/A | norm part A−B (share of A−C) | method part B−C (share of A−C) | gap A−T | gap closed: norm (A−B)/(A−T) | method (B−C)/(A−T) | total (A−C)/(A−T) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for d in DOMAINS:
            r = rows_all[d]
            for m in ("MAE", "RMSE"):
                A, B, C, T = (r[k][m]["mean"] for k in (SO_S, SO_T, meth, TO))
                sA, sC, sT = (r[k][m]["std"] for k in (SO_S, meth, TO))
                tot, nrm, mth, gap = A - C, A - B, B - C, A - T
                nm_tot = tot <= 0 or abs(tot) <= max(sA, sC) or abs(tot) < RES[d]
                nm_gap = gap <= 0 or abs(gap) <= max(sA, sT) or abs(gap) < RES[d]
                f = lambda x: fmt(x, d)
                sh = lambda a, b, bad: ("n/m " if bad else "") + f"{100 * a / b:.1f}%" if b != 0 else "n/a"
                L.append(f"| {d} | {m} | {f(A)} | {f(B)} | {f(C)} | {f(T)} | {100 * tot / A:.1f} | {f(nrm)} ({sh(nrm, tot, nm_tot)}) | "
                         f"{f(mth)} ({sh(mth, tot, nm_tot)}) | {f(gap)} | {sh(nrm, gap, nm_gap)} | {sh(mth, gap, nm_gap)} | {sh(tot, gap, nm_gap)} |")
                summary.append(dict(method=meth, domain=d, metric=m, A=A, B=B, C=C, T=T, total_pct=100 * tot / A,
                                    norm_share=nrm / tot if tot else np.nan, meth_share=mth / tot if tot else np.nan,
                                    gap_closed=tot / gap if gap else np.nan, nm_tot=nm_tot, nm_gap=nm_gap))
        L.append("")
    L += ["## Sanity check against the old slides (verified, not copied)", ""]
    e, c = rows_all["energy"], rows_all["climate"]
    L.append(f"- energy MAE: A = {e[SO_S]['MAE']['mean']:.1f} -> C (CORAL λ=10) = {e[C10]['MAE']['mean']:.1f} MW "
             f"(slides said roughly 803 -> 555)")
    L.append(f"- climate MAE: A = {c[SO_S]['MAE']['mean']:.3f} -> C = {c[C10]['MAE']['mean']:.3f} °C (slides said roughly 2.95 -> 1.70)")
    L.append("")
    (OUT / "headline.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return summary


# =================================================================== noise rule
RULE = ("For a method M compared with the reference row R = 'source-only, target-train stats' on one (domain, metric): "
        "d = mean(M) − mean(R) over seeds; s_R = std of R; s_M = std of M (ddof=1). "
        "Verdict: |d| > max(s_R, s_M) → \"exceeds both stds\"; s_R < |d| ≤ s_M → \"exceeds the target-stats std only\"; "
        "|d| ≤ min(s_R, s_M) → \"within both\". The remaining case s_M < |d| ≤ s_R does not fit the fixed list and is "
        "printed as \"within the target-stats std (exceeds method std)\" so it is never silently relabelled. "
        "The verdict is about magnitude; the direction column says whether M is lower (better) or higher (worse) error. "
        "Paired per-seed differences use the same seed number in both rows (M seed s − R seed s); they are reported "
        "as extra information and do not change the verdict.")


def noise_rule(rows_all):
    L = ["# Single rule for \"gain vs noise\"", "", RULE, "",
         "This is the only rule used in results_clean/ (headline, APPROVED_CLAIMS). Target test set, seeds 0-2.", "",
         "| domain | method vs SO-tgt | metric | mean(M) | mean(R) | d = mean diff | paired diffs (s0; s1; s2) | paired mean | paired std | s_R (target-stats std) | s_M | max(s_R, s_M) | direction | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    verdicts = []
    for d in DOMAINS:
        r = rows_all[d]
        R = r[SO_T]
        for meth in [C10, C1, ADA, TO]:
            M = r[meth]
            for m in ("MAE", "RMSE"):
                dd = M[m]["mean"] - R[m]["mean"]
                sR, sM = R[m]["std"], M[m]["std"]
                pairs = [a - b for a, b in zip(M[m]["per_seed"], R[m]["per_seed"])]
                pm, ps = mstd(pairs)
                a = abs(dd)
                if a > max(sR, sM):
                    v = "exceeds both stds"
                elif sR < a <= sM:
                    v = "exceeds the target-stats std only"
                elif a <= min(sR, sM):
                    v = "within both"
                else:
                    v = "within the target-stats std (exceeds method std)"
                dirn = "lower error" if dd < 0 else ("higher error" if dd > 0 else "equal")
                f = lambda x: fmt(x, d)
                L.append(f"| {d} | {meth} | {m} | {f(M[m]['mean'])} | {f(R[m]['mean'])} | {f(dd)} | "
                         f"{'; '.join(f(p) for p in pairs)} | {f(pm)} | {f(ps)} | {f(sR)} | {f(sM)} | {f(max(sR, sM))} | {dirn} | {v} |")
                verdicts.append(dict(domain=d, method=meth, metric=m, d=dd, sR=sR, sM=sM, verdict=v, direction=dirn))
    # finance_vol statement
    fv = load(EXP / "bakeoff" / "finance_vol_val_baselines.json")
    naive_val = fv["baselines"]["naive_train_mean"]["mse_norm"]
    fin_vals = [load(FINAL / "runs" / f"finance_vol_seed{s}.json")["source_val"]["mse_norm"] for s in SEEDS]
    fin_raw = [load(FINAL / "runs" / f"finance_seed{s}.json")["source_val"]["mse_norm"] for s in SEEDS]
    naive_raw = float(np.mean(prepare_pair(DOMAIN_CONFIGS["finance"])[0]["Y_val"].astype(np.float64) ** 2))
    naive_fv_re = float(np.mean(prepare_pair(DOMAIN_CONFIGS["finance_vol"])[0]["Y_val"].astype(np.float64) ** 2))
    r = rows_all["finance_vol"]
    L += ["", "## finance_vol (and finance) statement, scoped", "",
          f"- **Source validation** (Nifty IT |log-return|, normalised): naive train-mean val MSE = {naive_val:.4f} "
          f"(`experiments/bakeoff/finance_vol_val_baselines.json`); final TCN source-only val MSE per seed = "
          f"{', '.join(f'{v:.4f}' for v in fin_vals)} (`final_3seed/tcn/runs/finance_vol_seed*.json`). "
          f"{'No' if min(fin_vals) >= naive_val else 'Some'} seed beats the naive val MSE; in the bake-off no backbone did "
          f"(`finance_vol_val_baselines.json`). This is why finance_vol is a negative control (CLAUDE.md 2026-10-23).",
          f"- **Target test** (Nifty Pharma |log-return|): RMSE: naive {r[NAIVE]['RMSE']['mean']:.5f}, persistence "
          f"{r['persistence (last observed value repeated)']['RMSE']['mean']:.5f}, mean of last 5 "
          f"{r['mean of last 5 inputs (repeated)']['RMSE']['mean']:.5f}, SO-tgt {r[SO_T]['RMSE']['mean']:.5f} ± {r[SO_T]['RMSE']['std']:.5f}, "
          f"CORAL λ=10 {r[C10]['RMSE']['mean']:.5f} ± {r[C10]['RMSE']['std']:.5f}. The target-test comparison is reported, "
          "not used to choose anything; the negative-control label rests on source validation.",
          f"- Raw finance, source validation: naive train-mean val MSE = {naive_raw:.4f} (recomputed: mean(Y_val²) in source-normalised "
          f"space; finance_vol recomputed the same way = {naive_fv_re:.4f}, JSON {naive_val:.4f}); TCN val MSE per seed = "
          f"{', '.join(f'{v:.4f}' for v in fin_raw)}. {'No seed beats' if min(fin_raw) >= naive_raw else 'At least one seed is below'} the naive val MSE.", ""]
    (OUT / "noise_rule.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return verdicts


# =================================================================== main
def main():
    OUT.mkdir(exist_ok=True)
    summary = load(FINAL / "summary.json")
    rows_all, all_meta, align_log, missing = {}, {}, [], []
    for d in DOMAINS:
        runs, meta = runs_for(d)
        all_meta[d] = meta
        rows = {}
        b = baselines(d, summary, align_log)
        rows[NAIVE] = b[NAIVE]
        for lab, _, _ in PERSIST[d]:
            rows[lab] = b[lab]
        # deterministic rows
        for lab in list(rows):
            rows[lab] = {"n": "deterministic", "source": "recomputed: prepare_pair target-test windows (cross-checked vs final_3seed/tcn/summary.json where present)",
                         **{m: {"mean": rows[lab][m], "std": float("nan"), "per_seed": [rows[lab][m]]} for m in ("MAE", "RMSE")}}
        src_name = {SO_S: "final_3seed/tcn/runs/{d}_seed*.json source_only_on_target_test.source_stats",
                    SO_T: "final_3seed/tcn/runs/{d}_seed*.json source_only_on_target_test.target_stats",
                    C10: "coral/{d}/seed*.json coral_on_target_test.target_stats",
                    C1: "coral_lambda1/{d}/seed*.json coral_on_target_test.target_stats",
                    ADA: "adarnn/{d}/seed*.json adarnn_on_target_test.target_stats",
                    TO: "final_3seed/tcn/runs/{d}_seed*.json target_only_on_target_test.target_stats"}
        for lab in [SO_S, SO_T, C10, C1, ADA, TO]:
            per = runs[lab]
            if sorted(per) != SEEDS:
                missing.append(f"{d} / {lab}: seeds present {sorted(per)}")
            rec = {"n": len(per), "source": "experiments/" + src_name[lab].format(d=d)}
            for m in ("MAE", "RMSE"):
                vals = [per[s].get(m) for s in sorted(per)]
                if any(v is None for v in vals):
                    missing.append(f"{d} / {lab}: {m} missing in JSON")
                    vals = [v for v in vals if v is not None]
                mu, sd = mstd(vals)
                rec[m] = {"mean": mu, "std": sd, "per_seed": vals}
            rows[lab] = rec
        rows_all[d] = rows
    flags = provenance(all_meta)
    table = canonical(rows_all)
    canonical_md(table, align_log, missing)
    hl = headline(rows_all)
    verdicts = noise_rule(rows_all)
    json.dump({"rows": rows_all, "headline": hl, "verdicts": verdicts, "flags": flags, "missing": missing,
               "align_log": align_log}, open(OUT / "_tables.json", "w"), indent=1, default=float)
    print("\n".join(align_log))
    print("\nFLAGS:", *flags, sep="\n")
    print("MISSING:", missing)


if __name__ == "__main__":
    main()
