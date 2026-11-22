"""
Read-only consistency scan: every result-like number in the slides, report, README and demo script is checked
against results_clean/canonical.csv (+ headline / noise-rule derived numbers). Writes results_clean/CONSISTENCY.md.

  venv\\Scripts\\python -m results_clean.scripts.consistency_scan
"""
import csv
import html
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

OUT = Path("results_clean")
T = json.load(open(OUT / "_tables.json"))

# ------------------------------------------------------------------ value pools
CANON = []  # (value, label)
with open(OUT / "canonical.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        base = f"{r['domain']} / {r['setting']}"
        for c in ["MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std", "MAE_pct_naive", "RMSE_pct_naive",
                  "MAE_pct_best_baseline", "RMSE_pct_best_baseline", "MAE_reduction_vs_SO_src_pct",
                  "RMSE_reduction_vs_SO_src_pct"]:
            if r[c] not in ("", "nan"):
                CANON.append((float(r[c]), f"{base} / {c}"))
        for c in ["per_seed_MAE", "per_seed_RMSE"]:
            vals = r[c].split(";")
            if r["n_seeds"] != "deterministic":
                for s, v in enumerate(vals):
                    CANON.append((float(v), f"{base} / {c.replace('per_seed_', '')} seed {s}"))
DERIVED = []
for h in T["headline"]:
    b = f"headline {h['domain']} {h['metric']} {h['method']}"
    DERIVED += [(h["total_pct"], f"{b} / total reduction %"), (100 * h["norm_share"], f"{b} / norm share of drop %"),
                (100 * h["meth_share"], f"{b} / method share of drop %"), (100 * h["gap_closed"], f"{b} / gap closed %"),
                (h["A"] - h["C"], f"{b} / A-C"), (h["B"] - h["C"], f"{b} / B-C"), (h["A"] - h["B"], f"{b} / A-B"),
                (h["A"] - h["T"], f"{b} / A-T"), (100 * (h["A"] - h["B"]) / h["A"], f"{b} / norm-only reduction %"),
                (100 * (h["B"] - h["C"]) / h["B"], f"{b} / method vs target-stats reduction %"),
                (100 * (h["C"] - h["T"]) / h["T"], f"{b} / method above target-only %")]
for v in T["verdicts"]:
    DERIVED.append((v["d"], f"noise {v['domain']} {v['method']} {v['metric']} / mean diff vs SO-tgt"))
# paired per-seed differences (noise_rule.md)
import numpy as _np
_SET = {"CORAL lambda=10", "CORAL lambda=1 (sensitivity)", "AdaRNN-style lambda=10", "target-only (labels, reference)"}
for d, rows in T["rows"].items():
    R = rows["source-only, target-train stats"]
    for meth in _SET:
        for m in ("MAE", "RMSE"):
            diffs = _np.array(rows[meth][m]["per_seed"]) - _np.array(R[m]["per_seed"])
            DERIVED.append((float(diffs.std(ddof=1)), f"noise {d} {meth} {m} / paired diff std"))
            for s_, x_ in enumerate(diffs):
                DERIVED.append((float(x_), f"noise {d} {meth} {m} / paired diff seed {s_}"))
POOL = CANON + DERIVED


def json_numbers():
    """Every number in the experiment JSONs (for 'sourced elsewhere' classification only)."""
    out = []
    for p in Path("experiments").rglob("*.json"):
        try:
            j = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        stack = [(j, p.as_posix())]
        while stack:
            o, path = stack.pop()
            if isinstance(o, dict):
                stack += [(v, f"{path}:{k}") for k, v in o.items() if k not in ("train_loss", "train_mse", "rss_mb", "train_coral", "train_tdm")]
            elif isinstance(o, list) and len(o) < 60:
                stack += [(v, path) for v in o]
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                out.append((float(o), path))
    # mean / std over seed files of the same group (e.g. experiments/coral/energy/seed*.json, final runs per domain)
    groups = defaultdict(list)
    for p in Path("experiments").rglob("*seed*.json"):
        key = re.sub(r"seed\d+", "seed*", p.as_posix())
        try:
            groups[key].append(json.load(open(p, encoding="utf-8")))
        except Exception:
            pass
    def leaves(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from leaves(v, f"{path}:{k}")
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            yield path, float(o)
    import numpy as np
    for key, js in groups.items():
        if len(js) < 2:
            continue
        acc = defaultdict(list)
        for j in js:
            for path, v in leaves(j):
                acc[path].append(v)
        for path, vs in acc.items():
            if len(vs) == len(js):
                out.append((float(np.mean(vs)), f"mean over {key}{path}"))
                out.append((float(np.std(vs, ddof=1)), f"std over {key}{path}"))
    return out


JSON_POOL = json_numbers()

# ------------------------------------------------------------------ extraction
NUM = re.compile(r"(?<![\w.])([-−–]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-−–]?\d+(?:\.\d+)?(?:[eE][-−]?\d+)?)(\s*%|\s*×)?")
KEY = re.compile(r"MAE|RMSE|MSE|std|±|\\pm|%|MW|°C|degC|error|reduc|improv|lower|higher|gap|naive|baseline|persist|"
                 r"seasonal|drop|closes|better|worse|CORAL|AdaRNN|target-only|source-only|seed", re.I)


def units():
    """Yields (file, location, text). Slides: one unit per slide body / notes, with paragraph text."""
    for f in ["slides/IE643_Project_Slides.pptx", "slides/IE643_Project_Slides_v2.pptx"]:
        if not Path(f).exists():
            continue
        z = zipfile.ZipFile(f)
        for n in z.namelist():
            m = re.match(r"ppt/(slides/slide|notesSlides/notesSlide)(\d+)\.xml$", n)
            if not m:
                continue
            xml = z.read(n).decode("utf-8")
            kind = "slide" if "slides/slide" in n else "notes"
            num = int(m.group(2))
            if kind == "notes":  # map notes file to the slide it belongs to
                rels = z.read(f"ppt/notesSlides/_rels/notesSlide{num}.xml.rels").decode("utf-8")
                sm = re.search(r"slides/slide(\d+)\.xml", rels)
                num = int(sm.group(1)) if sm else num
            for pi, para in enumerate(re.findall(r"<a:p>.*?</a:p>|<a:p .*?</a:p>", xml, re.S)):
                txt = html.unescape("".join(re.findall(r"<a:t>([^<]*)</a:t>", para)))
                if txt.strip():
                    yield f, f"slide {num} {kind}, para {pi + 1}", txt
    for f in ["report/overleaf/report.tex", "README.md", "demo_files/DEMO_SCRIPT.md"]:
        if Path(f).exists():
            for i, line in enumerate(Path(f).read_text(encoding="utf-8").splitlines(), 1):
                if line.strip() and not line.lstrip().startswith("%"):
                    yield f, f"line {i}", line


def is_result_like(tok, pct, txt, start, end):
    ctx = txt[max(0, start - 45): end + 45]
    raw = tok.replace(",", "").lstrip("-−–")
    if re.fullmatch(r"(19|20)\d\d", raw) and not pct:
        return False  # years
    if re.search(r"\d{4}-\d\d-\d\d", txt[max(0, start - 8): end + 8]):
        return False  # dates
    before = txt[max(0, start - 12): start]
    if re.search(r"(first|last|split|train|threads?)\s*$", before, re.I) or re.match(r"\s*(\\,)?\s*(s|sec|seconds|min|ms)\b", txt[end:end + 8]):
        return False
    if re.search(r"(λ|lambda|\\lambda|seed|seeds|epoch|epochs|lag|period|horizon|input_len|K|slide|Fig\.?|Table|Sec\.?|§|v|step|steps|layers?|d_model|patience|batch(_size)?|lr)\s*[=:]?\s*$",
                 before, re.I):
        return False
    if re.search(r"^\s*(seeds?|epochs?|windows?|layers|params|parameters|threads|days|hours|h\b|steps|min|minutes|s\b|sec|×\s*\d)",
                 txt[end:end + 12], re.I) and not pct:
        return False
    if pct or "." in raw:
        return bool(KEY.search(ctx)) or pct
    # bare integers: only with result context and >= 3 digits or a % / ± nearby
    return len(raw) >= 3 and bool(re.search(r"MAE|RMSE|MSE|MW|°C|±|\\pm|error", ctx, re.I)) and float(raw) not in (24, 30, 168, 50, 32, 14)


def tol(tok):
    mant, _, exp = tok.lower().partition("e")
    d = len(mant.split(".")[1]) if "." in mant else 0
    return 0.5 * 10 ** (-d + (int(exp) if exp else 0)) * 1.0001 + 1e-15


def match(x, t, pool):
    hits = [(abs(abs(v) - abs(x)), v, l) for v, l in pool if abs(abs(v) - abs(x)) <= t]
    return sorted(hits)[0] if hits else None


def nearest(x, pool):
    return min(pool, key=lambda vl: abs(abs(vl[0]) - abs(x)) / max(abs(x), 1e-12))


def main():
    rows, cited = [], defaultdict(set)
    for f, loc, txt in units():
        for m in NUM.finditer(txt):
            tok, suf = m.group(1), (m.group(2) or "").strip()
            if suf == "×":
                continue
            if not is_result_like(tok, suf == "%", txt, m.start(), m.end()):
                continue
            raw = tok.replace(",", "").replace("−", "-").replace("–", "-")
            x = float(raw)
            if suf == "%" and x in (70.0, 15.0, 85.0) and not re.search(r"MAE|RMSE|error", txt, re.I):
                continue  # chronological split fractions
            t = tol(raw)
            hit = match(x, t, POOL)
            kind = None
            app_demo = f.endswith("DEMO_SCRIPT.md") and re.search(r"on screen|screen shows|Backtest|backtest|single window|upload", txt)
            if app_demo and not hit:
                kind = "app-demo"
            elif hit:
                kind = "match"
                cited[hit[2]].add((tok + suf, f, loc))
            else:
                nv, nl = nearest(x, POOL)
                rel = abs(abs(nv) - abs(x)) / max(abs(x), 1e-12)
                jh = match(x, t, JSON_POOL)
                if jh:
                    kind = "json-only"
                elif rel <= 0.05:
                    kind = "mismatch"
                else:
                    kind = "no-source"
            ctx = txt[max(0, m.start() - 70): m.end() + 50].replace("|", "/").replace("\n", " ")
            rows.append(dict(file=f, loc=loc, tok=tok + suf, x=x, kind=kind, ctx=ctx,
                             canon=(f"{hit[1]:.6g} ({hit[2]})" if hit else ""),
                             nearest=("" if hit else f"{nv:.6g} ({nl}), rel. diff {100 * rel:.1f}%"),
                             json=("" if hit or kind != "json-only" else f"{jh[1]:.6g} @ {jh[2]}")))
    files = sorted({r["file"] for r in rows})
    L = ["# Consistency scan (read-only)", "",
         "Every result-like number in the slides (text + notes), report/overleaf/report.tex, README.md and",
         "demo_files/DEMO_SCRIPT.md, checked against `canonical.csv` plus the numbers derived from it in",
         "`headline.md` and `noise_rule.md`. Nothing was fixed. Generated by `results_clean/scripts/consistency_scan.py`.", "",
         "**Heuristic** (state it before trusting the counts): a number is 'result-like' if it has a decimal point or %",
         "and a result keyword (MAE, RMSE, MSE, std, ±, %, MW, °C, error, reduction, gap, naive, CORAL, …) within ~45",
         "characters, or is a ≥3-digit integer next to MAE/RMSE/MSE/MW/°C/±/error. Years, dates, λ/seed/epoch/lag/window",
         "settings are skipped. 'Within rounding' = |doc − canonical| ≤ half a unit of the last digit shown in the doc",
         "(sign ignored, so '−30.9%' and '30.9% reduction' match).", "",
         "- **match**: equals a canonical (or headline/noise-rule) value within rounding.",
         "- **mismatch**: no canonical match and no JSON match, but a canonical value lies within 5% → likely a stale,",
         "  hand-rounded or differently computed version of a canonical number. Listed with the nearest canonical value;",
         "  the nearest value is found by magnitude only, so read the context: some mismatches are unrelated quantities.",
         "- **json-only**: no canonical match, but it equals (within rounding) a number in experiments/*.json, or a mean/std",
         "  over the seed files of one group (e.g. source val MSE, MMD, bake-off, λ grid, GRU reference). Not a mismatch,",
         "  but not a canonical target-test number. Coincidental matches are possible for short numbers.",
         "- **no source**: neither canonical nor in any JSON (may be a hand-computed number, a non-result number the",
         "  heuristic caught, or an error).", "",
         "## Counts", "", "| file | numbers checked | match | mismatch | json-only | no source | app-demo |", "|---|---|---|---|---|---|---|"]
    for f in files:
        rr = [r for r in rows if r["file"] == f]
        c = lambda k: sum(r["kind"] == k for r in rr)
        L.append(f"| {f} | {len(rr)} | {c('match')} | {c('mismatch')} | {c('json-only')} | {c('no-source')} | {c('app-demo')} |")
    for kind, title, col in [("app-demo", "DEMO_SCRIPT app values (uploaded demo CSV / backtest, not test-set results; not checkable against canonical)", "nearest"),
                             ("mismatch", "Mismatches", "nearest"), ("no-source", "Numbers with no source", "nearest"),
                             ("json-only", "Numbers sourced from a JSON but not in canonical.csv", "json")]:
        L += ["", f"## {title}", "", f"| file | location | number | context | {'nearest canonical' if col == 'nearest' else 'JSON value @ path'} |",
              "|---|---|---|---|---|"]
        for r in rows:
            if r["kind"] == kind:
                L.append(f"| {r['file']} | {r['loc']} | {r['tok']} | …{r['ctx']}… | {r[col]} |")
    L += ["", "## Same result in two different forms", "",
          "Canonical quantities that the documents cite with more than one written form (different rounding, sign",
          "convention or units), across or within files.", "", "| canonical quantity | forms (file: location) |", "|---|---|"]
    n_forms = 0
    for lab, s in sorted(cited.items()):
        forms = sorted({tok for tok, _, _ in s})
        if len(forms) > 1:
            n_forms += 1
            L.append(f"| {lab} | " + "; ".join(f"`{tok}` ({f.split('/')[-1]}: {loc})" for tok, f, loc in sorted(s)) + " |")
    L.append(f"\n{n_forms} quantities appear in more than one form (automatic count, includes the false positives below).")
    L += ["", "## Manual review notes (checked by reading the source line)", "",
          "Automatic matching is by value only, so a few 'matches' are coincidences. Verified by hand:",
          "- report.tex line 460 `5.17` is the GRU reference climate MAE with source stats (old single-seed GRU run), not the",
          "  energy CORAL paired seed-1 difference it collides with.",
          "- report.tex line 487 / slides 13 `0.00012` is a bake-off source-val MSE std (DLinear, energy), not the finance_vol",
          "  paired-difference std.",
          "- DEMO_SCRIPT.md line 53 `24.0` is an observed temperature (°C) in the app demo, not a paired difference.",
          "- slides v2 slide 14 notes / para 46 `6.8` is the energy CORAL-vs-target-stats MAE difference rounded (−6.84 MW),",
          "  not 'AdaRNN above target-only %'.",
          "- The one remaining **mismatch** (slides v2, slide 14, `19.7% / 23.8%`) is energy source-only (source stats) as a %",
          "  of the *source-stats* naive (4080.7 / 4457.5 MW: 802.88/4080.7 = 19.7%, 1061.17/4457.5 = 23.8%). It is",
          "  arithmetically right but uses a different naive row from canonical.csv, whose % columns use the target-train-stats",
          "  naive (46.7% / 46.8%). Same quantity, two frames → listed here as a two-forms case.",
          "- Real two-form cases worth unifying: energy CORAL−SO-tgt MAE written as −6.84, −6.838, 6.84 and '6.8'; RMSE as",
          "  −13.96, 13.96 and '14.0'; CORAL λ=10 energy RMSE std written as 14.6 and 15; climate SO-tgt MAE as 1.673 and",
          "  1.6731; energy target-only MAE as 524 and 524.44; finance stds as `1.2e-05` (slides) vs `1.2e-5` (report).",
          "- json-only numbers are mostly bake-off source-val MSEs, MMD diagnostics and the GRU reference; they have a",
          "  source but are not target-test results and are outside canonical.csv by design."]
    (OUT / "CONSISTENCY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump({f: {k: sum(r["kind"] == k and r["file"] == f for r in rows) for k in ("match", "mismatch", "json-only", "no-source", "app-demo")}
               for f in files} | {"_forms": n_forms}, open(OUT / "_consistency_counts.json", "w"), indent=1)
    print(open(OUT / "_consistency_counts.json").read())


if __name__ == "__main__":
    main()
