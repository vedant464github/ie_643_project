# UPDATE_LOG

## STEP 0: STATE
- Branch: main (up to date with origin/main), working tree clean.
- Remote: origin https://github.com/vedant464github/ie_643_project.git
- HEAD: 2862097 IE643 project: unsupervised domain adaptation for time-series forecasting

## STEP 1: COPY
- Compared src/, experiments/, results_clean/, architecture/, diagnostics/, DATASETS.md, requirements.txt (and app/ for reference) by content hash.
- Copied 258 files: 257 new (src/adarnn_full/, src/adarnn_fix/, experiments/novelty_adarnn/, novelty_fix/, novelty_fix_cap20/,
  results_clean/ in full incl. audit/, figures/, scripts/, novelty/, novelty2/; architecture/png/, README.md, CHECKS.md,
  IE643_Model_Architectures.pptx (56 KB)) + changed experiments/STATUS.md.
- Unchanged (identical already): requirements.txt, diagnostics/, all other experiments/ folders.
- Kept clean-copy versions (they differ from the original only by deliberate first-clean-copy edits:
  CLAUDE.md->CONSTRAINTS.md wording, raw-data wording, demo uploads built from data/processed):
  src/run_adarnn.py, src/run_coral.py, src/train.py, src/adaptation/adarnn.py, DATASETS.md, app/test_app.py.
- Skipped: all __pycache__/ (src, src/adaptation, src/adarnn_fix, src/adarnn_full, src/data, src/models, src/utils,
  results_clean/scripts, app); logs/ (experiments/logs, experiments/novelty_adarnn/logs, novelty_fix/logs,
  novelty_fix_cap20/logs); architecture/_build/ (not in the requested list); app/test_results.txt (app/ not in scope);
  no zips, no files over 20 MB, no .env/kaggle.json found in the copied trees.
  Not copied by instruction: venv/, data/raw/, demo_files/, walkthrough/, slides/, report/, poster/, STUDY/, submission.zip.

## STEP 2: CONSTRAINTS.md
- Regenerated from the original CLAUDE.md (73 lines, all Amendments through 2026-11-21 (k)); only change: CLAUDE.md -> CONSTRAINTS.md
  (CLAUDE.md contains no self-reference, so the text is otherwise verbatim; CRLF normalised to LF as in the committed file).
- Replaced "CLAUDE.md" with "CONSTRAINTS.md" in 11 src files (src/adarnn_full/{__init__,adarnn,driver,run,summarize,train_timed}.py,
  src/adarnn_fix/{ablation_cap20,driver,power,run,summarize}.py). DATASETS.md had none. README rewritten in STEP 3.
- Left unchanged (outside the named files; result docs/JSONs): "CLAUDE.md" still appears in experiments/STATUS.md,
  experiments/coral/summary.md, experiments/not_run_history.json, experiments/novelty_fix*/**/seed*.json,
  results_clean/*.md, results_clean/audit/*.md, results_clean/novelty*/*.md, results_clean/_tables.json,
  results_clean/scripts/build_tables.py, architecture/CHECKS.md. README says these use the file's original name.

## STEP 3: README
- Kept title, setup, app, reproduction and layout sections; added Datasets, Results (+ headline), How to read the numbers,
  Pre-registration, Novelty assessment (with commands from STATUS.md queue lines and module docstrings).
- Layout fixed: added src/adarnn_full/, src/adarnn_fix/, results_clean/, architecture/, novelty experiment folders.
- Note: results_clean/headline.md has tables but no prose two-sentence headline; the README headline is two sentences built
  only from numbers in headline.md, worded as APPROVED_CLAIMS.md claims 1-3.
- Removed the old status paragraph's blanket "more than seed noise" wording (forbidden in APPROVED_CLAIMS.md); replaced
  with noise-rule labels from APPROVED_CLAIMS.md claims 5-7, 11-12.

## STEP 4: SECURITY AND PATH SCAN
- Scanned all tracked + new files outside data/ (text: 282 files; plus .pt/.png/.csv/.pptx scanned as bytes).
- Personal absolute paths (C:\Users\DELL, E:\academics\sem7\ie643\project_repo, /c/Users/DELL, /e/academics): none found. Nothing replaced.
- Tokens / API keys / passwords / kaggle.json contents (patterns: api_key, secret, password, token, ghp_, sk-, AKIA, kaggle username/key): none.
  Only hits are prose: ".gitignore" comment "# Secrets", and "needs a Kaggle API token" in DATASETS.md and README.md.
- E-mail addresses: none at all in the repo (so none other than the course ones).
- architecture/IE643_Model_Architectures.pptx metadata: creator "IE643 project", lastModifiedBy "Steve Canny" (python-pptx's
  default template author, not personal). No paths or e-mails inside the pptx.
- The GitHub username appears only in this log (remote URL, STEP 0).

## STEP 5: CHECKS (no training)
- `python -m compileall -q src app`: rc 0 (no syntax errors).
- README.md / DATASETS.md references: 97 paths/modules resolve. All `python -m` modules exist (`venv` is the stdlib module).
  The checker's other "missing" hits are not repo paths: data/raw/... (excluded by design, recreated by the download steps),
  file names relative to their folder in context (CONSISTENCY.md, PROVENANCE.md, canonical.csv under results_clean/;
  narrative.md under results_clean/novelty2/; all exist), the Kaggle dataset id, a timezone and raw Kaggle file names.
- .gitignore still excludes venv/, __pycache__/, data/raw/, *.zip, .env, kaggle.json (verified with git check-ignore).
- Size: 38 MB excluding .git. Largest files: data/processed/energy/aep_clean.csv 1.93 MB, comed_clean.csv 1.91 MB,
  results_clean/figures/e_scatter_coral_energy_climate.png 0.82 MB, experiments/novelty_adarnn/grid/energy_adarnn_lam*_seed0.json
  (4 files, ~0.76 MB each), experiments/novelty_adarnn/energy/adarnn/seed2.json 0.53 MB, architecture/png/slide4.png 0.43 MB,
  slide5.png 0.39 MB. Files over 20 MB: none.
- app/test_app.py (OMP_NUM_THREADS=2, original venv's Python 3.12.3, run from the clean copy): it did NOT finish under
  3 minutes and was stopped at the 180 s limit. 29 tests had passed and 0 had failed by then; the remaining tests did not run.
  (The original repo's app/test_results.txt records 30/30 passed for the original test file.)

## STEP 6: COMMIT
- Staged with `git add -A`: 258 added (257 copied files + UPDATE_LOG.md), 3 modified (CONSTRAINTS.md, README.md,
  experiments/STATUS.md), 0 deleted. CRLF warnings are from core.autocrlf=true and do not change content.
- One commit on main; not pushed. Push command: git push origin main
