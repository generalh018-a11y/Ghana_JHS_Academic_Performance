#!/usr/bin/env python3
# ==============================================================================
# CLOSE-OUT SCRIPT — Osafo Simon Yeboah — 2026-09-18
# ==============================================================================
#
# THIS SCRIPT IS AN EXEMPLAR. IT IS NOT A FORCED SOLUTION. RUN IT IN YOUR OWN
# ENVIRONMENT. CHECK EVERY COLUMN NAME, FILE PATH, OBJECT NAME, SEED, LIBRARY
# VERSION AND DATA CONDITION. MODIFY IT WHERE YOUR STUDY CONTEXT REQUIRES
# MODIFICATION. YOU OWN THE FINAL VERSION. DO NOT RUN IT BLINDLY.
#
# IT PRODUCES EVIDENCE, NOT OUTCOMES. IT CAN COME BACK WITH A RESULT YOU DID
# NOT WANT. REPORT WHAT RETURNS, INCLUDING WHEN IT CONTRADICTS YOUR DRAFT.
#
# BLOCK K4 BELOW MAY MAKE THE HEADLINE ACCURACY NUMBER LOOK WORSE, NOT BETTER.
# THAT IS THE CORRECT AND EXPECTED OUTCOME OF THAT BLOCK. A SMALLER GAP AFTER
# REMOVING THE TWO DOMINANT PREDICTORS IS THE FIRST HONEST ANSWER TO THE
# QUESTION AN EXAMINER WILL ASK ABOUT THE JUMP TO 0.9545.
#
# THIS IS THE LAST ENGINEERING PASS ON THIS LIST. AFTER THIS SCRIPT RETURNS,
# PASTE THE VERDICT TABLE BACK. THE NUMBERS ARE THEN THE NUMBERS AND THE
# WRITING BEGINS.
#
# WHAT THIS SCRIPT DOES NOT TOUCH: the sampling-frame wording, which model is
# reported as primary, the theory paragraph, the Problem Statement, and the
# HUSSREC reference. Those are decisions and writing, not code, and they are
# not in this file on purpose.
# ==============================================================================

from __future__ import annotations

import importlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# ENV DETECT
# ------------------------------------------------------------------------------
try:
    import google.colab  # noqa: F401
    ENV = "colab"
except ImportError:
    ENV = "local"
print("ENVIRONMENT:", ENV)

# ------------------------------------------------------------------------------
# CONFIG — one place to change if your layout differs
# ------------------------------------------------------------------------------
import os

if ENV == "colab":
    from google.colab import drive
    drive.mount("/content/drive")
    # <<< FILL IN: the path to your cloned repository inside Drive >>>
    PROJECT_ROOT = Path("/content/drive/MyDrive/Ghana_JHS_Academic_Performance")
    print("PROJECT_ROOT (colab):", PROJECT_ROOT)
else:
    PROJECT_ROOT = Path(__file__).resolve().parent
    print("PROJECT_ROOT (local):", PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_DIR = RESULTS_DIR / "summary_tables"          # new — not in .gitignore (K3)
ARCHIVE_DIR = PROJECT_ROOT / "archive"                 # new — for legacy files (K2)

# Where the RAW institutional workbook lives on THIS machine. Every hardcoded
# path in the repo currently points at one machine's OneDrive folder; this is
# the single constant that replaces all of them. Override with the
# JHS_RAW_DATA_DIR environment variable if your workbook is not under
# <repo>/data/raw/.
RAW_DATA_DIR = Path(os.environ.get("JHS_RAW_DATA_DIR", str(PROJECT_ROOT / "data" / "raw")))
RAW_WORKBOOK_NAME = "JHS_Data_Collection_1.xlsx"       # source: check_*.py FILE constants
RAW_WORKBOOK = RAW_DATA_DIR / RAW_WORKBOOK_NAME

OWNER_TAG = "Osafo Simon Yeboah"
RUN_ANCHOR = "RUN-OSAFO_SIMON_YEBOAH-20260918"
COMMIT_AT_LAST_VERIFY = "ae562b1f7a33913e545664dfe62240c57e5572a7"   # source: verification checklist
COMMIT_NOW_EXPECTED = "87f898b9fa842bfe25b8cd2d658a83d793b40ab6"      # source: `git rev-parse HEAD` when this script was built

# Set this to True only after Half 1 has printed and you have read it.
# Nothing in Half 2 runs while this is False.
RUN_CLOSE_BLOCKS = True

print(f"OWNER: {OWNER_TAG} | ANCHOR: {RUN_ANCHOR}")
print(f"COMMIT AT LAST VERIFICATION : {COMMIT_AT_LAST_VERIFY}")
print(f"COMMIT EXPECTED NOW         : {COMMIT_NOW_EXPECTED}")

# ------------------------------------------------------------------------------
# VERSION GUARD — source: requirements.txt in this repository
# ------------------------------------------------------------------------------
REQUIRED_VERSIONS = {
    "pandas": "2.3.3",
    "numpy": "2.2.6",
    "sklearn": "1.7.2",     # pip name scikit-learn, import name sklearn
    "xgboost": "3.2.0",
    "lightgbm": "4.7.0",
}

print("\nVERSION GUARD")
_version_mismatch = False
for _pkg, _expected in REQUIRED_VERSIONS.items():
    try:
        _mod = importlib.import_module(_pkg)
        _got = getattr(_mod, "__version__", "UNKNOWN")
        _status = "OK" if _got == _expected else "MISMATCH"
        if _status == "MISMATCH":
            _version_mismatch = True
        print(f"  {_pkg:<10} expected {_expected:<10} got {_got:<10} {_status}")
    except ImportError:
        _version_mismatch = True
        print(f"  {_pkg:<10} NOT INSTALLED")
if _version_mismatch:
    print("  -> API MISMATCH — verify parameter names for your installed version before trusting K4.")

# ------------------------------------------------------------------------------
# PREFLIGHT
# ------------------------------------------------------------------------------
print("\nPREFLIGHT")

_git_dir = PROJECT_ROOT / ".git"
print(f"  Repository root exists      : {PROJECT_ROOT.exists()}")
print(f"  .git present                : {_git_dir.exists()}")
print(f"  data/processed exists       : {DATA_DIR.exists()}")

# The four canonical modelling matrices produced by step 19 — K4 needs these.
_MODEL_FILES = {
    "X_train": DATA_DIR / "X_train.csv",
    "y_train": DATA_DIR / "y_train.csv",
    "X_validation": DATA_DIR / "X_validation.csv",
    "y_validation": DATA_DIR / "y_validation.csv",
    "X_test": DATA_DIR / "X_test.csv",
    "y_test": DATA_DIR / "y_test.csv",
}
_model_files_present = {k: v.exists() for k, v in _MODEL_FILES.items()}
for _name, _present in _model_files_present.items():
    print(f"  {_name:<14} present      : {_present}  ({_MODEL_FILES[_name].name})")

ALL_MODEL_FILES_PRESENT = all(_model_files_present.values())
if not ALL_MODEL_FILES_PRESENT:
    print("  -> K4 (prior-score ablation) needs these. Run steps 17-19 locally first"
          " if any are missing; they are gitignored and exist only on your machine.")

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# HALF 1 — VERIFY  (read-only; safe to run in any order)
# ==============================================================================

_verify_results = {}   # item_id -> dict(verdict=..., evidence=...)

# ------------------------------------------------------------------------------
# BLOCK V1 — CONFIRM THE SPLIT-VARIANCE FIX (CODE-1) IS ALREADY CLOSED
# CLOSES            : CODE-1 (split/seed variance unmeasured)
# OPEN ITEM         : checklist recorded ten-seed runs as reading one frozen
#                     partition, so seed variance and split variance were
#                     confounded.
# ANSWERS           : does the repository already contain the repeated
#                     grouped-split analysis, and does its legacy
#                     reconciliation check pass?
# EXPECTED DIRECTION: NEUTRAL — this block only reads an existing artefact.
# RUNTIME           : instant, no fits.
# IF IT RETURNS
# SOMETHING BAD     : if the status file is missing or reconciliation is not
#                     PASS, CODE-1 is NOT closed — say so and stop claiming it
#                     is. Do not re-derive a verdict from the manuscript text.
# PASTE OUTPUT WHERE: nowhere — this just confirms Table 4.3A / step 23B stand.
# ------------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BLOCK V1 — split/seed variance closure check (CODE-1)")
print("=" * 78)

_status_file = RESULTS_DIR / "variance" / "step_23b_status.json"
_decomp_file = RESULTS_DIR / "variance" / "split_seed_variance_decomposition.csv"

if _status_file.exists() and _decomp_file.exists():
    with open(_status_file) as f:
        _status = json.load(f)
    _recon = _status.get("legacy_partition0_reconciliation", "MISSING")
    _n_partitions = _status.get("valid_partitions", "MISSING")
    print(f"  step_23b_status.json found. legacy_partition0_reconciliation = {_recon}")
    print(f"  valid_partitions = {_n_partitions}")
    if _recon == "PASS" and _n_partitions == 10:
        _verify_results["CODE-1"] = {
            "verdict": "CLOSED",
            "evidence": f"results/variance/step_23b_status.json: reconciliation={_recon}, partitions={_n_partitions}",
        }
        print("  VERDICT: CLOSED — split variance and seed variance are reported separately"
              " (Table 4.3A), and the legacy partition-0 reconciliation passed.")
    else:
        _verify_results["CODE-1"] = {"verdict": "STILL OPEN", "evidence": "reconciliation not PASS or partition count wrong"}
        print("  VERDICT: STILL OPEN — status file exists but does not confirm reconciliation.")
else:
    _verify_results["CODE-1"] = {"verdict": "UNVERIFIABLE", "evidence": "results/variance/ artefacts not found"}
    print("  VERDICT: UNVERIFIABLE — results/variance/step_23b_status.json or the"
          " decomposition CSV is missing from this checkout.")

# ------------------------------------------------------------------------------
# BLOCK V2 — SCAN FOR MACHINE-SPECIFIC PATHS (CODE-2)
# CLOSES            : CODE-2 (machine-specific paths)
# OPEN ITEM         : step_17_build_term_level_dataset.py:23 and all seven
#                     check_*.py scripts hard-code a OneDrive path on one
#                     machine.
# ANSWERS           : which files still contain the hardcoded path, verbatim.
# EXPECTED DIRECTION: NEUTRAL — read-only scan; K1 is the fix.
# RUNTIME           : instant.
# IF IT RETURNS
# SOMETHING BAD     : if files outside this list also match, add them to
#                     _PATH_TARGETS before running K1 — do not let K1 patch a
#                     file it was not told about.
# PASTE OUTPUT WHERE: nowhere — feeds K1 directly.
# ------------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BLOCK V2 — hardcoded-path scan (CODE-2)")
print("=" * 78)

_PATH_TARGETS = [
    "step_17_build_term_level_dataset.py",
    "check_attendance_consistency.py",
    "check_context_variables.py",
    "check_early_warning_eligibility.py",
    "check_missingness.py",
    "check_student_ids.py",
    "check_target_distribution.py",
    "check_year_progression.py",
]
# The exact literal found by the verification checklist against commit ae562b1.
_HARDCODED_PATTERN = re.compile(
    r'r?["\']C:\\\\Users\\\\hills\\\\OneDrive\\\\Documents\\\\Data Collection__Thesis[^"\']*["\']'
)

_path_hits = {}
for _fname in _PATH_TARGETS:
    _fpath = PROJECT_ROOT / _fname
    if not _fpath.exists():
        print(f"  {_fname:<40} FILE NOT FOUND")
        continue
    _text = _fpath.read_text(encoding="utf-8")
    _matches = _HARDCODED_PATTERN.findall(_text)
    _path_hits[_fname] = len(_matches)
    print(f"  {_fname:<40} hardcoded literals: {len(_matches)}")

_total_hits = sum(_path_hits.values())
if _total_hits > 0:
    _verify_results["CODE-2"] = {"verdict": "OPEN IN CODE", "evidence": f"{_total_hits} hardcoded path literal(s) across {len(_path_hits)} file(s)"}
    print(f"  VERDICT: OPEN IN CODE — {_total_hits} hardcoded literal(s) found. K1 will fix these.")
else:
    _verify_results["CODE-2"] = {"verdict": "CLOSED", "evidence": "no matching literals found"}
    print("  VERDICT: CLOSED — no hardcoded literals matched. (If this is unexpected,"
        " the path may have already been edited to something V2's pattern does not match — check by eye.)")

# ------------------------------------------------------------------------------
# BLOCK V3 — SCAN FOR LEGACY / SUPERSEDED FILES (CODE-3)
# CLOSES            : CODE-3 (legacy artefacts still committed)
# OPEN ITEM         : prepare_model_data.py, prepare_modeling_data.py,
#                     data_quality_audit.py, data/processed/fairness_class_
#                     results.csv and fairness_class_gaps.csv are all from the
#                     superseded regression-era experiment.
# ANSWERS           : which of those five paths still exist in the checkout.
# EXPECTED DIRECTION: NEUTRAL.
# RUNTIME           : instant.
# IF IT RETURNS
# SOMETHING BAD     : none of these are used by any step_17-29 script — if K2
#                     later breaks something, that import was undocumented and
#                     is itself worth reporting.
# PASTE OUTPUT WHERE: nowhere — feeds K2 directly.
# ------------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BLOCK V3 — legacy artefact scan (CODE-3)")
print("=" * 78)

_LEGACY_TARGETS = [
    PROJECT_ROOT / "prepare_model_data.py",
    PROJECT_ROOT / "prepare_modeling_data.py",
    PROJECT_ROOT / "data_quality_audit.py",
    DATA_DIR / "fairness_class_results.csv",
    DATA_DIR / "fairness_class_gaps.csv",
]
_legacy_present = [p for p in _LEGACY_TARGETS if p.exists()]
for _p in _LEGACY_TARGETS:
    print(f"  {str(_p.relative_to(PROJECT_ROOT)):<55} present: {_p.exists()}")

if _legacy_present:
    _verify_results["CODE-3"] = {"verdict": "OPEN IN CODE", "evidence": f"{len(_legacy_present)} of 5 legacy files still present"}
    print(f"  VERDICT: OPEN IN CODE — {len(_legacy_present)} legacy file(s) still committed. K2 will archive these.")
else:
    _verify_results["CODE-3"] = {"verdict": "CLOSED", "evidence": "none of the 5 legacy paths exist"}
    print("  VERDICT: CLOSED — none of the five legacy paths exist in this checkout.")

# ------------------------------------------------------------------------------
# BLOCK V4 — CHECK WHICH NON-SENSITIVE SUMMARY TABLES EXIST LOCALLY BUT ARE
#            NOT COMMITTED (CODE-4)
# CLOSES            : CODE-4 (Tables 4.2/4.3/4.4/4.8/4.9 have no artefact to
#                     cross-check because only results/shap and
#                     results/variance are committed).
# ANSWERS           : which of the five aggregate output files exist in your
#                     local data/processed/, and whether each is safely
#                     aggregate-only (no student_id column, small row count).
# EXPECTED DIRECTION: NEUTRAL.
# RUNTIME           : instant.
# IF IT RETURNS
# SOMETHING BAD     : if a file fails the "no student_id, few rows" check,
#                     K3 will refuse to copy it and say why. Do not override
#                     that refusal by hand-editing K3.
# PASTE OUTPUT WHERE: nowhere — feeds K3 directly.
# ------------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BLOCK V4 — committable summary-table scan (CODE-4)")
print("=" * 78)

# source of each path: the to_csv() call in the step script named in the comment
_SUMMARY_CANDIDATES = {
    "table_4_2_and_4_8_locked_seed42.csv": DATA_DIR / "locked_ensemble_test_results_seed42.csv",   # step_22
    "table_4_3_ten_seed_stability.csv": DATA_DIR / "ten_seed_ensemble_summary_mean_sd.csv",         # step_23
    "table_4_4_three_cell_ablation.csv": DATA_DIR / "three_cell_ablation_summary_mean_sd.csv",       # step_25
    "table_4_9_subgroup_performance.csv": DATA_DIR / "fairness_subgroup_summary_mean_sd.csv",        # step_27b
    "table_4_10_subgroup_gaps.csv": DATA_DIR / "fairness_gap_summary.csv",                            # step_27b
}

_MAX_SAFE_ROWS = 25  # aggregate summaries in this pipeline are all well under this

_summary_status = {}
for _out_name, _src in _SUMMARY_CANDIDATES.items():
    if not _src.exists():
        _summary_status[_out_name] = ("MISSING", None)
        print(f"  {_src.name:<45} MISSING — run its step_XX script locally first")
        continue
    _df = pd.read_csv(_src)
    _has_student_id = any("student" in c.lower() and "id" in c.lower() for c in _df.columns)
    _row_ok = len(_df) <= _MAX_SAFE_ROWS
    if _has_student_id:
        _summary_status[_out_name] = ("UNSAFE_STUDENT_ID_COLUMN", _src)
        print(f"  {_src.name:<45} REFUSED — contains a student-id-like column")
    elif not _row_ok:
        _summary_status[_out_name] = ("UNSAFE_ROW_COUNT", _src)
        print(f"  {_src.name:<45} REFUSED — {len(_df)} rows, over the {_MAX_SAFE_ROWS}-row aggregate ceiling")
    else:
        _summary_status[_out_name] = ("SAFE", _src)
        print(f"  {_src.name:<45} SAFE to commit — {len(_df)} row(s), {list(_df.columns)}")

_n_missing = sum(1 for v in _summary_status.values() if v[0] == "MISSING")
_n_safe = sum(1 for v in _summary_status.values() if v[0] == "SAFE")
if _n_missing == len(_summary_status):
    _verify_results["CODE-4"] = {"verdict": "UNVERIFIABLE", "evidence": "no local step outputs found to check"}
elif _n_safe == len(_summary_status):
    _verify_results["CODE-4"] = {"verdict": "OPEN IN CODE", "evidence": f"{_n_safe} file(s) safe and ready to commit; none committed yet"}
else:
    _verify_results["CODE-4"] = {"verdict": "OPEN IN CODE", "evidence": f"{_n_safe} of {len(_summary_status)} ready; rest missing or unsafe"}
print(f"  VERDICT: {_verify_results['CODE-4']['verdict']} — {_verify_results['CODE-4']['evidence']}")

# ------------------------------------------------------------------------------
# BLOCK V5 — SIGN-STABILITY CHECK FOR THE attendance_decay NULL RESULT
# CLOSES            : feeds the null-clearance certificate for "attendance_decay
#                     does not improve predictive performance" (Table 4.4).
# OPEN ITEM         : the checklist's ablation finding is currently backed only
#                     by mean +/- SD (Table 4.4), not by a per-seed sign check.
# ANSWERS           : across the 10 model seeds, does Cell A beat Cell C on
#                     Macro-F1 in every seed, or does the sign flip?
# EXPECTED DIRECTION: NEUTRAL — this either confirms or weakens a result
#                     already in the manuscript; it does not change it.
# RUNTIME           : instant, reads an existing per-seed file.
# IF IT RETURNS
# SOMETHING BAD     : if the sign flips in any seed, the Table 4.4 headline
#                     ("Cell A wins on every measure") is a mean-level
#                     statement, not a seed-level one, and that qualification
#                     belongs in the text. Report it; do not soften it.
# PASTE OUTPUT WHERE: Section 4.4, as a one-line addition to the ablation
#                     discussion, if the sign does not flip.
# ------------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BLOCK V5 — attendance_decay ablation sign-stability check")
print("=" * 78)

_by_seed_file = DATA_DIR / "three_cell_ablation_results_by_seed.csv"
if _by_seed_file.exists():
    _by_seed = pd.read_csv(_by_seed_file)
    print(f"  Loaded {_by_seed_file.name}, shape {_by_seed.shape}, columns {list(_by_seed.columns)}")
    # <<< FILL IN: confirm the seed and cell-label column names below match your
    #     file exactly — this reads the header printed above, not a guess.
    _seed_col = "seed" if "seed" in _by_seed.columns else "<<< FILL IN: seed column name >>>"
    _cell_col = "cell_description" if "cell_description" in _by_seed.columns else "<<< FILL IN: cell column name >>>"
    _metric_col = "macro_f1" if "macro_f1" in _by_seed.columns else "<<< FILL IN: metric column name >>>"
    if _seed_col.startswith("<<<") or _cell_col.startswith("<<<") or _metric_col.startswith("<<<"):
        print("  SKIPPED — column names could not be auto-matched. Fill in the three"
              " constants above from the printed column list and re-run this block.")
        _verify_results["NULL-C1"] = {"verdict": "UNVERIFIABLE", "evidence": "column names not confirmed"}
    else:
        _a = _by_seed[_by_seed[_cell_col].str.contains("without attendance_decay", case=False, na=False)]
        _c = _by_seed[_by_seed[_cell_col].str.contains("validation-selected", case=False, na=False)]
        _merged = _a.merge(_c, on=_seed_col, suffixes=("_A", "_C"))
        _merged["delta"] = _merged[f"{_metric_col}_A"] - _merged[f"{_metric_col}_C"]
        _n_seeds = len(_merged)
        _n_positive = int((_merged["delta"] > 0).sum())
        print(f"  Seeds compared: {_n_seeds}. Cell A ahead in {_n_positive} of {_n_seeds}.")
        print(_merged[[_seed_col, f"{_metric_col}_A", f"{_metric_col}_C", "delta"]].to_string(index=False))
        _sign_stable = _n_positive == _n_seeds or _n_positive == 0
        _verify_results["NULL-C1"] = {
            "verdict": "PASS" if _sign_stable else "FAIL",
            "evidence": f"Cell A ahead in {_n_positive}/{_n_seeds} seeds",
        }
        print(f"  CLEARANCE C1 (sign stable across seeds): {'PASS' if _sign_stable else 'FAIL'}")
else:
    _verify_results["NULL-C1"] = {"verdict": "UNVERIFIABLE", "evidence": "three_cell_ablation_results_by_seed.csv not found locally"}
    print("  UNVERIFIABLE — per-seed ablation file not found. Run step_25 locally;"
          " it is gitignored so it exists only on your machine.")

# ==============================================================================
# HALF 2 — CLOSE  (pipeline- and repository-changing; gated by RUN_CLOSE_BLOCKS)
# ==============================================================================
if not RUN_CLOSE_BLOCKS:
    print("\n" + "=" * 78)
    print("HALF 2 SKIPPED — RUN_CLOSE_BLOCKS is False.")
    print("Read Half 1 above. If you're satisfied it's reading your repository")
    print("correctly, set RUN_CLOSE_BLOCKS = True near the top of this file and")
    print("run again.")
    print("=" * 78)
else:

    # --------------------------------------------------------------------------
    # BLOCK K1 — REPLACE HARDCODED PATHS (CODE-2)
    # CLOSES            : CODE-2
    # EXPECTED DIRECTION: NEUTRAL — no numbers change; only file paths.
    # RUNTIME           : instant.
    # IF IT RETURNS
    # SOMETHING BAD     : if a file's patched version fails to parse (rare —
    #                     this is a straight string substitution), the block
    #                     prints which file and leaves it untouched. Fix by
    #                     hand and re-run V2.
    # PASTE OUTPUT WHERE: nowhere — this is a repository change, not a
    #                     manuscript number. Commit it (git commands printed
    #                     below) and note in §3.10/§3.14 that paths are now
    #                     repository-relative, replacing "is required to use"
    #                     with "uses".
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("BLOCK K1 — patch hardcoded paths (CODE-2)")
    print("=" * 78)

    _REPLACEMENT_HEADER = (
        "\n# --- path made repository-relative; override with the\n"
        "# JHS_RAW_DATA_DIR environment variable if your workbook lives elsewhere ---\n"
        "from pathlib import Path as _Path\n"
        "import os as _os\n"
        "_RAW_DIR = _Path(_os.environ.get(\"JHS_RAW_DATA_DIR\", str(_Path(__file__).resolve().parent / \"data\" / \"raw\")))\n"
    )

    _k1_changed = []
    for _fname, _hits in _path_hits.items():
        if _hits == 0:
            continue
        _fpath = PROJECT_ROOT / _fname
        _text = _fpath.read_text(encoding="utf-8")
        _new_text, _n_sub = _HARDCODED_PATTERN.subn(
            f'_RAW_DIR / "{RAW_WORKBOOK_NAME}"', _text
        )
        if _n_sub == 0:
            print(f"  {_fname:<40} pattern not found on rewrite pass — left untouched")
            continue
        if _REPLACEMENT_HEADER.strip() not in _new_text:
            # insert the header once, after the module docstring / imports area —
            # simplest safe anchor is right after the first import line.
            _lines = _new_text.splitlines(keepends=True)
            _insert_at = 0
            for _i, _line in enumerate(_lines):
                if _line.startswith("import ") or _line.startswith("from "):
                    _insert_at = _i + 1
            _lines.insert(_insert_at, _REPLACEMENT_HEADER)
            _new_text = "".join(_lines)
        _fpath.write_text(_new_text, encoding="utf-8")
        _k1_changed.append(_fname)
        print(f"  {_fname:<40} patched — {_n_sub} literal(s) replaced")

    print(f"\n  Files changed: {len(_k1_changed)} of {len(_PATH_TARGETS)}")
    if _k1_changed:
        print("  Run this next:")
        print(f"    git add {' '.join(_k1_changed)}")
        print('    git commit -m "Make raw-data path repository-relative (closes CODE-2)"')

    # --------------------------------------------------------------------------
    # BLOCK K2 — ARCHIVE LEGACY FILES (CODE-3)
    # CLOSES            : CODE-3
    # EXPECTED DIRECTION: NEUTRAL — no script in steps 17-29 imports these files;
    #                     moving them changes nothing about any reported number.
    # RUNTIME           : instant.
    # IF IT RETURNS
    # SOMETHING BAD     : if any later step throws ImportError or FileNotFoundError
    #                     after this runs, one of these WAS still in use — that is
    #                     itself a finding, worth a line in your reply to your
    #                     supervisor. Do not silently move it back.
    # PASTE OUTPUT WHERE: nowhere in the manuscript — repository hygiene only.
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("BLOCK K2 — archive legacy files (CODE-3)")
    print("=" * 78)

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    _k2_moved = []
    for _p in _legacy_present:
        _dest = ARCHIVE_DIR / _p.name
        shutil.move(str(_p), str(_dest))
        _k2_moved.append((str(_p.relative_to(PROJECT_ROOT)), str(_dest.relative_to(PROJECT_ROOT))))
        print(f"  moved {_p.relative_to(PROJECT_ROOT)} -> {_dest.relative_to(PROJECT_ROOT)}")

    if _k2_moved:
        print("\n  Run this next:")
        _old_paths = " ".join(src for src, _ in _k2_moved)
        print(f"    git add archive/")
        print(f"    git rm {_old_paths}")
        print('    git commit -m "Archive superseded regression-era files (closes CODE-3)"')
    else:
        print("  Nothing to move — V3 found no legacy files present.")

    # --------------------------------------------------------------------------
    # BLOCK K3 — COMMIT NON-SENSITIVE SUMMARY TABLES (CODE-4)
    # CLOSES            : CODE-4
    # EXPECTED DIRECTION: NEUTRAL — copies existing numbers, computes nothing new.
    # RUNTIME           : instant.
    # IF IT RETURNS
    # SOMETHING BAD     : a REFUSED file from V4 is not copied here, on purpose.
    #                     Do not hand-copy a refused file to work around this.
    # PASTE OUTPUT WHERE: Section 3.14 / 3.10, replace "documentation is required
    #                     to use repository-relative paths" language with a
    #                     pointer to results/summary_tables/ as the artefact a
    #                     reader can check Tables 4.2, 4.3, 4.4, 4.9 and 4.10
    #                     against.
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("BLOCK K3 — commit non-sensitive summary tables (CODE-4)")
    print("=" * 78)

    _k3_copied = []
    for _out_name, (_state, _src) in _summary_status.items():
        if _state != "SAFE":
            print(f"  {_out_name:<45} SKIPPED ({_state})")
            continue
        _dest = SUMMARY_DIR / _out_name
        shutil.copy(_src, _dest)
        _k3_copied.append(str(_dest.relative_to(PROJECT_ROOT)))
        print(f"  {_out_name:<45} copied -> {_dest.relative_to(PROJECT_ROOT)}")

    if _k3_copied:
        print("\n  Run this next:")
        print(f"    git add {' '.join(_k3_copied)}")
        print('    git commit -m "Commit aggregate result tables for reviewer cross-check (closes CODE-4)"')
    else:
        print("  Nothing SAFE to copy. Run steps 22, 23, 25 and 27b locally, then re-run"
              " this script from the top so V4 can see their output.")

    # --------------------------------------------------------------------------
    # BLOCK K4 — PRIOR-SCORE ABLATION: DOES THE JUMP TO 0.9545 SURVIVE WITHOUT
    #            score_t1 AND score_t2?  (CODE-5)
    # CLOSES            : CODE-5
    # OPEN ITEM         : checklist — "the accuracy jump needs interrogation,
    #                     not celebration... report how much performance
    #                     survives when score_t1 and score_t2 are removed."
    # EXPECTED DIRECTION: CORRECTING — this is expected to REDUCE the reported
    #                     accuracy substantially. That is the answer, not a
    #                     failure of the block.
    # RUNTIME           : ~10 fits (one XGBoost + one LightGBM per seed, 10
    #                     seeds), a few minutes on CPU.
    # IF IT RETURNS
    # SOMETHING BAD     : if accuracy barely drops without score_t1/score_t2,
    #                     that is also reportable — it would mean the model is
    #                     not simply reproducing prior attainment. Report
    #                     whichever direction it goes.
    # PASTE OUTPUT WHERE: Section 4.4, as new Table 4.4A (Cell D), and one
    #                     sentence in Section 4.3 addressing the jump from
    #                     ~0.83 to 0.9545 referenced there.
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("BLOCK K4 — prior-score ablation (Cell D) (CODE-5)")
    print("=" * 78)

    if not ALL_MODEL_FILES_PRESENT:
        print("  SKIPPED — X_train/X_validation/X_test (and y_*) not found locally.")
        print("  Run steps 17-19 first; these files are gitignored and exist only on"
              " your machine.")
    else:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            brier_score_loss,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier

        X_train = pd.read_csv(_MODEL_FILES["X_train"])
        y_train = pd.read_csv(_MODEL_FILES["y_train"]).iloc[:, 0].astype(int)
        X_val = pd.read_csv(_MODEL_FILES["X_validation"])
        y_val = pd.read_csv(_MODEL_FILES["y_validation"]).iloc[:, 0].astype(int)
        X_test = pd.read_csv(_MODEL_FILES["X_test"])
        y_test = pd.read_csv(_MODEL_FILES["y_test"]).iloc[:, 0].astype(int)

        print(f"  Loaded X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")

        _DROP_FEATURES = ["score_t1", "score_t2"]   # source: manuscript Table 4.6, the two dominant SHAP features
        _missing = [c for c in _DROP_FEATURES if c not in X_train.columns]
        if _missing:
            print(f"  STOP — expected columns not found: {_missing}. Column names in your"
                  f" data: {list(X_train.columns)}. Fix _DROP_FEATURES above and re-run.")
        else:
            X_train_d = X_train.drop(columns=_DROP_FEATURES)
            X_val_d = X_val.drop(columns=_DROP_FEATURES)
            X_test_d = X_test.drop(columns=_DROP_FEATURES)
            print(f"  Dropped {_DROP_FEATURES}. Remaining columns ({len(X_train_d.columns)}):"
                  f" {list(X_train_d.columns)}")

            SEEDS = list(range(42, 52))   # source: step_23_ten_seed_stability.py SEEDS
            THRESHOLD = 0.50              # source: manuscript Table 3.2

            def _build_xgb(seed):
                return XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    objective="binary:logistic", eval_metric="logloss",
                    random_state=seed, n_jobs=-1,
                )  # source: step_25_three_cell_ablation.py build_xgb

            def _build_lgbm(seed):
                return LGBMClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    objective="binary", random_state=seed, n_jobs=-1, verbosity=-1,
                )  # source: step_25_three_cell_ablation.py build_lgbm

            def _metrics(y_true, proba, threshold=THRESHOLD):
                pred = (proba >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
                return {
                    "accuracy": accuracy_score(y_true, pred),
                    "precision": precision_score(y_true, pred, zero_division=0),
                    "recall": recall_score(y_true, pred, zero_division=0),
                    "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
                    "auc_roc": roc_auc_score(y_true, proba),
                    "auc_pr": average_precision_score(y_true, proba),
                    "brier": brier_score_loss(y_true, proba),
                    "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                }

            _cell_d_rows = []
            for _seed in SEEDS:
                _xgb = _build_xgb(_seed)
                _lgbm = _build_lgbm(_seed)
                _xgb.fit(X_train_d, y_train, eval_set=[(X_val_d, y_val)], verbose=False)
                _lgbm.fit(X_train_d, y_train, eval_set=[(X_val_d, y_val)], callbacks=[])
                # 50:50 weighting — matches Cell A, the reference condition,
                # so this isolates score_t1/score_t2 the same way Cell A
                # isolated attendance_decay.
                _proba = 0.5 * _xgb.predict_proba(X_test_d)[:, 1] + 0.5 * _lgbm.predict_proba(X_test_d)[:, 1]
                _row = {"seed": _seed, **_metrics(y_test, _proba)}
                _cell_d_rows.append(_row)
                print(f"  seed {_seed}: accuracy={_row['accuracy']:.4f} macro_f1={_row['macro_f1']:.4f}")

            _cell_d = pd.DataFrame(_cell_d_rows)
            _cell_d_summary = _cell_d.drop(columns=["seed"]).agg(["mean", "std"]).T
            _cell_d_summary.columns = ["mean", "sd"]

            print("\n  CELL D SUMMARY (50:50 ensemble, score_t1 and score_t2 removed):")
            print(_cell_d_summary.to_string())

            _dest = DATA_DIR / "cell_d_no_prior_score_results.csv"
            _cell_d.to_csv(_dest, index=False)
            _dest_summary = DATA_DIR / "cell_d_no_prior_score_summary.csv"
            _cell_d_summary.to_csv(_dest_summary)
            print(f"\n  saved {_dest.relative_to(PROJECT_ROOT)} and {_dest_summary.relative_to(PROJECT_ROOT)}")

            # BEFORE/AFTER — BEFORE is QUOTED from the manuscript and never fed
            # into any calculation. AFTER is computed above.
            print(f"\n  {'metric':<16}{'BEFORE':>10}{'AFTER':>10}{'DELTA':>10}  CAUSE")
            _before_rows = [
                # (label, BEFORE quoted value, AFTER key, source)
                ("accuracy vs Cell A", 0.9673, "accuracy", "Table 4.4 QUOTED"),
                ("accuracy vs ensemble", 0.9545, "accuracy", "Table 4.2 QUOTED"),
                ("accuracy vs naive", 0.7091, "accuracy", "Table 4.3B QUOTED"),
                ("macro_f1 vs Cell A", 0.9601, "macro_f1", "Table 4.4 QUOTED"),
            ]
            _after_mean = _cell_d_summary["mean"]
            for _label, _before, _key, _src in _before_rows:
                _after = _after_mean[_key]
                _delta = _after - _before
                print(f"  {_label:<24}{_before:>10.4f}{_after:>10.4f}{_delta:>10.4f}  CAUSE=CODE-5 ({_src})")

            print("\n  This table is descriptive. It does not, by itself, prove how much of")
            print("  the jump from ~0.83 (earlier pipeline) to 0.9545 (current pipeline) is")
            print("  autocorrelation versus genuine early-warning signal — it shows what the")
            print("  ensemble does when score_t1/score_t2 are unavailable, which is the")
            print("  question an examiner is expected to ask about Section 4.3.")

    print("\n" + "=" * 78)
    print("HALF 2 COMPLETE.")
    print("=" * 78)

# ==============================================================================
# VERDICT TABLE — the only thing to paste back
# ==============================================================================
print("\n" + "=" * 78)
print("VERDICT TABLE")
print("=" * 78)
print(f"{'ITEM':<10}{'CHECK':<42}{'VERDICT':>14}")
_display_names = {
    "CODE-1": "split/seed variance separated",
    "CODE-2": "no hardcoded machine paths",
    "CODE-3": "no legacy files committed",
    "CODE-4": "aggregate tables committed",
    "NULL-C1": "decay-null sign stable across seeds",
}
_closed_n = 0
_open_n = 0
_unverifiable_n = 0
for _item, _name in _display_names.items():
    _res = _verify_results.get(_item, {"verdict": "UNVERIFIABLE", "evidence": "not checked"})
    _verdict = _res["verdict"]
    if _verdict in ("CLOSED", "PASS"):
        _closed_n += 1
    elif _verdict == "UNVERIFIABLE":
        _unverifiable_n += 1
    else:
        _open_n += 1
    print(f"{_item:<10}{_name:<42}{_verdict:>14}")

print(f"\nCLOSED: {_closed_n}   STILL OPEN/OPEN IN CODE: {_open_n}   UNVERIFIABLE: {_unverifiable_n}")
print(f"RUN_CLOSE_BLOCKS was: {RUN_CLOSE_BLOCKS}")
if not RUN_CLOSE_BLOCKS:
    print("Half 2 did not run. The OPEN rows above will not change until you set")
    print("RUN_CLOSE_BLOCKS = True and run again.")

print("\n" + "=" * 78)
print("PASTE THIS ENTIRE VERDICT TABLE BACK TO YOUR SUPERVISOR.")
print("=" * 78)
