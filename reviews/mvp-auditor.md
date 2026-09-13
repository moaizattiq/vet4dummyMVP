# MVP audit: run from a clean copy

**Verdict: RUNS WITH ISSUES.** Every documented `make` target ran from a clean copy of the working tree. Training is bit-for-bit reproducible. The API and the static page answer correctly. The issues are around setup and hygiene: no `.gitignore` while the README tells people to put a database password in `.env` at the repo root, no dependency manifest or version pins, an undocumented native dependency (libomp) for LightGBM on macOS, hard cwd-relative paths that make "run from repo root" mandatory, and a holidays table that ends 2025-12-31, which means every forecast the demo can show has the holiday features switched off.

Audit date: 2026-09-13. Auditor: mvp-auditor (clean-copy run). Nothing in the original repo was modified except this file.

## Setup for the audit

- Original repo: `/Users/moaizattiq/vets4war-dummy` (git init'd, no commits, no `.env`, no `.gitignore`).
- Clean copy: `/private/tmp/claude-501/-Users-moaizattiq/b9883508-bb1c-48a6-9c88-6f94889e7936/scratchpad/mvp-clean`, made with `rsync -a --exclude .git --exclude __pycache__ --exclude .pytest_cache`.
- Interpreter: `/Users/moaizattiq/v4w-staffing/.venv/bin/python` (Python 3.12.13), passed as `PYTHON=` to make. The repo itself has no venv, `requirements.txt`, or `pyproject.toml`.
- Database URL: loaded into the shell from `/Users/moaizattiq/v4w-staffing/.env` only where stated below. The scratch copy never had a `.env` file.

## Step-by-step log

### 1. Inventory

```
$ find . -type f | sort   (sizes in bytes)
5977     ./api/main.py
2803     ./db/aht.py
3424     ./db/erlang.py
7902     ./db/features.py
3726     ./db/forecast.py
6415     ./db/staffing.py
10716    ./db/train.py
1268     ./Makefile
17740    ./models/metadata.json
1743314  ./models/volume_lgbm.txt
5021     ./README.md
1800     ./tests/test_erlang.py
2317     ./tests/test_features.py
11522    ./web/index.html
```

14 files. No `__init__.py` in `db/`, `api/`, or `tests/` (implicit namespace packages; works on Python 3 but means the repo is not pip-installable). No `.gitignore`, no `.env`, no `requirements.txt`, no `pyproject.toml`, no CI config.

README and Makefile read in full. Makefile recipes are tab-indented (verified with `cat -vet`: lines 17, 29, 32, 35 start with `^I`; no space-indented lines).

**Dependency check.** Imports across `db/`, `api/`, `tests/`: `psycopg`, `dotenv`, `pandas`, `numpy`, `lightgbm`, `fastapi` (+ `fastapi.staticfiles`), `pytest`, and `uvicorn` (invoked by `make serve`). The README line

```
pip install "psycopg[binary]" python-dotenv pandas numpy lightgbm fastapi uvicorn pytest
```

covers every import. It is the only dependency spec in the repo and pins nothing. Versions that were actually exercised in this audit:

```
psycopg==3.3.5  psycopg-binary==3.3.5  python-dotenv==1.2.3  pandas==3.0.5  numpy==2.5.3
lightgbm==4.7.0  fastapi==0.141.1  starlette==1.6.0  pydantic==2.13.5  uvicorn==0.52.4  pytest==9.1.1
```

Note pandas 3.x and numpy 2.x: the code runs on them, but nothing records that.

**Native dependency not in the README.** The LightGBM 4.7.0 macOS wheel links `@rpath/libomp.dylib` with rpaths `/opt/homebrew/opt/libomp/lib` and `/opt/local/lib/libomp` and does not bundle the library:

```
$ otool -L .../lightgbm/lib/lib_lightgbm.dylib | grep omp
	@rpath/libomp.dylib (compatibility version 5.0.0, current version 5.0.0)
$ ls .../site-packages/lightgbm/lib/
lib_lightgbm.dylib            # no libomp.dylib alongside it
```

It imported here only because `/opt/homebrew/opt/libomp/lib/libomp.dylib` is installed on this machine. A teammate on a fresh Mac who follows the README will get `OSError: dlopen(...libomp.dylib...)` on `import lightgbm`.

### 2. `make db-check`

Without the variable (no `.env`, nothing in the shell):

```
$ make db-check PYTHON=/Users/moaizattiq/v4w-staffing/.venv/bin/python
/Users/moaizattiq/v4w-staffing/.venv/bin/python -c "\
import os, sys; \
...
conn.close()"
V4W_DATABASE_URL is not set
make: *** [db-check] Error 1
```

Exit code 2 from make. Clear message, no traceback.

With the variable loaded (`set -a; . /Users/moaizattiq/v4w-staffing/.env; set +a`):

```
calls         (93407, datetime.datetime(2021, 1, 1, 0, 54, 55), datetime.datetime(2025, 12, 31, 23, 55, 58))
holidays      (65,)
calls_hourly  (35872,)
```

Exit 0. Note the data ends 2025-12-31.

### 3. `make test`

With the database URL:

```
$ make test PYTHON=...
/Users/moaizattiq/v4w-staffing/.venv/bin/python -m pytest tests -q
............F....                                                        [100%]
=================================== FAILURES ===================================
_______________________ test_required_agents_known_case ________________________
        agents = required_agents(30, 600, 0.80, 30)
>       assert agents == 6
E       assert 8 == 6
tests/test_erlang.py:77: AssertionError
FAILED tests/test_erlang.py::test_required_agents_known_case - assert 8 == 6
1 failed, 16 passed in 1.19s
make: *** [test] Error 1
```

16 pass, 1 fail. The failure is the known `8 == 6` assertion at `tests/test_erlang.py:77`; reported here without judgment (the math-auditor owns it). Because of it `make test` exits non-zero.

Without the database URL:

```
$ python -m pytest tests -q
1 failed, 12 passed, 4 errors in 0.05s
ERROR tests/test_features.py::test_no_nulls_in_any_column - Failed: V4W_DATABASE_URL is not set
ERROR tests/test_features.py::test_row_count_unchanged - ...
ERROR tests/test_features.py::test_hour_encoding_on_unit_circle - ...
ERROR tests/test_features.py::test_is_holiday_sums_to_plausible_count - ...
```

All four feature tests hard-fail (not skip) offline, by design (`tests/test_features.py:26-28`). The 12 Erlang unit tests pass with no database.

### 4. `make train`

With the database URL. Wall time 55 s (README says 2-3 min), exit 0:

```
$ time make train PYTHON=...
grid rows 43,824  (view rows 35,872, zero-filled 7,952)  features 14
fold 2024-01  train=26,280  test=744
...
fold 2025-12  train=43,080  test=744

## Overall (pooled over all test hours)
| model | mae | rmse | mape_pct | daily_mae |
|---|---|---|---|---|
| lgbm_poisson | 1.256 | 1.632 | 53.016 | 9.281 |
| seasonal_naive | 1.336 | 1.778 | 56.416 | 10.341 |
| overall_mean | 1.463 | 1.946 | 50.844 | 12.664 |

saved models/volume_lgbm.txt and models/metadata.json
32.99s user 122.36s system 280% cpu 55.315 total
```

Both files were rewritten in the scratch copy (mtime 15:11, sizes unchanged: 1743314 and 17740 bytes).

**Determinism vs the original repo's `models/`:**

| Artifact | Result |
|---|---|
| `models/volume_lgbm.txt` | byte-identical (sha256 `3a773b05...ba33a` before and after) |
| `models/metadata.json` `backtest.overall` | numerically identical, all 12 floats to full precision |
| `models/metadata.json` `backtest.per_fold` | identical |
| every other key | identical except `created_at_utc` (`2026-09-13T19:56:26+00:00` → `2026-09-13T20:11:58+00:00`) |

`cmp` reports the only byte difference at line 2 (the timestamp). Training is fully reproducible on the same machine and data.

### 5. `make serve` on port 8799

Started with `nohup make serve PYTHON=... PORT=8799 &` from the scratch root with the database URL in the shell. Answered within about 1 s. Log confirms `--reload` with StatReload watching the scratch directory.

| Request | Status | Time | First part of body |
|---|---|---|---|
| `GET /api/model` | 200 | 0.002 s | `{"created_at_utc":"2026-09-13T20:11:58+00:00","target":"offered","train_start":"2021-01-01T00:00:00","train_end":"2025-12-31T23:00:00","train_rows":43824,"features":["dow","hour","month",...` |
| `GET /api/staffing?start=2026-01-01&end=2026-01-07` | 200 | 1.02 s | `{"params":{"start":"2026-01-01","end":"2026-01-07","efficiency":0.65,"target_pct":0.8},"days":[{"date":"2026-01-01","predicted_calls":70.19,"shifts":{"Overnight":{"predicted_calls":11.66,"agents_required":2,"staff_needed":4,"capped":false},"Day":{...` |
| `GET /api/forecast?start=2026-01-01&end=2026-01-01` | 200 | 1.03 s | `{"params":{...},"rows":[{"hour_start":"2026-01-01T00:00:00","predicted_offered":2.176109083,"predicted_aht_sec":936.4076875335,"offered_load_erlangs":0.5660347984,"agents_required":2,"scheduled_needed":4,"service_level_at_schedul...` |
| `GET /` | 200 | 0.003 s | `<!doctype html><html lang="en"><head><meta charset="utf-8">...<title>Vets4Warriors Staffing Forecast</title>...` |
| `GET /api/staffing?...&efficiency=1.5` | **422** | 0.001 s | `{"detail":[{"type":"less_than_equal","loc":["query","efficiency"],"msg":"Input should be less than or equal to 1","input":"1.5","ctx":{"le":1.0}}]}` |
| `GET /api/staffing?start=2026-01-07&end=2026-01-01` | **400** | 0.001 s | `{"detail":"end 2026-01-01 is before start 2026-01-07"}` |

Extra probes:

| Request | Status | Body |
|---|---|---|
| `GET /api/staffing` (no params) | 200 | defaults to `2026-09-13` → `2026-10-12` (30 days, server local date) |
| `GET /api/staffing?start=&end=2026-01-07` (what the page sends if a date input is cleared) | 422 | `{"detail":[{"type":"date_from_datetime_parsing","loc":["query","start"],"msg":"Input should be a valid date or datetime, input is too short",...` |
| `GET /api/staffing?start=2026-01-01&end=2027-01-02` | 400 | `{"detail":"range is 367 days; maximum is 366"}` |
| `GET /api/forecast?start=2030-06-01&end=2030-06-01` | 200 | forecasts fine; warning below printed to server log |

Server stderr on every forecast/staffing request:

```
.../mvp-clean/db/staffing.py:108: UserWarning: holidays table ends 2025-12-31; holidays after that are not flagged in the forecast. Load future holidays first.
```

Shutdown: `pkill -f "uvicorn api.main:app --host 127.0.0.1 --port 8799"` plus killing the make process. Verified afterwards: `lsof -iTCP:8799` empty, `pgrep -fl "uvicorn api.main"` empty. No server left running.

### 6. Run from a different working directory

```
$ cd /tmp && PYTHONPATH=<scratch> python -m uvicorn api.main:app --host 127.0.0.1 --port 8798
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    raise RuntimeError(f"Directory '{directory}' does not exist")
RuntimeError: Directory 'web' does not exist
(exit 1)

$ cd /tmp && PYTHONPATH=<scratch> python -c "from db.forecast import _load_model; _load_model()"
FileNotFoundError: models/volume_lgbm.txt not found. Run `python -m db.train` first.

$ cd /tmp && PYTHONPATH=<scratch> python -c "from db.train import MODEL_PATH; import os; print(os.path.abspath(MODEL_PATH))"
/private/tmp/models/volume_lgbm.txt
```

So: the server will not start from any cwd other than the repo root, forecasting fails, and `python -m db.train` from elsewhere would write `models/` into whatever the cwd is. The "must run from repo root" assumption is real and unguarded.

`.env` discovery is less cwd-bound than the paths. Tested with a throwaway package in the scratchpad: with python-dotenv 1.2.3, `load_dotenv()` called from a module walks up from that module's directory, so a repo-root `.env` is found from any cwd when running via `python -m ...` (verified from `/tmp`). The exception is `python -c` (the `db-check` recipe), where dotenv uses the cwd, so `make db-check` only sees `.env` when run from the repo root. Since the Makefile is run from the root anyway this is consistent, just worth knowing.

### 7. Secrets scan

```
$ grep -rniE "postgres(ql)?://|password|secret|api[_-]?key|token|supabase|okqdxhvbcdhzriftwute" . --exclude=volume_lgbm.txt
README.md:60:V4W_DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DBNAME
```

One hit, a placeholder, not a credential. The model dump `models/volume_lgbm.txt` matched nothing. `models/metadata.json` contains only feature names, params, and metrics (no URLs, hosts, or paths). No `.env` exists in the original repo or the scratch copy. Makefile and tests contain no credentials.

**No `.gitignore` exists** in the original repo. The README instructs creating `.env` with a real password at the repo root, and `.pytest_cache/` already sits untracked in the original. Nothing is committed yet, so nothing has leaked, but the first `git add -A` would commit `.env` if one has been created. A `.gitignore` should at minimum contain:

```
.env
.env.*
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

and a deliberate decision on `models/` (1.7 MB binary that `make train` regenerates deterministically; either ignore it and document `make train` as a setup step, or commit it and accept binary churn). Not created, per instructions.

### 8. Hardcoded path scan

```
$ grep -rnE "/Users/|/home/|~/|C:\\\\" .   (excluding the model dump)
(no hits)
```

No absolute paths anywhere in code, Makefile, README, or web. Every path is cwd-relative:

| file:line | Path | Assumes |
|---|---|---|
| `db/train.py:66-68` | `Path("models")`, `models/volume_lgbm.txt`, `models/metadata.json` | cwd = repo root (read by forecast and API, written by train) |
| `api/main.py:48`, `api/main.py:176` | `Path("web")` mounted as static root | cwd = repo root; raises at import otherwise |
| `api/main.py:167-170` | `METADATA_PATH` re-read per `/api/model` call | cwd = repo root |
| `db/forecast.py:42-50` | `MODEL_PATH`, `METADATA_PATH` re-read per forecast | cwd = repo root |
| `Makefile:8` | comment says "All targets run from the repo root" | documented, not enforced |
| `api/main.py:28`, `db/aht.py:22`, `db/features.py:27`, `db/forecast.py:26`, `db/train.py:38`, `Makefile:19` | `load_dotenv()` with no path | walks up from the module dir (works from any cwd via `-m`); cwd-only in the `-c` one-liner |
| `tests/test_erlang.py:12`, `tests/test_features.py:15` | `sys.path.insert(0, <tests>/..)` | makes `db` importable without packaging; works from any cwd |

### 9. Makefile portability

- Recipes use real tabs (verified). No space-indented recipe lines.
- `make -n` for each target expands cleanly (see the db-check expansion in step 2; `train` → `python -m db.train`; `serve` → `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload`; `test` → `python -m pytest tests -q`).
- Default `SHELL = /bin/sh` (bash 3.2 in sh mode on this Mac). The multi-line `python -c "\ ... "` one-liner in `db-check` was run with `SHELL=/bin/sh`, `/bin/bash`, `/bin/zsh`, and `/bin/dash`: all four parsed it and printed `V4W_DATABASE_URL is not set`. The backslash-newline continuation inside double quotes is POSIX-safe.
- `HOST`/`PORT` work both as `make` arguments and as environment variables (`?=` respects the environment), matching the README's table.
- `make serve` bakes in `--reload`; fine for a demo, wrong for anything shared.

### 10. Fresh-machine review of `api/main.py` and `web/index.html`

- **CORS**: none configured; the page is served by the same FastAPI process at `/`, so requests are same-origin and this is fine. Anyone opening `web/index.html` from disk (`file://`) instead of the server gets nothing, since all fetches are root-relative.
- **Default horizon**: `DEFAULT_HORIZON_DAYS = 30` in both `api/main.py:46` and `web/index.html:116`; both compute end = start + 29. No mismatch.
- **Default dates**: the page uses `new Date().toISOString().slice(0, 10)` (`web/index.html:123-128`), which is the UTC date; the server default uses local `date.today()` (`api/main.py:58`). The page always sends explicit dates, so the server default is only hit by direct API calls. Effect: any user west of UTC opening the page in the evening (after 19:00 CDT, 17:00 PDT) gets tomorrow as the default start. At audit time (15:12 CDT) both agreed on 2026-09-13.
- **Validation error rendering**: FastAPI's 422 body has `detail` as an array; `web/index.html:149-150` interpolates it into a string, so the table shows `422: [object Object]` for out-of-range params or a cleared date input (verified the template expression with node). The README's API section documents only the 400 path.
- **Every request** reloads the 1.7 MB booster and rescans all 93k answered calls for AHT (`db/forecast.py:79,93`); measured about 1.0 s per request. README acknowledges. Slider input is debounced 250 ms, so dragging fires a request every quarter second; the `inflight` counter discards stale responses, so the UI stays consistent.
- **Startup failure mode**: missing `psycopg` calls `sys.exit` at import (`api/main.py:33`), which is a clean message under uvicorn. A missing `models/` returns 503 with a "run make train" message on the API (`api/main.py:89-90`) and 404 on `/api/model`; the page shows the 503 text in the table. Good.
- `warnings.warn` for the holiday calendar fires per request into the server log (`db/forecast.py:66-73`) and is never surfaced in the JSON or the page.

## Findings

### CRITICAL

None. No secret is present in the tree, and nothing is committed yet.

### HIGH

1. **No `.gitignore`, while the README tells users to create `.env` with a live password in the repo root.** Files: (missing) `.gitignore`; `README.md:56-61`. The first `git add -A` after setup commits the credential, plus `.pytest_cache/` (already present untracked) and `__pycache__/`. Fix: add the `.gitignore` listed in step 7 before the first commit, and decide whether `models/` is committed.
2. **LightGBM's macOS wheel needs Homebrew `libomp`, and the README does not say so.** `README.md:49-53`. On a fresh Mac `import lightgbm` fails with a `dlopen ... libomp.dylib` error, which kills `make train`, `make serve`, and `make test` (feature tests import through `db.features`, and `db.forecast` imports lightgbm). Fix: add `brew install libomp` to Setup, or pin a LightGBM build that bundles it.
3. **No dependency manifest and no version pins.** Only `README.md:52`. The line is complete for today's imports, but pandas 3.x / numpy 2.x / psycopg 3.3 are what was tested and nothing records that; a `pip install` next month may resolve differently. Fix: add `requirements.txt` (or `pyproject.toml`) with pinned versions from the list in step 1, and have the README point to it.
4. **The holidays table ends 2025-12-31, so every forecast the demo can show runs with the holiday features effectively off.** Data issue, surfaced by `db/forecast.py:66-73` as a per-request stderr warning; documented in README "Known gaps". For any date after 2026-01-14, `is_holiday` is 0 and `days_to_nearest_holiday` saturates at -14 for every hour, so Thanksgiving, Christmas, Veterans Day 2026 will forecast as ordinary days in the dashboard. Fix: load 2026-2027 rows into `holidays` before the demo; consider returning the warning in the API `params` block or the page banner so it is not invisible.
5. **All model and static paths are relative to the process cwd.** `db/train.py:66-68`, `api/main.py:48,176`. The server refuses to start from any other directory (`RuntimeError: Directory 'web' does not exist`), forecasting raises `FileNotFoundError`, and a stray `python -m db.train` writes `models/` into the cwd. Fix: anchor on the file, e.g. `REPO_ROOT = Path(__file__).resolve().parent.parent` and `MODEL_DIR = REPO_ROOT / "models"`, `WEB_DIR = REPO_ROOT / "web"`; or make `V4W_MODEL_DIR` an env var.

### MEDIUM

6. **`make test` exits non-zero on a known failing assertion.** `tests/test_erlang.py:77` (`assert 8 == 6`). Reported, not judged; whichever side is right, the target is red until it is resolved, so it cannot gate anything.
7. **The feature tests hard-fail without a database instead of skipping.** `tests/test_features.py:26-28` (`pytest.fail`). Offline or in CI without the DB, `make test` shows 4 errors. Fix: `pytest.skip` when `V4W_DATABASE_URL` is unset, or a marker (`-m "not db"`) so the pure Erlang tests can run alone.
8. **Page default dates are UTC; server defaults are local.** `web/index.html:123-128` vs `api/main.py:57-59`. Evening users in the Americas get tomorrow as the default start. Fix: build the ISO string from local `getFullYear/getMonth/getDate` in JS, or have the page ask the API for its defaults.
9. **422 validation errors render as `[object Object]` in the page.** `web/index.html:149`. Triggered by a cleared date input or any out-of-range value from a hand-edited URL. Fix: if `detail` is an array, join `detail.map(d => d.msg)`. Also document the 422 path in the README API section.
10. **`.env` handling is duplicated and inconsistent.** `load_dotenv()` at import in five modules (`api/main.py:28`, `db/aht.py:22`, `db/features.py:27`, `db/forecast.py:26`, `db/train.py:38`) plus once in `Makefile:19`. Works, but the `python -c` form in `db-check` resolves `.env` from cwd while the modules resolve from their own directory, so the two can disagree if someone runs a target from outside the root. Fix: one `load_dotenv(REPO_ROOT / ".env")` in a tiny `db/env.py` (or drop it from library modules and load only at the entry points).

### LOW

11. **Docstrings reference a file that is not in this repo.** `db/train.py:22`, `db/forecast.py:12`, `db/features.py:16`, `db/aht.py:11` all say "same pattern as db/load_calls.py"; that file exists only in the main repo. Fix: reword or drop the reference.
12. **Per-request holiday warning spams the server log.** `db/forecast.py:66-73` via `db/staffing.py:108`. Every staffing call prints the same two lines. Fix: warn once at startup or return it as a field.
13. **`make serve` always passes `--reload`.** `Makefile:32`. Fine for one developer; add a `serve-prod` target or make the flag a variable before anyone points a shared box at it.
14. **README timing is off for this hardware.** `README.md:70` says 2-3 minutes; measured 55 s wall on this machine. Harmless.
15. **No `__init__.py` files.** `db/`, `api/`, `tests/` rely on namespace packages and a `sys.path.insert` in each test file. Works on Python 3.12 as long as cwd or `PYTHONPATH` includes the root; a `pyproject.toml` with a package layout would remove the `sys.path` hack.
16. **Per-request cost about 1 s** from reloading the booster and rescanning the `calls` table for AHT (`db/forecast.py:79,93`). Acknowledged in the README; a module-level cache keyed on the model file's mtime would keep the "retrain without restart" behaviour and drop this to milliseconds.

## What did not run

Nothing was skipped. Every step above executed. No fix was applied to either copy; the scratch copy is left in place with its retrained `models/` and a `.pytest_cache/` created by the test run. No `uvicorn` or `db.train` process is running.
