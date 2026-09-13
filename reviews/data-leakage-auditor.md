# Data-leakage audit: `db/train.py` backtest and train/predict feature parity

Audited 2026-09-13 against the live database (`V4W_DATABASE_URL`), the repo at
`/Users/moaizattiq/vets4war-dummy`, and the saved artifacts in `models/`
(both dated Sep 13 14:56). Read-only review; nothing in the repo was changed
except this file. All numbers below are pasted from the verification scripts
that were run; none are estimated.

## Verdict: NO LEAK FOUND

No test-month `offered` value reaches any training set, any baseline, or any
prediction in the rolling-origin backtest. Fold boundaries are strict and
verified on the real grid. `build_features` produces byte-identical feature
rows in the training path and the forecast path, and the saved booster's
feature order matches `metadata["features"]`.

Two things are **leak-adjacent** and are documented honestly below, but
neither moves test-period target values into a fold:

1. The holiday `volume_multiplier` / `evening_multiplier` columns are a fixed
   reference table of constants (13 distinct values, one per holiday name,
   identical in every year 2021-2025). Their provenance is a CSV loaded by
   `load_calls.py --holidays`; they were not measured from this dataset's
   per-year ratios (they disagree with the actuals by a wide margin, see
   check 12). Ablation on the July 2024 fold shows they do not help.
2. The final saved model is fit on all data through 2025-12-31 (intended, and
   stated in the docstring). The backtest metrics in `metadata.json` describe
   the *procedure*, not the saved artifact.

Environment: lightgbm 4.7.0, pandas 3.0.5, numpy 2.5.3, Python 3.12.

## Data shape (context for every check)

```
view rows 35872 min 2021-01-01 00:00:00 max 2025-12-31 23:00:00
grid rows 43824 min 2021-01-01 00:00:00 max 2025-12-31 23:00:00
features ['dow', 'hour', 'month', 'week_of_year', 'days_since_epoch', 'hour_sin',
          'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'is_holiday',
          'days_to_nearest_holiday', 'holiday_volume_multiplier', 'holiday_evening_multiplier']
featured cols ['hour_start', 'offered', <the 14 above>]
```

43,824 = 5 years x 365.25 x 24 exactly (2024 is a leap year); 7,952 hours
zero-filled.

---

## Check 1: `complete_hourly_grid` built from the full dataset

**Code.** `db/train.py:77-89`. `first = hourly.hour_start.min().normalize()`,
`last = hourly.hour_start.max().normalize() + 23h`, `pd.date_range(first, last,
freq="h")`, then `reindex(grid, fill_value=0)` on `offered`. Called once in
`main()` at `db/train.py:263` before any split.

**What crosses the split.** Only the first and last calendar dates. Those
determine the grid extent, which is calendar knowledge, not a target value.
The zero-fill is per-hour and does not read any other hour's `offered`.
Test-month rows carry their own true `offered` (or 0 for hours with no calls),
which is exactly the actual the backtest should score against.

**Conclusion.** Not a leak. The grid extent being known in advance is the
same assumption the backtest itself makes (it iterates a fixed list of months).

## Check 2: `build_features` called once on the full frame; feature classification

**Code.** `db/train.py:264` `featured = build_features(grid, holidays)` with
`epoch=None`; `db/features.py:187-210`. No feature reads `offered` or any other
row: `_calendar_features` (`features.py:107-123`) reads only `hour_start`;
`_holiday_features` (`features.py:158-184`) reads only `hour_start` and the
`holidays` table. There are no lags, rolling stats, target encodings, or
group aggregates (the docstring at `features.py:10-14` says so and the code
confirms it).

| Feature | Source | Class |
|---|---|---|
| dow, hour, month, week_of_year | `hour_start` only | pure calendar |
| hour_sin, hour_cos, dow_sin, dow_cos, is_weekend | `hour_start` only | pure calendar |
| days_since_epoch | `hour_start` minus anchor | pure calendar given a fixed anchor (see below) |
| is_holiday | `holidays.holiday_date` | holidays table, date only |
| days_to_nearest_holiday | `holidays.holiday_date` | holidays table, date only |
| holiday_volume_multiplier | `holidays.volume_multiplier` | holidays table, numeric constant (check 12) |
| holiday_evening_multiplier | `holidays.evening_multiplier` | holidays table, numeric constant (check 12) |

**days_since_epoch anchor.** With `epoch=None` the anchor is
`hour_start.min()` of the frame passed in (`features.py:206`). In `train.py`
that frame is the full grid, so the anchor is 2021-01-01, which is the
*earliest* date, i.e. the start of the training window, not anything in a
test month. Knowing the first day of your data is not a leak. `forecast.py:84`
passes `epoch=metadata["train_start"]` = `"2021-01-01T00:00:00"`, which
resolves to the same anchor (proven numerically in check 9).

**Holiday dates for test months.** `_days_to_nearest_holiday` for late-train
rows (e.g. 2023-12-30) looks ahead to 2024-01-01, which is in the first test
month. That is a calendar date, fixed years in advance; the README
(line 118) already requires future holidays to be loaded for the forecast.
Not a leak.

**Conclusion.** Twelve of fourteen features are pure calendar. The remaining
two (multipliers) are addressed in check 12. Calling `build_features` once
before splitting is safe *because* no feature depends on other rows; if a lag
or rolling feature were ever added to `build_features`, this call site would
become a leak immediately (see Findings, LOW #3).

## Check 3: `split_fold` boundaries

**Code.** `db/train.py:153-160`.
```python
month_end = month_start + pd.offsets.MonthBegin(1)
is_test = (hour_start >= month_start) & (hour_start < month_end)
train = featured[hour_start < month_start]
```
Train is strictly `< month_start`; test is the half-open `[month_start,
month_end)`. Nothing at or after `month_end` is used by the fold.

**Evidence** (three folds on the real grid; `after` = rows at/after
month_end; `sum_ok` = train + test + after == 43,824; `overlap` = shared
index labels):
```
2024-01-01 train max 2023-12-31 23:00:00 test min 2024-01-01 00:00:00 test max 2024-01-31 23:00:00 n_train 26280 n_test 744 expected_test 744 after 16800 sum_ok True overlap 0
2025-06-01 train max 2025-05-31 23:00:00 test min 2025-06-01 00:00:00 test max 2025-06-30 23:00:00 n_train 38688 n_test 720 expected_test 720 after 4416 sum_ok True overlap 0
2025-12-01 train max 2025-11-30 23:00:00 test min 2025-12-01 00:00:00 test max 2025-12-31 23:00:00 n_train 43080 n_test 744 expected_test 744 after 0 sum_ok True overlap 0
```
`assert train.hour_start.max() < test.hour_start.min()` passed for all three.
26,280 = 3 x 365 x 24 (2021-2023 have no leap day). 744 = 31 x 24, 720 = 30 x 24.

**Conclusion.** No off-by-one, no overlap, no gap.

## Check 4: `predict_seasonal_naive`

**Code.** `db/train.py:109-119`. `window_start = test.hour_start.min() - 56
days`; `window = train[train.hour_start >= window_start]`; medians grouped by
`(dow, hour)` on `window` only; reindexed onto the test rows' `(dow, hour)`
keys. The test frame is used only to read the window start date and the
`(dow, hour)` calendar keys, never `test["offered"]`.

**Evidence** (fold 2024-01):
```
window_start 2023-11-06 00:00:00 window min 2023-11-06 00:00:00 window max 2023-12-31 23:00:00 rows 1344 cells (168,)
naive pred shape (744,) nan False
```
1,344 = 56 x 24; 168 = 7 x 24 cells, all populated.

**Conclusion.** Train only. Correct.

## Check 5: `predict_overall_mean`

**Code.** `db/train.py:122-123`: `np.full(len(test), train[TARGET].mean())`.
Reads only `len(test)`. **Conclusion.** Train only.

## Check 6: `fit_lgbm`

**Code.** `db/train.py:100-102`: `lgb.Dataset(train[features],
label=train[TARGET])`, `lgb.train(LGBM_PARAMS, dataset,
num_boost_round=500)`. No `valid_sets`, no `callbacks`, no
`early_stopping_rounds`, no `eval` set. Params:
```
{'objective': 'poisson', 'learning_rate': 0.05, 'num_leaves': 31, 'min_data_in_leaf': 20,
 'feature_fraction': 0.9, 'seed': 42, 'verbose': -1} rounds 500
```
`predict_lgbm` (`train.py:105-106`) passes `test[features]` only; `offered` is
not in `features` (`feature_columns`, `train.py:92-93`, excludes it).

**Conclusion.** No path from test rows into fitting. Fixed 500 rounds means
no data-driven stopping at all.

## Check 7: pooled metrics over test rows only

**Code.** `db/train.py:172-200`. Per fold: `actual = test[TARGET]`;
`pooled[name].append(predictions[name])`; `pooled_hours.append(test["hour_start"])`;
`pooled_actual.append(actual)`. Then `score(concat(hours), concat(actual),
concat(pooled[name]))`. Only `test` frames are ever appended.

**Evidence.**
```
folds 24 pooled test rows 17544 = hours in 2024+2025: True
```
(366 + 365) x 24 = 17,544. Every 2024-2025 hour is scored exactly once, as a
test hour, by a model that never saw its month.

**Conclusion.** Correct. Also noted: `passes_gate` (`train.py:217-219`)
compares pooled `daily_mae` only, and the gate runs *before* the final fit at
`train.py:276-281`, so a failing model is never saved.

## Check 8: final model on all data

**Code.** `db/train.py:281` `booster = fit_lgbm(featured, features)`;
`save_model` writes `train_start`/`train_end` from the full frame
(`train.py:234-235`).

**Evidence.**
```
metadata train_start 2021-01-01T00:00:00 train_end 2025-12-31T23:00:00 rows 43824
```

**Conclusion.** Intended and documented. Not a backtest leak: the backtest
numbers were computed by 24 separate boosters that each saw only prior
months. Caveat for readers of `metadata.json`: the "backtest" block reports
the accuracy of the *procedure*, not of the booster in `volume_lgbm.txt`,
which has seen 2024-2025. Anyone re-scoring the saved booster on 2024-2025
would get in-sample numbers and must not present them as the backtest.

## Check 9: feature parity train vs forecast

**Code.** Train: `train.py:264` `build_features(grid, holidays)` (epoch=None,
grid min 2021-01-01 00:00). Forecast: `forecast.py:84`
`build_features(grid, holidays, epoch=metadata["train_start"])`;
`forecast.py:86-91` selects `featured[metadata["features"]]` in metadata
order.

**Evidence** (grid 2025-06-01..2025-06-02 built three ways; compared to the
slice of the full training frame):
```
epoch string 2021-01-01T00:00:00
forecast-way vs full-grid slice equal: True
small-grid epoch=None vs slice equal: False
days_since_epoch fc-way 1612 slice 1612 epoch=None small grid 0
differing cols fc-way: []
dtypes equal: True
metadata features == computed features (order): True
booster feature_name == metadata features: True
booster feature_name: ['dow', 'hour', 'month', 'week_of_year', 'days_since_epoch', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend', 'is_holiday', 'days_to_nearest_holiday', 'holiday_volume_multiplier', 'holiday_evening_multiplier']
```

**Conclusion.** The forecast path reproduces the training features exactly
(values and dtypes) for the same `hour_start`. Column order at predict time
equals the order the booster was trained with (LightGBM stores
`feature_name` and the saved list matches position for position). The
`epoch=None` default is a proven footgun: on a small grid it yields
`days_since_epoch = 0` instead of 1612, which would make the model forecast
2021 levels. `forecast.py` avoids it correctly; nothing else in the repo
calls `build_features` without an epoch except `train.py` on the full grid.

## Check 10: AHT

**Code.** `db/aht.py:37-41` selects every answered call with no date cutoff;
`db/aht.py:56-58` weights by age relative to the newest call. `db/train.py`
does not import or reference `aht` (grep for `aht`/`AHT` in `train.py`
returned nothing).

**Evidence.**
```
answered calls (datetime(2021,1,1,0,54,55), datetime(2025,12,31,23,42,57), 85879)
```

**Conclusion.** Zero effect on the backtest metrics; `train.py` never touches
AHT. AHT only enters `forecast_range` (`forecast.py:93`) and therefore only
the staffing numbers. Since the forecast is meant for dates after
2025-12-31, using all history through 2025-12-31 is the correct,
non-leaking choice for production use. It would only be a concern if someone
ran `forecast_range` for a 2024-2025 date and called the resulting staffing
numbers a backtest; there is no AHT backtest in the repo, so nothing is
currently misreported.

## Check 11: nondeterminism

**Code.** `grep -rn 'random\|shuffle\|\.sample(' db/*.py` returned nothing.
`seed: 42` in `LGBM_PARAMS` (`train.py:61`); `feature_fraction: 0.9` is the
only stochastic element and is governed by that seed.

**Evidence** (fold 2024-01 fitted twice in one process):
```
identical predictions: True max abs diff: 0.0
```

**Conclusion.** Deterministic on this machine. `deterministic=True` and
`num_threads` are not pinned, so bit-identical results across machines or
thread counts are not guaranteed (LightGBM histogram construction can differ
under different thread counts). Not a leakage issue.

## Check 12: holiday multipliers

**Code.** `features.py:173-174` maps `volume_multiplier` and
`evening_multiplier` from the holidays table onto every hour of a holiday
date (0.0 otherwise). Both are applied to all 24 hours identically;
`evening_multiplier` is not restricted to evening hours.

**What the table actually contains** (queried live; 65 rows, 2021-01-01 to
2025-12-31, every `volume_multiplier` non-null):
```
distinct names vs distinct (name, volume_multiplier, evening_multiplier) tuples: (13, 13)
```
So there are exactly 13 constants, one per holiday name, repeated unchanged
in each of the five years (e.g. Independence Day 1.550/1.850 in 2021, 2022,
2023, 2024, 2025). They are not per-year measurements.

**Provenance.** The holidays table is populated from a CSV
(`v4w_holidays_2021_2025.csv`) by `load_calls.py --holidays` in the sibling
repo; `schema.sql:65-66` calls it a "reference calendar" whose multipliers
are "nullable: fill them in once measured from real data". I could not find
the CSV or any code that computes the values, so how they were derived is
unverifiable from the repos.

**Do they encode this dataset's test-period ratios?** Compared each holiday's
actual daily total to the mean of the same weekday +/-1 and +/-2 weeks:
```
        date                name table_mult  actual_total  neighbor_mean  actual_ratio
2021-07-04    Independence Day      1.550            84           36.8         2.286
2022-07-04    Independence Day      1.550           108           50.5         2.139
2023-07-04    Independence Day      1.550           108           48.5         2.227
2024-07-04    Independence Day      1.550           165           46.5         3.548
2025-07-04    Independence Day      1.550            82           57.0         1.439
2024-06-19          Juneteenth      1.040            38           53.8         0.707
2024-09-11        September 11      1.280            59           70.5         0.837
2024-12-25       Christmas Day      1.450           158           64.8         2.440
2025-01-01      New Year's Day      1.300            77           87.0         0.885
2025-02-14     Valentine's Day      1.120            50           57.5         0.870
```
(full 65-row table was inspected; these are representative.) The table
constants are far from the realised ratios in the test years and do not
track them year to year. They were not fitted to 2024-2025 outcomes.

**Ablation** (fold 2024-07, the fold containing the most extreme holiday):
```
all 14 features:            daily_mae=11.923 mae=1.307  Jul4 predicted total=82.7 actual=165
without 2 multiplier cols:  daily_mae=11.169 mae=1.311  Jul4 predicted total=76.6 actual=165
naive daily_mae: 13.581
```
Removing the two multiplier columns slightly *improves* daily MAE on this
fold, and the model with them still underpredicts July 4 by half. The
columns carry no effective test-period information.

**Conclusion.** Leak-adjacent in principle (a numeric column whose value was
set by a human with access to all years), but in practice a per-holiday
constant that is (a) identical across train and test years, so the model
cannot learn anything test-specific from it that `is_holiday` +
`holiday_name`-equivalent identity does not already provide, (b) clearly not
calibrated to the 2024-2025 realised ratios, and (c) ablated to no benefit.
Since each holiday name maps to a unique multiplier pair, the two columns
are functionally a categorical encoding of "which holiday", i.e. calendar
knowledge. Not a leak of target values. It is a data-quality and
documentation issue (see Findings).

---

## Findings (ranked)

No CRITICAL or HIGH findings.

### MEDIUM-1: holiday multiplier columns have unverifiable provenance and no measured benefit
- `db/features.py:173-174`, `:182-183`; holidays table (loaded by sibling repo `db/load_calls.py:311-326`).
- They are 13 per-holiday constants of unknown origin, applied to all 24 hours (the "evening" multiplier is not restricted to evening hours), and the July 2024 ablation shows daily MAE 11.92 with them vs 11.17 without. Functionally they are a holiday-identity code, which `is_holiday` already provides in a coarser form.
- Recommended fix (not applied): either (a) drop the two multiplier columns from `build_features` and re-run the gate, or (b) document the CSV's derivation and, if the values were computed from call data, recompute them from data strictly before `FIRST_TEST_MONTH` (2024-01-01) so the backtest cannot be questioned. If keeping `evening_multiplier`, apply it only for evening hours (`hour >= 16` or whatever the shift definition is) so the column means what its name says.

### LOW-1: `epoch=None` default silently re-anchors the trend on the input frame
- `db/features.py:190`, `:206`; `db/train.py:264`.
- Verified: a 2-day grid with `epoch=None` yields `days_since_epoch = 0` where the training value is 1612. `forecast.py` handles it correctly today, but any new caller (a notebook, a test, a second forecaster) that forgets `epoch` gets a model that predicts 2021 volumes with no error raised.
- Recommended fix: make `epoch` a required argument, and have `train.py` pass `epoch=grid["hour_start"].min()` explicitly and write that same value to `metadata["train_start"]`, so the anchor is one named value rather than an implicit default.

### LOW-2: `build_features` is called once on the full frame before splitting
- `db/train.py:264` then `split_fold` at `:182`.
- Safe today only because every feature is row-local calendar/holiday data. Adding any lag, rolling mean, or target-encoded feature to `build_features` would leak test months into training with no warning, and the "feature parity" guarantee would then be false too (forecast grids have no history).
- Recommended fix: add a comment at `train.py:264` stating the invariant ("build_features must remain row-local; if history-dependent features are added, features must be built per fold from `train` only"), and add a unit test in `tests/test_features.py` asserting that `build_features` on a single-row frame equals the corresponding row from a multi-row frame given the same epoch.

### LOW-3: saved model has seen 2024-2025; metadata backtest block can be misread
- `db/train.py:281`, `:240-245`.
- Intended and documented in the module docstring, but `metadata.json` places the backtest metrics beside `train_end: 2025-12-31`, which invites reading them as the saved booster's out-of-sample error.
- Recommended fix: add a `"backtest_note"` string to metadata (e.g. "metrics are from 24 rolling-origin boosters; the saved booster was refit on all rows through train_end") and surface the same sentence in the README's error-bar paragraph.

### LOW-4: cross-machine reproducibility not pinned
- `db/train.py:55-63`.
- `seed` is set and repeated fits are bit-identical here, but `deterministic` and `num_threads` are not set, so a different machine or thread count may produce slightly different boosters and metrics.
- Recommended fix: add `"deterministic": True, "num_threads": <fixed>` to `LGBM_PARAMS` if exact reproducibility of the reported numbers matters for the demo.

### INFO: AHT has no date cutoff
- `db/aht.py:37-41`.
- Correct for production forecasting (uses all history for future dates). Not used by `train.py`, so the backtest metrics are unaffected. Only becomes a concern if someone ever backtests staffing numbers; that would need a `as_of` cutoff parameter in `estimate_aht`.

## Verification scripts

Run from the repo root with `V4W_DATABASE_URL` set, using
`/Users/moaizattiq/v4w-staffing/.venv/bin/python`; the scripts import
`db.train` functions and never call `main()`, so `models/` is untouched:
- `/private/tmp/claude-501/-Users-moaizattiq/b9883508-bb1c-48a6-9c88-6f94889e7936/scratchpad/audit.py` (checks 3, 4, 9, 12 table, 6, 10)
- `/private/tmp/claude-501/-Users-moaizattiq/b9883508-bb1c-48a6-9c88-6f94889e7936/scratchpad/audit2.py` (checks 11, 12 ablation, 7)
