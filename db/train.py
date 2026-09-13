"""Train and backtest the hourly call-volume model.

Target    : `offered` per hour, from the calls_hourly view, on a COMPLETE
            hourly grid (hours absent from the view are zero calls).
Model     : LightGBM, objective="poisson" (count data).
Features  : db.features.build_features. Imported, never reimplemented.

Backtest  : rolling origin. First fold trains 2021-01-01..2023-12-31 and
            predicts 2024-01. Then the test month walks forward one month at
            a time through 2025-12, retraining on everything before it.
            Every fold predicts a full month with no access to that month.

Baselines : (a) seasonal naive = median `offered` for the same (dow, hour)
                over the 8 weeks before the test month
            (b) overall mean of the training window

Gate      : if LightGBM does not beat seasonal naive on daily-total MAE
            (pooled over all folds), the script prints STOP and exits 1
            without saving anything.

Run from the repo root:  python -m db.train
Connection comes from $V4W_DATABASE_URL (same pattern as db/load_calls.py).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

try:
    import psycopg
except ImportError:
    sys.exit("psycopg is required:  pip install 'psycopg[binary]'")

from db.features import build_features, load_holidays, load_hourly  # noqa: E402

TARGET = "offered"
NON_FEATURE_COLUMNS = ("hour_start", TARGET)

FIRST_TEST_MONTH = "2024-01-01"
LAST_TEST_MONTH = "2025-12-01"
NAIVE_WINDOW_DAYS = 8 * 7

# Fixed. Not tuned: the gate below decides whether this is good enough.
LGBM_PARAMS = {
    "objective": "poisson",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "seed": 42,
    "verbose": -1,
}
NUM_BOOST_ROUNDS = 500

# Anchored on this file, not the process cwd, so `python -m db.train` and the
# API find the same models/ from any working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "volume_lgbm.txt"
METADATA_PATH = MODEL_DIR / "metadata.json"

MODEL_NAMES = ("lgbm_poisson", "seasonal_naive", "overall_mean")


# =============================================================================
# Data
# =============================================================================

def complete_hourly_grid(hourly: pd.DataFrame) -> pd.DataFrame:
    """Every hour from the first day to the last day of the data, with
    `offered` = 0 where calls_hourly has no row (no calls that hour)."""
    first = hourly["hour_start"].min().normalize()
    last = hourly["hour_start"].max().normalize() + pd.Timedelta(hours=23)
    grid = pd.date_range(first, last, freq="h", name="hour_start")

    offered = (
        hourly.set_index("hour_start")[TARGET]
        .reindex(grid, fill_value=0)
        .astype("int64")
    )
    return pd.DataFrame({"hour_start": grid, TARGET: offered.to_numpy()})


def feature_columns(featured: pd.DataFrame) -> list[str]:
    return [c for c in featured.columns if c not in NON_FEATURE_COLUMNS]


# =============================================================================
# Models
# =============================================================================

def fit_lgbm(train: pd.DataFrame, features: list[str]) -> lgb.Booster:
    dataset = lgb.Dataset(train[features], label=train[TARGET])
    return lgb.train(LGBM_PARAMS, dataset, num_boost_round=NUM_BOOST_ROUNDS)


def predict_lgbm(booster: lgb.Booster, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    return booster.predict(test[features])


def predict_seasonal_naive(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Median `offered` per (dow, hour) over the trailing 8 weeks of train."""
    window_start = test["hour_start"].min() - pd.Timedelta(days=NAIVE_WINDOW_DAYS)
    window = train[train["hour_start"] >= window_start]
    medians = window.groupby(["dow", "hour"])[TARGET].median()

    keys = pd.MultiIndex.from_frame(test[["dow", "hour"]])
    predicted = medians.reindex(keys).to_numpy()
    if np.isnan(predicted).any():
        raise ValueError("seasonal naive: some (dow, hour) cells have no trailing history")
    return predicted


def predict_overall_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.full(len(test), train[TARGET].mean())


# =============================================================================
# Scoring
# =============================================================================

def score(hour_start: pd.Series, actual: np.ndarray, predicted: np.ndarray) -> dict:
    """MAE, RMSE, MAPE (hours with offered > 0), daily-total MAE."""
    error = predicted - actual
    positive = actual > 0

    daily = (
        pd.DataFrame({"day": hour_start.dt.normalize().to_numpy(),
                      "actual": actual, "predicted": predicted})
        .groupby("day")[["actual", "predicted"]].sum()
    )

    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape_pct": float(np.mean(np.abs(error[positive]) / actual[positive]) * 100),
        "daily_mae": float(np.mean(np.abs(daily["predicted"] - daily["actual"]))),
    }


# =============================================================================
# Backtest
# =============================================================================

def split_fold(featured: pd.DataFrame, month_start: pd.Timestamp):
    month_end = month_start + pd.offsets.MonthBegin(1)
    is_test = (featured["hour_start"] >= month_start) & (featured["hour_start"] < month_end)
    train = featured[featured["hour_start"] < month_start]
    test = featured[is_test]
    if train.empty or test.empty:
        raise ValueError(f"fold {month_start:%Y-%m}: empty train or test window")
    return train, test


def predict_fold(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> dict[str, np.ndarray]:
    booster = fit_lgbm(train, features)
    return {
        "lgbm_poisson": predict_lgbm(booster, test, features),
        "seasonal_naive": predict_seasonal_naive(train, test),
        "overall_mean": predict_overall_mean(train, test),
    }


def run_backtest(featured: pd.DataFrame, features: list[str]):
    """Returns (per_fold DataFrame, overall DataFrame). Overall metrics are
    pooled over every test hour of every fold, not averaged across folds."""
    months = pd.date_range(FIRST_TEST_MONTH, LAST_TEST_MONTH, freq="MS")
    fold_rows = []
    pooled = {name: [] for name in MODEL_NAMES}
    pooled_hours = []
    pooled_actual = []

    for month_start in months:
        train, test = split_fold(featured, month_start)
        predictions = predict_fold(train, test, features)
        actual = test[TARGET].to_numpy()

        for name in MODEL_NAMES:
            metrics = score(test["hour_start"], actual, predictions[name])
            fold_rows.append({"fold": f"{month_start:%Y-%m}", "model": name, **metrics})
            pooled[name].append(predictions[name])
        pooled_hours.append(test["hour_start"])
        pooled_actual.append(actual)
        print(f"fold {month_start:%Y-%m}  train={len(train):,}  test={len(test):,}", file=sys.stderr)

    hours = pd.concat(pooled_hours, ignore_index=True)
    actual = np.concatenate(pooled_actual)
    overall_rows = [
        {"model": name, **score(hours, actual, np.concatenate(pooled[name]))}
        for name in MODEL_NAMES
    ]
    return pd.DataFrame(fold_rows), pd.DataFrame(overall_rows)


# =============================================================================
# Reporting
# =============================================================================

def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = []
    for row in df.itertuples(index=False):
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body])


def passes_gate(overall: pd.DataFrame) -> bool:
    by_model = overall.set_index("model")["daily_mae"]
    return by_model["lgbm_poisson"] < by_model["seasonal_naive"]


# =============================================================================
# Persistence
# =============================================================================

def save_model(booster: lgb.Booster, featured: pd.DataFrame, features: list[str],
               per_fold: pd.DataFrame, overall: pd.DataFrame) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_PATH))

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": TARGET,
        "train_start": featured["hour_start"].min().isoformat(),
        "train_end": featured["hour_start"].max().isoformat(),
        "train_rows": int(len(featured)),
        "features": features,
        "lgbm_params": LGBM_PARAMS,
        "num_boost_rounds": NUM_BOOST_ROUNDS,
        "backtest": {
            "first_test_month": FIRST_TEST_MONTH,
            "last_test_month": LAST_TEST_MONTH,
            "overall": overall.to_dict(orient="records"),
            "per_fold": per_fold.to_dict(orient="records"),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    dsn = os.environ.get("V4W_DATABASE_URL")
    if not dsn:
        sys.exit("no connection string. set V4W_DATABASE_URL")

    with psycopg.connect(dsn) as conn:
        hourly = load_hourly(conn)
        holidays = load_holidays(conn)

    grid = complete_hourly_grid(hourly)
    featured = build_features(grid, holidays)
    features = feature_columns(featured)
    print(f"grid rows {len(grid):,}  (view rows {len(hourly):,}, "
          f"zero-filled {len(grid) - len(hourly):,})  features {len(features)}")

    per_fold, overall = run_backtest(featured, features)

    print("\n## Per fold\n")
    print(markdown_table(per_fold))
    print("\n## Overall (pooled over all test hours)\n")
    print(markdown_table(overall))

    if not passes_gate(overall):
        print("\nSTOP: LightGBM did not beat seasonal naive on daily-total MAE. "
              "Nothing saved.")
        return 1

    booster = fit_lgbm(featured, features)
    save_model(booster, featured, features, per_fold, overall)
    print(f"\nsaved {MODEL_PATH} and {METADATA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
