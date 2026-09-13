"""Hourly call-volume and handle-time forecast for a future date range.

    forecast_range(conn, start_date, end_date) -> DataFrame
        hour_start, predicted_offered, predicted_aht_sec  (one row per hour)

Loads the trained LightGBM booster from models/volume_lgbm.txt and its
metadata (train_start, feature list). Features come from
db.features.build_features, the same function training used, anchored to
the training epoch so the trend index lines up. AHT comes from db.aht.

No fallback forecaster: a missing model file raises.
Connection comes from $V4W_DATABASE_URL (same pattern as db/load_calls.py).
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import date

import lightgbm as lgb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

try:
    import psycopg  # noqa: F401  (callers pass an open connection)
except ImportError:
    sys.exit("psycopg is required:  pip install 'psycopg[binary]'")

from db.aht import estimate_aht  # noqa: E402
from db.features import build_features, load_holidays  # noqa: E402
from db.train import METADATA_PATH, MODEL_PATH  # noqa: E402

OUTPUT_COLUMNS = ("hour_start", "predicted_offered", "predicted_aht_sec")
LAST_HOUR_OF_DAY = 23


def _load_model() -> tuple[lgb.Booster, dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run `python -m db.train` first."
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"{METADATA_PATH} not found. Run `python -m db.train` first."
        )
    metadata = json.loads(METADATA_PATH.read_text())
    for key in ("train_start", "features"):
        if key not in metadata:
            raise ValueError(f"{METADATA_PATH} is missing `{key}`")
    return lgb.Booster(model_file=str(MODEL_PATH)), metadata


def _hourly_grid(start_date, end_date) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError(f"end_date {end.date()} is before start_date {start.date()}")
    hours = pd.date_range(start, end + pd.Timedelta(hours=LAST_HOUR_OF_DAY), freq="h")
    return pd.DataFrame({"hour_start": hours})


def _warn_if_beyond_holiday_calendar(holidays: pd.DataFrame, end_date) -> None:
    last_known = holidays["holiday_date"].max()
    if pd.Timestamp(end_date) > last_known:
        warnings.warn(
            f"holidays table ends {last_known.date()}; holidays after that "
            f"are not flagged in the forecast. Load future holidays first.",
            stacklevel=3,
        )


def forecast_range(conn, start_date: date | str, end_date: date | str) -> pd.DataFrame:
    """Predicted offered calls and AHT for every hour from start_date 00:00
    through end_date 23:00 (both dates inclusive)."""
    booster, metadata = _load_model()
    holidays = load_holidays(conn)
    _warn_if_beyond_holiday_calendar(holidays, end_date)

    grid = _hourly_grid(start_date, end_date)
    featured = build_features(grid, holidays, epoch=metadata["train_start"])

    features = metadata["features"]
    missing = [f for f in features if f not in featured.columns]
    if missing:
        raise ValueError(f"build_features did not produce trained features: {missing}")

    predicted_offered = booster.predict(featured[features])

    aht_by_cell = estimate_aht(conn)
    predicted_aht = [
        aht_by_cell[(int(dow), int(hour))]
        for dow, hour in zip(featured["dow"], featured["hour"])
    ]

    return pd.DataFrame({
        "hour_start": featured["hour_start"].to_numpy(),
        "predicted_offered": predicted_offered,
        "predicted_aht_sec": predicted_aht,
    })
