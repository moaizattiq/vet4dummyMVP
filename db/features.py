"""Feature engineering for the Vets4Warriors hourly call-volume model.

This is the ONLY place feature logic lives. Training and prediction both
call build_features(), so the two can never drift apart.

    build_features(df, holidays) -> df with feature columns added
    load_hourly(conn)            -> the calls_hourly view, all columns
    load_holidays(conn)          -> the holidays table

Deliberate omissions:
  * No lag features. The model forecasts 30 days out; lagged actuals would
    not exist at prediction time.
  * Holiday rows are kept, not dropped. The model learns them from the
    is_holiday / multiplier columns.

Connection comes from $V4W_DATABASE_URL (same pattern as db/load_calls.py).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

try:
    import psycopg
except ImportError:
    sys.exit("psycopg is required:  pip install 'psycopg[binary]'")


# days_to_nearest_holiday is clipped to this window. Beyond two weeks the
# distance carries no signal and would only stretch the feature's range.
HOLIDAY_WINDOW_DAYS = 14

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
SATURDAY_ISODOW = 6

REQUIRED_HOLIDAY_COLUMNS = (
    "holiday_date",
    "volume_multiplier",
    "evening_multiplier",
)


# =============================================================================
# Loading
# =============================================================================

def _read_table(conn, sql: str) -> pd.DataFrame:
    """Run one SELECT and return a DataFrame with the cursor's column names.

    Uses a plain cursor rather than pd.read_sql so psycopg3 connections
    work without SQLAlchemy.
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        columns = [d.name for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def load_hourly(conn) -> pd.DataFrame:
    """Every column of the calls_hourly view, ordered by hour_start.

    offered_load_erlangs arrives as Decimal from Postgres; cast to float so
    downstream math does not trip on it.
    """
    df = _read_table(conn, "SELECT * FROM calls_hourly ORDER BY hour_start")
    df["hour_start"] = pd.to_datetime(df["hour_start"])
    if "offered_load_erlangs" in df.columns:
        df["offered_load_erlangs"] = df["offered_load_erlangs"].astype(float)
    return df


def load_holidays(conn) -> pd.DataFrame:
    """Every row of the holidays table, ordered by date."""
    df = _read_table(conn, "SELECT * FROM holidays ORDER BY holiday_date")
    df["holiday_date"] = pd.to_datetime(df["holiday_date"])
    return df


# =============================================================================
# Features
# =============================================================================

def _validate_inputs(df: pd.DataFrame, holidays: pd.DataFrame) -> None:
    if "hour_start" not in df.columns:
        raise ValueError("df must have an `hour_start` column")
    if not pd.api.types.is_datetime64_any_dtype(df["hour_start"]):
        raise ValueError("df.hour_start must be a datetime column")
    if df["hour_start"].isna().any():
        raise ValueError("df.hour_start contains nulls")
    missing = [c for c in REQUIRED_HOLIDAY_COLUMNS if c not in holidays.columns]
    if missing:
        raise ValueError(f"holidays is missing columns: {missing}")
    if len(holidays) == 0:
        raise ValueError(
            "holidays is empty; days_to_nearest_holiday is undefined"
        )


def _calendar_features(hour_start: pd.Series, epoch: pd.Timestamp) -> pd.DataFrame:
    """dow, hour, month, week_of_year, trend index, cyclic encodings, weekend."""
    dow = hour_start.dt.isocalendar().day.astype("int64")       # 1=Mon..7=Sun
    hour = hour_start.dt.hour.astype("int64")

    return pd.DataFrame({
        "dow": dow,
        "hour": hour,
        "month": hour_start.dt.month.astype("int64"),
        "week_of_year": hour_start.dt.isocalendar().week.astype("int64"),
        "days_since_epoch": (hour_start.dt.normalize() - epoch).dt.days.astype("int64"),
        "hour_sin": np.sin(2 * np.pi * hour / HOURS_PER_DAY),
        "hour_cos": np.cos(2 * np.pi * hour / HOURS_PER_DAY),
        "dow_sin": np.sin(2 * np.pi * (dow - 1) / DAYS_PER_WEEK),
        "dow_cos": np.cos(2 * np.pi * (dow - 1) / DAYS_PER_WEEK),
        "is_weekend": (dow >= SATURDAY_ISODOW).astype("int64"),
    }, index=hour_start.index)


def _days_to_nearest_holiday(dates: pd.Series, holiday_dates: np.ndarray) -> pd.Series:
    """Signed days from each date to the nearest holiday.

    Positive = holiday is ahead, negative = holiday has passed, 0 = today.
    Ties go to the upcoming holiday. Clipped to +/- HOLIDAY_WINDOW_DAYS.
    """
    day_values = dates.to_numpy(dtype="datetime64[D]").astype("int64")
    holiday_values = np.sort(holiday_dates.astype("datetime64[D]").astype("int64"))

    next_idx = np.searchsorted(holiday_values, day_values, side="left")
    prev_idx = next_idx - 1

    has_next = next_idx < len(holiday_values)
    has_prev = prev_idx >= 0

    to_next = np.where(
        has_next,
        holiday_values[np.minimum(next_idx, len(holiday_values) - 1)] - day_values,
        np.iinfo("int64").max,
    )
    to_prev = np.where(
        has_prev,
        holiday_values[np.maximum(prev_idx, 0)] - day_values,
        np.iinfo("int64").min,
    )

    pick_next = np.abs(to_next) <= np.abs(to_prev)
    signed = np.where(pick_next, to_next, to_prev)
    clipped = np.clip(signed, -HOLIDAY_WINDOW_DAYS, HOLIDAY_WINDOW_DAYS)
    return pd.Series(clipped.astype("int64"), index=dates.index)


def _holiday_features(hour_start: pd.Series, holidays: pd.DataFrame) -> pd.DataFrame:
    """is_holiday, days_to_nearest_holiday, and the two multipliers.

    Multipliers are 0.0 when the row is not a holiday. A holiday whose
    multiplier is NULL in the table (not yet measured) also gets 0.0 so the
    output has no nulls; is_holiday still marks it, so the model can learn
    the day from that flag alone.
    """
    calendar = holidays.copy()
    calendar["holiday_date"] = pd.to_datetime(calendar["holiday_date"]).dt.normalize()
    calendar = calendar.drop_duplicates(subset="holiday_date")

    row_dates = hour_start.dt.normalize()
    lookup = calendar.set_index("holiday_date")

    volume = row_dates.map(lookup["volume_multiplier"]).astype(float).fillna(0.0)
    evening = row_dates.map(lookup["evening_multiplier"]).astype(float).fillna(0.0)
    is_holiday = row_dates.isin(lookup.index).astype("int64")

    return pd.DataFrame({
        "is_holiday": is_holiday,
        "days_to_nearest_holiday": _days_to_nearest_holiday(
            row_dates, calendar["holiday_date"].to_numpy()
        ),
        "holiday_volume_multiplier": volume,
        "holiday_evening_multiplier": evening,
    }, index=hour_start.index)


def build_features(
    df: pd.DataFrame,
    holidays: pd.DataFrame,
    epoch: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Return a NEW DataFrame: every input column plus the feature columns.

    Input `df` needs a datetime column `hour_start`. Input `holidays` needs
    holiday_date, volume_multiplier, evening_multiplier. Neither input is
    mutated. Row count and order are preserved.

    `epoch` anchors days_since_epoch. Training leaves it None (= the data's
    first day, recorded in models/metadata.json as train_start). Prediction
    MUST pass that recorded train_start, otherwise the trend index restarts
    at 0 and the model forecasts the past.
    """
    _validate_inputs(df, holidays)

    hour_start = pd.to_datetime(df["hour_start"])
    anchor = hour_start.min() if epoch is None else pd.Timestamp(epoch)
    calendar = _calendar_features(hour_start, anchor.normalize())
    holiday = _holiday_features(hour_start, holidays)

    return pd.concat([df, calendar, holiday], axis=1)
