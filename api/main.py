"""FastAPI surface for the Vets4Warriors staffing forecast.

    GET /api/forecast?start=&end=&efficiency=&target_pct=   hourly rows
    GET /api/staffing?start=&end=&efficiency=&target_pct=   daily + shift roll-up
    GET /api/model                                          models/metadata.json
    GET /                                                   web/index.html (static)

Every request opens its own database connection from $V4W_DATABASE_URL and
runs the same db.staffing code the CLI uses. No numbers are computed here.

Run from the repo root:  uvicorn api.main:app --reload
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

load_dotenv()

try:
    import psycopg
except ImportError:
    sys.exit("psycopg is required:  pip install 'psycopg[binary]'")

from db.features import load_holidays  # noqa: E402
from db.forecast import holiday_coverage  # noqa: E402
from db.staffing import (  # noqa: E402
    DEFAULT_EFFICIENCY,
    DEFAULT_TARGET_PCT,
    SHIFT_ORDER,
    rollup_shifts,
    staff_for_range,
)
from db.train import METADATA_PATH  # noqa: E402

log = logging.getLogger("v4w.api")

DEFAULT_HORIZON_DAYS = 30
MAX_RANGE_DAYS = 366
REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"

app = FastAPI(title="Vets4Warriors staffing forecast", version="0.1.0")


# =============================================================================
# Helpers
# =============================================================================

def _default_range() -> tuple[date, date]:
    start = date.today()
    return start, start + timedelta(days=DEFAULT_HORIZON_DAYS - 1)


def _validate_range(start: date | None, end: date | None) -> tuple[date, date]:
    default_start, default_end = _default_range()
    start = start or default_start
    end = end or default_end
    if end < start:
        raise HTTPException(400, f"end {end} is before start {start}")
    span = (end - start).days + 1
    if span > MAX_RANGE_DAYS:
        raise HTTPException(400, f"range is {span} days; maximum is {MAX_RANGE_DAYS}")
    return start, end


def _connection():
    dsn = os.environ.get("V4W_DATABASE_URL")
    if not dsn:
        raise HTTPException(503, "V4W_DATABASE_URL is not set")
    try:
        return psycopg.connect(dsn)
    except psycopg.Error as exc:
        log.exception("database connection failed")
        raise HTTPException(503, f"database connection failed: {exc}") from exc


def _hourly_staffing(start: date, end: date, efficiency: float, target_pct: float) -> tuple[pd.DataFrame, dict]:
    """Hourly staffing frame plus the holiday-coverage status for this range."""
    try:
        with _connection() as conn:
            hourly = staff_for_range(conn, start, end, efficiency, target_pct)
            coverage = holiday_coverage(load_holidays(conn), end)
            return hourly, coverage
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except psycopg.Error as exc:
        log.exception("query failed")
        raise HTTPException(500, f"query failed: {exc}") from exc


def _records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe rows: timestamps to ISO strings, numpy scalars to Python."""
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return json.loads(out.to_json(orient="records"))


def _days_with_shifts(shifts: pd.DataFrame) -> list[dict]:
    """One entry per date, shifts nested by name, plus the daily call total."""
    days = []
    for shift_date, group in shifts.groupby("shift_date", sort=True):
        by_name = {
            row.shift: {
                "predicted_calls": round(float(row.predicted_calls), 2),
                "agents_required": int(row.agents_required),
                "staff_needed": int(row.staff_needed),
                "capped": bool(row.capped),
            }
            for row in group.itertuples(index=False)
        }
        days.append({
            "date": shift_date.strftime("%Y-%m-%d"),
            "predicted_calls": round(float(group["predicted_calls"].sum()), 2),
            "shifts": {name: by_name.get(name) for name in SHIFT_ORDER},
        })
    return days


# =============================================================================
# Routes
# =============================================================================

@app.get("/api/forecast")
def api_forecast(
    start: date | None = None,
    end: date | None = None,
    efficiency: float = Query(DEFAULT_EFFICIENCY, gt=0, le=1),
    target_pct: float = Query(DEFAULT_TARGET_PCT, gt=0, lt=1),
):
    start, end = _validate_range(start, end)
    hourly, coverage = _hourly_staffing(start, end, efficiency, target_pct)
    return {
        "params": {"start": str(start), "end": str(end),
                   "efficiency": efficiency, "target_pct": target_pct},
        "holiday_coverage": coverage,
        "rows": _records(hourly),
    }


@app.get("/api/staffing")
def api_staffing(
    start: date | None = None,
    end: date | None = None,
    efficiency: float = Query(DEFAULT_EFFICIENCY, gt=0, le=1),
    target_pct: float = Query(DEFAULT_TARGET_PCT, gt=0, lt=1),
):
    start, end = _validate_range(start, end)
    hourly, coverage = _hourly_staffing(start, end, efficiency, target_pct)
    shifts = rollup_shifts(hourly)
    return {
        "params": {"start": str(start), "end": str(end),
                   "efficiency": efficiency, "target_pct": target_pct},
        "holiday_coverage": coverage,
        "days": _days_with_shifts(shifts),
    }


@app.get("/api/model")
def api_model():
    if not METADATA_PATH.exists():
        raise HTTPException(404, f"{METADATA_PATH} not found; run `make train`")
    try:
        return json.loads(METADATA_PATH.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"{METADATA_PATH} is not valid JSON: {exc}") from exc


# Static site last so /api/* routes win.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
