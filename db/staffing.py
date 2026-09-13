"""Turn the hourly forecast into staffing numbers.

    staff_for_hour(predicted_offered, aht_sec, efficiency, target_pct, target_sec)
        -> dict: offered_load_erlangs, agents_required, scheduled_needed,
                 service_level_at_scheduled, capped

    staff_for_range(conn, start_date, end_date, efficiency, target_pct)
        -> hourly DataFrame: forecast columns + the dict above per hour

    rollup_shifts(hourly)
        -> one row per (shift_date, shift): Day / Evening / Overnight

Shift staffing is the MAX of the hourly requirement inside the shift, not
the mean. You staff for the peak or you miss the SLA at the peak.

The client caps a shift at 7 people. `capped` flags any requirement above
that. It is surfaced, never clamped.

Shifts overlap by 30 minutes (Day 08:00-16:30, Evening 16:00-00:30,
Overnight 00:00-08:30). For the staffing max, the boundary hour counts
toward both shifts. For predicted call totals, each hour counts once, in
the shift that starts it, so shift totals add up to the day total.
"""

from __future__ import annotations

import math

import pandas as pd

from db.erlang import erlang_c, required_agents
from db.forecast import forecast_range

DEFAULT_EFFICIENCY = 0.65
DEFAULT_TARGET_PCT = 0.80
DEFAULT_TARGET_SEC = 30
MAX_STAFF_PER_SHIFT = 7
SECONDS_PER_HOUR = 3600

# core_hours: hours whose start falls inside the shift (counted once).
# boundary_hour: the hour containing the 30-minute overlap at the shift's
# end; it joins the staffing max but not the call total.
# boundary_day_offset: Evening runs past midnight, so hour 0 of day D
# belongs to the Evening shift dated D-1.
SHIFTS = (
    {"shift": "Overnight", "core_hours": range(0, 8), "boundary_hour": 8, "boundary_day_offset": 0},
    {"shift": "Day", "core_hours": range(8, 16), "boundary_hour": 16, "boundary_day_offset": 0},
    {"shift": "Evening", "core_hours": range(16, 24), "boundary_hour": 0, "boundary_day_offset": -1},
)
SHIFT_ORDER = tuple(s["shift"] for s in SHIFTS)


# =============================================================================
# One hour
# =============================================================================

def _service_level(agents: int, load: float, aht_sec: float, target_sec: float) -> float:
    """Share of calls answered within target_sec with `agents` on inbound."""
    if agents <= 0:
        return 0.0
    wait_prob = erlang_c(agents, load)
    return 1.0 - wait_prob * math.exp(-(agents - load) * target_sec / aht_sec)


def staff_for_hour(
    predicted_offered: float,
    aht_sec: float,
    efficiency: float = DEFAULT_EFFICIENCY,
    target_pct: float = DEFAULT_TARGET_PCT,
    target_sec: float = DEFAULT_TARGET_SEC,
) -> dict:
    """Agents needed on inbound, people to schedule to get that many after
    efficiency loss, and the service level those scheduled people deliver."""
    if not 0 < efficiency <= 1:
        raise ValueError(f"efficiency must be in (0, 1], got {efficiency}")
    if predicted_offered < 0:
        raise ValueError(f"predicted_offered must be >= 0, got {predicted_offered}")

    load = predicted_offered * aht_sec / SECONDS_PER_HOUR
    agents = required_agents(predicted_offered, aht_sec, target_pct, target_sec)
    scheduled = math.ceil(agents / efficiency)

    # Of `scheduled` people, only efficiency * scheduled are on inbound.
    effective_agents = math.floor(scheduled * efficiency)
    service_level = _service_level(effective_agents, load, aht_sec, target_sec)

    return {
        "offered_load_erlangs": load,
        "agents_required": agents,
        "scheduled_needed": scheduled,
        "service_level_at_scheduled": service_level,
        "capped": scheduled > MAX_STAFF_PER_SHIFT,
    }


# =============================================================================
# A range of hours
# =============================================================================

def staff_for_range(
    conn,
    start_date,
    end_date,
    efficiency: float = DEFAULT_EFFICIENCY,
    target_pct: float = DEFAULT_TARGET_PCT,
) -> pd.DataFrame:
    """Forecast every hour in the range and attach staffing numbers."""
    forecast = forecast_range(conn, start_date, end_date)
    staffing = pd.DataFrame([
        staff_for_hour(row.predicted_offered, row.predicted_aht_sec, efficiency, target_pct)
        for row in forecast.itertuples(index=False)
    ])
    return pd.concat([forecast, staffing], axis=1)


# =============================================================================
# Shift roll-up
# =============================================================================

def _shift_memberships(hourly: pd.DataFrame) -> pd.DataFrame:
    """Long frame: one row per (hour, shift it belongs to), with is_core."""
    day = hourly["hour_start"].dt.normalize()
    hour = hourly["hour_start"].dt.hour
    pieces = []
    for spec in SHIFTS:
        core_mask = hour.isin(spec["core_hours"])
        core = hourly[core_mask].assign(
            shift=spec["shift"], shift_date=day[core_mask], is_core=True,
        )
        boundary_mask = hour == spec["boundary_hour"]
        boundary = hourly[boundary_mask].assign(
            shift=spec["shift"],
            shift_date=day[boundary_mask] + pd.Timedelta(days=spec["boundary_day_offset"]),
            is_core=False,
        )
        pieces.extend([core, boundary])
    return pd.concat(pieces, ignore_index=True)


def rollup_shifts(hourly: pd.DataFrame) -> pd.DataFrame:
    """One row per (shift_date, shift). Staffing = max over the shift's
    hours including the overlap hour; calls = sum over core hours only.
    Shifts with no core hours in the range (edge of the window) are dropped."""
    long = _shift_memberships(hourly)
    long["core_calls"] = long["predicted_offered"].where(long["is_core"], 0.0)

    grouped = long.groupby(["shift_date", "shift"]).agg(
        predicted_calls=("core_calls", "sum"),
        agents_required=("agents_required", "max"),
        staff_needed=("scheduled_needed", "max"),
        core_hours=("is_core", "sum"),
    ).reset_index()

    complete = grouped[grouped["core_hours"] > 0].drop(columns="core_hours")
    complete = complete.assign(capped=complete["staff_needed"] > MAX_STAFF_PER_SHIFT)
    complete["shift"] = pd.Categorical(complete["shift"], categories=SHIFT_ORDER, ordered=True)
    return complete.sort_values(["shift_date", "shift"]).reset_index(drop=True)
