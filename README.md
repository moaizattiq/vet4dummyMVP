# Vets4Warriors staffing forecast (demo)

Forecasts inbound call volume per hour for the Vets4Warriors peer-support
line and turns it into "how many people to schedule" per shift. Built for a
customer demo. Not the production repo.

## What it does, in plain English

The model looks at five years of hourly call counts and learns the shape of
demand: which hour of the day, which day of the week, which month, holidays,
and a slow trend. Given a future date range it predicts calls per hour. A
second, simpler estimate gives the typical call length for that hour of the
week. Erlang C (standard call-center queueing math) then says how many people
must be on the phones so that a target share of callers (default 80%) is
answered within 30 seconds. Dividing by staff efficiency (default 65%, the
share of a shift someone is actually free for inbound) gives the number to
put on the schedule. Each shift is staffed for its busiest hour, not its
average, because you either cover the peak or you miss the target at the
peak.

**Error bars.** In a rolling backtest (train on everything before a month,
predict that whole month with no access to it, repeat for 24 months) the
model's daily-total error was about **9.3 calls per day**, against 10.3 for a
"same hour, same weekday, last 8 weeks" baseline and 12.7 for a flat average.
Hourly error is about 1.26 calls per hour on a mean of roughly 1.6, which is
what you get with small counts. In staffing terms the honest reading is
"plus or minus one person per shift". The exact numbers are in
`models/metadata.json` and shown in the banner at the top of the dashboard.

Staffing numbers above the client's cap of 7 per shift are flagged in red,
not silently clamped.

## Layout

```
db/erlang.py     Erlang C: erlang_c(), required_agents()      (pure math)
db/features.py   the ONLY place feature logic lives            (+ DB loaders)
db/aht.py        handle time per (weekday, hour), recency-weighted
db/train.py      LightGBM Poisson model, rolling backtest, saves models/
db/forecast.py   forecast_range(): hourly calls + AHT for a date range
db/staffing.py   Erlang -> agents -> scheduled staff, shift roll-up
api/main.py      FastAPI: /api/forecast, /api/staffing, /api/model, serves web/
web/index.html   one static file, vanilla JS, inline SVG chart
models/          volume_lgbm.txt + metadata.json (written by `make train`)
tests/           pytest; the feature tests hit the real database
```

## Setup

Python 3.12. No package manifest yet; install directly:

```
python3 -m venv .venv && source .venv/bin/activate
pip install "psycopg[binary]" python-dotenv pandas numpy lightgbm fastapi uvicorn pytest
```

Create `.env` in the repo root. It is gitignored and must never be committed:

```
cp .env.example .env
```

Then paste the real connection string from the Supabase project
(Connect button, "Session pooler" tab):
https://supabase.com/dashboard/project/okqdxhvbcdhzriftwute

```
V4W_DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DBNAME
```

Then:

```
make db-check     # prints row counts from calls, holidays, calls_hourly
make train        # ~2-3 min: 24-fold backtest, then writes models/
make serve        # http://127.0.0.1:8000
```

`make` targets use `python3`; pass `PYTHON=.venv/bin/python` to override.

## Environment variables

| Name | Required | Meaning |
|---|---|---|
| `V4W_DATABASE_URL` | yes | Postgres connection string. Only data source. Loaded via `python-dotenv` from `.env` or the shell. |
| `HOST`, `PORT` | no | `make serve` bind address, default `127.0.0.1:8000`. |

## Retraining

New call data lands in the `calls` table (see the loader in the main repo).
Then:

```
make train
```

This reruns the full rolling-origin backtest against seasonal naive and a flat
mean. If LightGBM does not beat seasonal naive on daily-total MAE the script
prints `STOP` and exits non-zero **without** overwriting `models/`. Read the
per-fold table it prints before trusting a new model.

The API reads `models/metadata.json` on every `/api/model` call and reloads
`models/volume_lgbm.txt` on every forecast, so a retrain takes effect without
restarting the server.

## API

```
GET /api/forecast?start=YYYY-MM-DD&end=YYYY-MM-DD&efficiency=0.65&target_pct=0.80
    one row per hour: hour_start, predicted_offered, predicted_aht_sec,
    offered_load_erlangs, agents_required, scheduled_needed,
    service_level_at_scheduled, capped

GET /api/staffing?start=&end=&efficiency=&target_pct=
    one entry per day with Day / Evening / Overnight nested:
    predicted_calls, agents_required, staff_needed, capped

GET /api/model
    models/metadata.json verbatim
```

`start`/`end` default to today and today + 29 days. Maximum range 366 days.

## Known gaps

- The `holidays` table must contain future holidays for them to affect the
  forecast. The forecaster warns when the requested range runs past the last
  known holiday.
- No caching: every request re-reads the model file and recomputes handle
  times from the database. Fine for a demo, a few seconds per request.
- Shift hours are fixed in `db/staffing.py` (Day 08:00-16:30, Evening
  16:00-00:30, Overnight 00:00-08:30). The 30-minute overlaps count toward
  the staffing peak of both adjacent shifts.
