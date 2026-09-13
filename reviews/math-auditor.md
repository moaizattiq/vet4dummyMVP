# Math audit: Erlang C, staffing, shift rollup, Poisson objective, AHT

**Verdict: the Erlang C implementation, service-level formula, efficiency arithmetic and shift rollup are correct. One test asserts a wrong expected value (6; the correct answer is 8). The `ceil(load)+1` search start over-states hourly `agents_required` by one for ~10% of hours on the real forecast surface, but the shift-level output is unchanged. Poisson is adequate for this data.**

Scope: read-only review of `db/erlang.py`, `db/staffing.py`, `db/train.py`, `db/aht.py`, `tests/test_erlang.py`, `models/metadata.json`. Every number below comes from code run during this audit (scripts in the session scratchpad) or from the cited source. Interpreter: `/Users/moaizattiq/v4w-staffing/.venv/bin/python` (3.12.13, pandas 3.0.5, numpy 2.5.3, lightgbm 4.7.0, scipy 1.18.1). No repo file other than this report was written. `models/` was not touched (verified by `ls -la models/` timestamps 14:56 before and after).

---

## Task 1. Erlang C correctness

Independent reference implementations: (a) closed form in log-space with `math.lgamma`, (b) exact rational arithmetic with `fractions.Fraction` and `math.factorial`. Compared against `erlang_c()` over the requested grid.

```
=== Task 1: erlang_c vs closed-form (lgamma) and exact Fraction ===
   N      A           erlang_c             lgamma              exact |diff exact|
   1    0.5  0.500000000000000  0.500000000000000  0.500000000000000    0.000e+00
   2    1.5  0.642857142857143  0.642857142857143  0.642857142857143    1.110e-16
   5      4  0.554112554112554  0.554112554112554  0.554112554112554    0.000e+00
   8      5  0.167266506661757  0.167266506661756  0.167266506661757    0.000e+00
  10      8  0.409180150796443  0.409180150796444  0.409180150796444    5.551e-17
  20     15  0.160429387416924  0.160429387416924  0.160429387416924    5.551e-17
  50     45  0.363864467206988  0.363864467206983  0.363864467206988    5.551e-17
 100     90  0.216940480906366  0.216940480906370  0.216940480906366    1.388e-16
 200    190  0.365263856562546  0.365263856562546  0.365263856562546    5.551e-17
max abs diff over grid (vs lgamma and exact): 5.551e-15
```

**Max absolute difference: 5.6e-15** (the lgamma path is the less precise one; vs exact rationals the max is 1.4e-16).

Published reference values. Source: Call Centre Helper, "Erlang C Formula – Made Simple With an Easy Worked Example", https://www.callcentrehelper.com/erlang-c-formula-example-121281.htm (200 calls/hour, AHT 180 s, A = 10 erlangs, 20 s target). Fetched during this audit.

```
Call Centre Helper example: 200 calls/hr, AHT 180s, A=10 erlangs, target 20s
  N=11: erlang_c=68.21% (ref 68.2%) diff=0.012pp | SL20=39.0% (ref 39.0%) diff=0.04pp
  N=12: erlang_c=44.94% (ref 44.94%) diff=0.001pp | SL20=64.0% (ref 64.0%) diff=0.02pp
  N=13: erlang_c=28.53% (ref 28.53%) diff=0.003pp | SL20=79.6% (ref 79.6%) diff=0.04pp
  N=14: erlang_c=17.41% (ref 17.41%) diff=0.003pp | SL20=88.8% (ref 88.8%) diff=0.04pp
  N=14 ASA = C*AHT/(N-A) = 7.8s (ref 7.8s); occupancy A/N = 71.4% (ref 71.4%)
```

All four P(wait) values, all four service levels, the ASA and the occupancy match the published example to the precision printed there. Two further analytic references (hand-derived closed forms, exact fractions) also match:

```
A=1,N=2   closed form 2B/(2-A(1-B)), B=A^2/2/(1+A+A^2/2)=1/5 -> C=0.4/(2-0.8)=1/3: erlang_c=0.333333 ref=0.333333 diff=5.55e-17
A=2,N=3   closed form: B(3,2)=4/19, C=3B/(3-2(1-B)) = (12/19)/(3-30/19)=(12/19)/(27/19)=4/9: erlang_c=0.444444 ref=0.444444 diff=1.11e-16
```

(The two values given in the task brief, 0.4494 for A=10/N=12 and 0.1673 for A=5/N=8, are the 4-decimal roundings of 0.449388 and 0.167267 respectively; the first is the Call Centre Helper value above.)

The B-to-C conversion used at `db/erlang.py:57`, `C = N·B / (N − A(1−B))`, is the standard `C = B / (1 − ρ(1−B))` with ρ = A/N, multiplied through by N. Correct.

Edge cases:

```
erlang_c(5,5.0)   agents==load -> 1.0
erlang_c(3,5.0)   agents<load  -> 1.0
erlang_c(4,0.0)   load 0       -> 0.0
erlang_c(4,-1.0)  load <0      -> 0.0
erlang_c(0,0.5)   agents 0     -> 1.0
erlang_c(1000,990.0)          -> 0.6590804218808543  lgamma: 0.6590804218809606
erlang_c(1000,500.0)          -> 3.3048302555026975e-86  lgamma: 3.304830255504035e-86
erlang_c(1000,999.5)          -> 0.9804842961888139  lgamma: 0.9804842961888138
erlang_c(5000,4900.0)         -> 0.09993787723448762  lgamma: 0.09993787723448927
erlang_c(2, 1.9999999)        -> 0.9999999250000006
erlang_c(2.5, 1.0) float agents -> raises TypeError 'float' object cannot be interpreted as an integer
```

No overflow at N = 1000 or 5000; agrees with lgamma to ~1e-13. `agents <= load` returns 1.0; load 0 returns 0.0. A float `agents` raises `TypeError` from `range()` rather than a clear `ValueError` (LOW, see Findings).

## Task 2. Service-level formula

With exponential service at rate μ = 1/AHT and N servers, the Erlang C waiting-time tail is

    P(W > t) = C(N, A) · exp(−(Nμ − λ) t) = C(N, A) · exp(−μ (N − A) t),   A = λ/μ

so SL(t) = 1 − C(N, A) · exp(−(N − A) · t / AHT). That is exactly `db/erlang.py:94-96` and `db/staffing.py:62`. It is the same formula the Call Centre Helper page states ("SL = 1 − (Pw · EXP(−(N − A) · (TargetTime / AHT)))") and it reproduced their service levels above to 0.04 percentage points.

Units: `target_sec` and `aht_sec` are both seconds, so the exponent is dimensionless. `load = calls_per_hour · aht_sec / 3600` is erlangs (calls/hour × hours/call). Consistent throughout `erlang.py` and `staffing.py` (`SECONDS_PER_HOUR = 3600`).

Sanity checks run:

```
N=8 A=5.0 AHT=600.0s t=0: SL=0.832733
N=8 A=5.0 AHT=600.0s t=30: SL=0.856032
N=8 A=5.0 AHT=600.0s t=60: SL=0.876086
N=8 A=5.0 AHT=600.0s t=300: SL=0.962678
N=8 A=5.0 AHT=600.0s t=1000000000.0: SL=1.000000
SL at t=0 equals 1-C: True
ASA = C*AHT/(N-A) = 33.453s (for reference)
```

SL(0) = 1 − C and SL(∞) = 1, as required. Note the formula assumes infinite patience (no abandonment). The `calls` table has 7,528 abandoned of 93,407 calls (8.1%), so Erlang C is conservative relative to an Erlang A model. This is a modelling choice, not an error.

## Task 3. `required_agents` search start

The queue is stable for any N > A. `ceil(A)` already exceeds A whenever A is non-integer, so `ceil(A)` is the first stable integer in that case; `ceil(A)+1` is only needed when A is an exact integer. Starting at `ceil(A)+1` therefore skips a candidate that may already meet the target.

Concrete case from the brief (load 0.3, AHT 600):

```
cph=1.8 aht=600 -> load=0.3
  N=1: C=0.300000 SL=0.710318
  N=2: C=0.039130 SL=0.964058
  N=3: C=0.003704 SL=0.996764
  required_agents (impl): 2  alt start: 2
```

At load 0.3 one agent gives only 71%, so both rules return 2. The skipped candidate only wins at smaller loads. Synthetic grid (offered 0.1–15.0 calls/hour step 0.1 × AHT {300,450,600,750,900,1200}), comparing the implementation against a search that starts at `max(1, ceil(load))` and bumps once if `N <= load`:

```
grid: offered 0.1..15.0 step 0.1 x AHT [300, 450, 600, 750, 900, 1200], target 0.80/30s: 900 cases, impl != alt in 76 (8.4%)
all mismatches have impl == alt+1: True

over-staffing rate by load bucket (erlangs):
 load bucket  cases   over    rate
         0-1    372     76   20.4%
         1-2    280      0    0.0%
         2-3    149      0    0.0%
         3-4     68      0    0.0%
         4-5     30      0    0.0%
         5-6      1      0    0.0%
largest load in grid where ceil(load) already meets target: 0.20833333333333334
target 0.8/30s: over-staff by 1 in 76/900 = 8.4%
target 0.8/20s: over-staff by 1 in 76/900 = 8.4%
target 0.9/30s: over-staff by 1 in 38/900 = 4.2%
target 0.7/60s: over-staff by 1 in 129/900 = 14.3%
```

At 80%/30 s the rule over-states only when load ≤ ~0.21 erlangs (one agent at 20% occupancy already meets 80/30). Above 1 erlang it never matters.

On the tool's actual operating surface (saved model's predictions for all 43,824 training hours × the current per-cell AHT):

```
AHT by cell: min=573s mean=747s max=1024s
hours: 43824; load min=0.109 median=0.404 mean=0.417 max=3.029
load<1 share: 99.2%   load<0.5 share: 73.1%
agents_required distribution (impl): {2: 42178, 3: 1586, 4: 52, 5: 7, 6: 1}
agents_required distribution (alt):  {1: 4489, 2: 37689, 3: 1586, 4: 52, 5: 7, 6: 1}
hours where impl over-staffs vs alt: 4489 / 43824 = 10.2%
      over_rate  impl_mean  alt_mean  load_mean
hour
0          0.00       2.01      2.01       0.44
1          0.00       2.00      2.00       0.37
2          1.64       2.00      1.98       0.29
3         32.42       2.00      1.68       0.24
4         62.71       2.00      1.37       0.20
5         69.44       2.00      1.31       0.19
6         60.02       2.00      1.40       0.20
7          8.76       2.00      1.91       0.27
8         10.84       2.00      1.89       0.28
9          0.00       2.00      2.00       0.38
... (hours 9-23: 0.00 over_rate)
scheduled_needed @eff=0.65: impl mean=4.04, alt mean=3.84; hours differing=4489
min SL delivered by alt staffing over all hours: 0.8000 (must be >= 0.80)
```

So the hourly `agents_required` column is one too high for 10.2% of hours, concentrated at 03:00–08:00 (up to 69% of hours at 05:00). Because the implementation starts at ≥ 2 for any positive load, it can never output 1 agent. The alternative rule still delivers ≥ 80% SL at every hour (min 0.8000).

Shift-level impact, running `rollup_shifts` on both hourly frames:

```
shift rows: 5478
shift-level staff_needed differs in 0 shifts
shift-level agents_required differs in 0 shifts
          staff_needed_impl         staff_needed_alt
                       mean min max             mean min max
shift
Day                    4.01   4   7             4.01   4   7
Evening                4.59   4  10             4.59   4  10
Overnight              4.02   4   7             4.02   4   7
staff_needed distribution impl: {4: 4424, 5: 1021, 7: 28, 8: 4, 10: 1}
staff_needed distribution alt:  {4: 4424, 5: 1021, 7: 28, 8: 4, 10: 1}
capped shifts impl: 5  alt: 5
```

**Zero shift-level difference**: every Overnight shift contains hours 00:00–02:00 (load ~0.3–0.44) that need 2 agents, and the shift takes the max. So the over-statement is real in the hourly table but invisible in the shift output.

Classification: this is a **spec issue**, not an implementation bug. The implementation does exactly what the spec ("start at agents = ceil(load) + 1") says; the spec's premise that ceil(load)+1 is "the first stable staffing level" is wrong for non-integer load. The docstring at `db/erlang.py:73` repeats the wrong premise.

## Task 4. The failing test `required_agents(30, 600, 0.80, 30) == 6`

```
=== pytest ===
FAILED tests/test_erlang.py::test_required_agents_known_case - assert 8 == 6
1 failed, 12 passed in 0.02s
```

Independent computation (load = 30 × 600 / 3600 = 5.0 erlangs):

```
load = 5.0
  N=5: C=1.000000  SL(30s)=0.000000  SL(60s)=0.000000  SL(120s)=0.000000  ASA=infs
  N=6: C=0.587516  SL(30s)=0.441137  SL(60s)=0.468393  SL(120s)=0.518982  ASA=352.5s
  N=7: C=0.324150  SL(30s)=0.706697  SL(60s)=0.734608  SL(120s)=0.782716  ASA=97.2s
  N=8: C=0.167267  SL(30s)=0.856032  SL(60s)=0.876086  SL(120s)=0.908202  ASA=33.5s
  N=9: C=0.080510  SL(30s)=0.934084  SL(60s)=0.946032  SL(120s)=0.963824  ASA=12.1s
  N=10: C=0.036105  SL(30s)=0.971881  SL(60s)=0.978101  SL(120s)=0.986718  ASA=4.3s
impl result: 8  alt-start result: 8
target_sec needed for N=6 to hit 80%: 646.6s
SL that N=6 achieves at 30s: 0.4411
  cph=10: required=4
  cph=12: required=4
  cph=14: required=5
  cph=15: required=5
  cph=16: required=5
  cph=18: required=6
  cph=20: required=6
```

**The correct answer is 8.** N=6 answers 44% within 30 s and N=7 answers 71%; the target of 80% is first reached at N=8 (85.6%). This matches the C(8,5)=0.1673 reference in Task 1. The search-start rule is irrelevant here because load is an exact integer (both rules start at 6).

Is 6 achievable under any reasonable reading? No:
- Different start rule: no, the answer is 8 under both.
- Different `target_sec`: N=6 would need a 647-second (10.8-minute) answer target to hit 80%. Not reasonable.
- Different SL denominator (answered-only, as with abandonment): the model has no abandonment term, so there is no alternative denominator.
- Different volume: 6 is the answer for 18–20 calls/hour at AHT 600, not 30.

The most plausible origin of "6" is the docstring's own phrase: 6 = ceil(5.0)+1 = the first stable level, mistaken for the answer. The test's expected value is wrong, not the code. Fix: assert 8, and ideally also assert the intermediate SLs (e.g. N=7 → 0.7067 < 0.80 ≤ 0.8560 at N=8) so the test pins the formula, not just the integer.

## Task 5. Efficiency division

Direction: `scheduled = ceil(agents / efficiency)`. Lower efficiency → larger denominator effect → more people. Verified:

```
  agents=4 eff=0.5: scheduled=8
  agents=4 eff=0.65: scheduled=7
  agents=4 eff=0.85: scheduled=5
  agents=4 eff=1.0: scheduled=4
```

Claim: `floor(scheduled · eff) ≥ agents`. Proof: `scheduled = ceil(a/e) ≥ a/e`, so `scheduled · e ≥ a`, and since `a` is an integer, `floor(scheduled · e) ≥ a`. Numerically over efficiency 0.50–0.85 step 0.01 and agents 1–20, in both float and exact rational arithmetic:

```
checked 720 combos; violations: 0
violations under exact rational arithmetic: 0
```

Floating-point spurious rounding (e.g. `7/0.7` landing just above 10 and ceiling to 11) was also checked for efficiency 0.50–1.00 and agents 1–20 / scheduled 1–40:

```
ceil(agents/eff) FP vs exact mismatches: 0
floor(scheduled*eff) FP vs exact mismatches: 0
```

No issue. Note that `service_level_at_scheduled` is computed with `floor(scheduled·eff)` agents, which is ≥ `agents_required`, so the reported SL is always ≥ target. Correct and slightly conservative.

## Task 6. Shift rollup

Hand-built 48-hour frame, 2026-03-02 00:00 to 2026-03-03 23:00. `predicted_offered = day_index·100 + hour` (so per-shift sums are checkable by hand); `scheduled_needed` = 1 everywhere except spikes placed on boundary hours: day 0 hour 8 → 5, hour 16 → 6, hour 23 → 2; day 1 hour 0 → 9, hour 8 → 3, hour 16 → 4.

```
memberships count: 54 (expect 48 core + 6 boundary rows = 54)
shift_date  shift      is_core
2026-03-01  Evening    False      1
2026-03-02  Day        False      1
                       True       8
            Evening    False      1
                       True       8
            Overnight  False      1
                       True       8
2026-03-03  Day        False      1
                       True       8
            Evening    True       8
            Overnight  False      1
                       True       8

rollup_shifts output:
  shift_date      shift  predicted_calls  agents_required  staff_needed  capped
0 2026-03-02  Overnight             28.0                5             5   False
1 2026-03-02        Day             92.0                6             6   False
2 2026-03-02    Evening            156.0                9             9    True
3 2026-03-03  Overnight            828.0                9             9    True
4 2026-03-03        Day            892.0                4             4   False
5 2026-03-03    Evening            956.0                4             4   False

hand-computed daily totals: day0 = 276  day1 = 2676
  sum of shift totals 2026-03-02: 276.0  match: True
  sum of shift totals 2026-03-03: 2676.0  match: True

expected per-shift totals day0: Overnight 0..7 = 28  Day 8..15 = 92  Evening 16..23 = 156
expected max staffing day0: Overnight max(h0..h8)=5 (boundary hr 8), Day max(h8..h16)=6 (boundary 16), Evening max(h16..23, day1 h0)=9 (boundary hr 0 of day1)
expected max staffing day1: Overnight max(h0..h8)=9, Day max(h8..h16)=4, Evening max(h16..23, day2 h0 missing)=4
staffing matches expectations: {(Timestamp('2026-03-02 00:00:00'), 'Overnight'): (5, 5, True), (Timestamp('2026-03-02 00:00:00'), 'Day'): (6, 6, True), (Timestamp('2026-03-02 00:00:00'), 'Evening'): (9, 9, True), (Timestamp('2026-03-03 00:00:00'), 'Overnight'): (9, 9, True), (Timestamp('2026-03-03 00:00:00'), 'Day'): (4, 4, True), (Timestamp('2026-03-03 00:00:00'), 'Evening'): (4, 4, True)}
ordering: [('03-02', 'Overnight'), ('03-02', 'Day'), ('03-02', 'Evening'), ('03-03', 'Overnight'), ('03-03', 'Day'), ('03-03', 'Evening')]
shift dtype: category
capped column: [False, False, True, True, False, False]

Evening of 2026-03-01 present? (should be dropped): False
Day0 Day-shift predicted_calls: 92.0 expected 92
```

Verified:
- Staffing is the **max**, not the mean (Day-0 Day shift = 6 from a single spike hour; mean would be ~1.6).
- Boundary hour joins the max of both shifts: day-0 hour 8 (=5) is the Overnight max and hour 16 (=6) is the Day max; day-1 hour 0 (=9) is both day-0 Evening max and day-1 Overnight max.
- Evening of day D includes hour 0 of D+1 for the max (day-0 Evening = 9).
- Boundary hour's calls are **not** added to the shift total (Day shift total is 92 = 8+…+15, hour 16's 16 calls excluded). Per-shift totals sum exactly to the daily totals (276 and 2676).
- Ordering Overnight → Day → Evening by date; leading-edge Evening of 2026-03-01 (boundary only) correctly dropped; last-day Evening has no D+1 hour 0 and gracefully uses only its 8 core hours.
- `capped` is `staff_needed > 7`.
- Input frame is not mutated (columns unchanged after the call).

Two edge behaviours worth knowing (not bugs in the documented contract):

```
partial window ending 16:00 on day1:
  ...
5 2026-03-03    Evening            116.0                4             4   False
```
A shift with only some of its core hours in the window is still emitted, with a partial total (116 = hour 16 only) and a max over partial hours. `staff_for_range` always feeds whole days so this cannot happen in the current pipeline; a `core_hours == 8` filter (or a flag) would make it robust.

```
duplicate hour row -> Day0 Day predicted_calls: 102.0 (double-counts if != 92)
```
Duplicate `hour_start` rows double-count. `forecast_range` builds a unique grid, so also unreachable today; a uniqueness assert on `hour_start` would be cheap.

**No bug found in `rollup_shifts` or `_shift_memberships`.**

## Task 7. Poisson objective appropriateness

Actual `offered` distribution on the complete 43,824-hour grid (built with the repo's own `complete_hourly_grid` from the `calls_hourly` view):

```
view rows: 35872  range: 2021-01-01 00:00:00 .. 2025-12-31 23:00:00
grid rows: 43824  zeros: 7952 share zeros: 18.15%
mean=2.1314 var=3.1501 var/mean=1.4779 max=18 p99=7.0
value counts (0..max):
0      7952
1     10824
2      9467
3      6867
4      4275
5      2339
6      1188
7       499
8       227
9       106
10       49
11       15
12        9
13        4
14        1
15        1
18        1
marginal Poisson(mean) expected share vs observed:
  k=0: obs= 18.15%  pois= 11.87%
  k=1: obs= 24.70%  pois= 25.29%
  k=2: obs= 21.60%  pois= 26.96%
  k=3: obs= 15.67%  pois= 19.15%
  k=4: obs=  9.75%  pois= 10.20%
  k=5: obs=  5.34%  pois=  4.35%
  k=6: obs=  2.71%  pois=  1.55%
  k=7: obs=  1.14%  pois=  0.47%
  k=8: obs=  0.52%  pois=  0.13%
  k>=9: obs=  0.42%  pois=  0.04%
conditional (year,month,dow,hour) cells: 10080; pooled within-cell var/mean = 1.0627
conditional (dow,hour) cells: 168; pooled within-cell var/mean = 1.1247
by hour of day:  (ratio = var/mean)
hour   mean    var  ratio
0     1.757  2.094  1.192
4     0.758  0.781  1.030
14    2.850  3.318  1.164
21    3.402  4.228  1.243
(full table in scratchpad output; ratios range 1.00 (hour 1) to 1.245 (hour 20))
saved model in-sample: mean pred=2.1314 mean y=2.1314  Pearson dispersion phi=0.9568  pred min=0.4090 max=12.6468
deviance dispersion = 1.0875
```

(Note: the brief said mean ~1.6/hour; the measured grid mean is 2.13. The zero count 7,952 and the 35,872/43,824 row counts match the brief exactly.)

Out-of-sample check, one in-memory fold (train < 2025-01-01, test = all of 2025, nothing saved):

```
OOS fold train<2025, test=2025: n=8760 mean y=2.4655 mean pred=2.4249
  Pearson dispersion phi = 1.0918   deviance dispersion = 1.1788
  var(y)=3.6582; E[pred]=2.4249; var(pred)=1.0315; Poisson decomposition var(y) ~ E[mu]+var(mu) = 3.4563
  Poisson q50 coverage: 0.630 (nominal >= 0.5)
  Poisson q80 coverage: 0.852 (nominal >= 0.8)
  Poisson q90 coverage: 0.925 (nominal >= 0.9)
  Poisson q95 coverage: 0.962 (nominal >= 0.95)
```

Interpretation:
- The **marginal** variance/mean of 1.48 and the excess zeros (18% vs 12% for a single Poisson) are what you expect from a mixture of hours with different rates (overnight mean 0.7, evening mean 3.4). That heterogeneity is exactly what the covariates explain.
- **Conditional** on the features the model has (dow × hour, plus year-month), the within-cell variance/mean is 1.06–1.12. Out of sample, the Pearson dispersion of the LightGBM model is 1.09 and the Poisson-mixture identity var(y) ≈ E[μ] + var(μ) holds within 6% (3.46 vs 3.66). Poisson predictive quantiles cover at or slightly above nominal.
- Residual over-dispersion is therefore mild (≈ 10%). A negative-binomial or Tweedie objective would widen predictive intervals slightly but would not change the conditional mean, and LightGBM's Poisson objective is a consistent estimator of the conditional mean even under over-dispersion (quasi-likelihood argument). Erlang C consumes only the mean.

**Verdict: Poisson objective is appropriate. No change recommended.** If the tool ever reports prediction intervals, inflate Poisson variance by ~1.1 (or switch to Tweedie with power 1.1–1.3), but that is uncertainty, not staffing.

One side observation from the same run: the saved model's minimum prediction is 0.409 calls/hour and the minimum per-cell AHT is 573 s, so the smallest load the tool ever sees is 0.109 erlangs and the largest 3.03. Combined with the `ceil(load)+1` start this means `agents_required` is always ≥ 2, `scheduled_needed` always ≥ 4, and 4,424 of 5,478 shifts (81%) come out at exactly 4 people. The output is flat by construction, which is worth knowing when interpreting it.

## Task 8. AHT

Weight formula (`db/aht.py:56-58`): `0.5 ** (age_days / halflife_days)` where age is days before the newest call. Verified:

```
dates: [datetime.date(2026, 1, 1), datetime.date(2025, 1, 1), datetime.date(2024, 1, 1), datetime.date(2025, 7, 2)]
weights: [1.         0.5        0.24952569 0.70643569]
expected: 1.0, 0.5**(365/365)=0.5, 0.5**(731/365)=0.249526, 0.5**(183/365)=0.706436
```

Exponentially weighted mean = Σ w·sec / Σ w, computed per cell and globally. Correct.

Data checks from the `calls` table (answered calls only):

```
n, mean, median, sd, min, max, first, last: (85879, 703.18, 571.0, 504.67, 42, 7200, 2021-01-01 00:54:55, 2025-12-31 23:42:57)
abandoned / total: (7528, 93407)
dow vs isodow vs pg dow: [(1,1,1), (2,2,2), ..., (6,6,6), (7,7,0)]
(dow,hour) cells with < 30 answered calls: 0  smallest cells: [(7, 4, 149), (6, 6, 156), (5, 4, 156), ...]
answered calls with null/nonpositive agent_sec: 0
```

- **Mean vs median**: Erlang C assumes exponential service with mean 1/μ; the parameter it needs is the mean handle time. `agent_sec` is right-skewed (mean 703 s, median 571 s), so using the median would understate load by ~19% and understaff. The code uses the mean. Correct.
- **`MIN_CELL_CALLS = 30`**: sensible as a noise floor (relative SE of a mean with CV≈1 at n=30 is ~18%). With this data it never triggers (smallest cell has 149 calls). One subtlety: the threshold counts raw calls, but the weighted mean's effective sample size is smaller (for a uniform 5-year stream with a 365-day half-life, effective n ≈ 54% of raw). Harmless at current volumes.
- **Units**: `agent_sec` is integer seconds; the lookup key `(dow, hour)` uses the table's `dow` which is ISO (1=Mon…7=Sun) and matches `features.py`'s `isocalendar().day`. Consistent. No unit issue.
- **Scope of AHT**: `agent_sec` appears to be talk time only. If the centre has after-call work, Erlang load should use talk + wrap; that cannot be determined from the schema and is flagged as a question, not a defect. Max `agent_sec` is exactly 7200 for at least one call, which looks like a source-system cap.

## Task 9. Other observations

- `db/staffing.py:57-62` `_service_level` returns a negative number when `agents <= load` (Erlang C = 1, exponent positive): `_service_level(1, 1.5, 600, 30) = -0.0253`. Unreachable from `staff_for_hour` because `effective_agents ≥ agents_required > load`, but the helper is not safe to reuse.
- `db/erlang.py:54` `range(1, agents + 1)` raises an opaque `TypeError` for float `agents`. `staff_for_hour` always passes ints, so no live impact.
- `db/erlang.py:84` allows `target_pct = 1.0`. It terminates in practice because C(N,A) underflows to give SL == 1.0 in float before N reaches 1000 (e.g. C(1000,500) = 3e-86), but the docstring's "absurd inputs" comment is the only guard.
- `db/train.py:145` MAPE excludes zero-actual hours (18% of the grid) so it is not comparable across models with different zero handling; the gate uses daily MAE, which is fine.
- Erlang C ignores abandonment (8.1% of calls). Staffing is conservative; acceptable for a hotline.

---

## Findings (ranked)

| # | Severity | Location | Finding | Recommended fix (not applied) |
|---|---|---|---|---|
| 1 | **HIGH** | `tests/test_erlang.py:77` | Asserts `required_agents(30, 600, 0.80, 30) == 6`. Correct value under the stated formula is **8** (SL: N=6 → 0.441, N=7 → 0.707, N=8 → 0.856). 6 is not reachable under any reasonable reading; it appears to be ceil(load)+1 mistaken for the answer. The suite is red because of the test, not the code. | Change expectation to 8. Add an assertion on the underlying SL values (e.g. `1 - erlang_c(8,5)*exp(-3*30/600) ≈ 0.8560`) and a cited reference case (A=10, N=13/14 from Call Centre Helper). |
| 2 | **MEDIUM** | `db/erlang.py:90` (and docstring `:73`) | Search starts at `ceil(load)+1`, but `ceil(load)` is already stable whenever load is non-integer. On the real forecast surface this over-states hourly `agents_required` by 1 for 4,489 of 43,824 hours (10.2%), all at load ≤ 0.21 erlangs (03:00–08:00). Shift-level output is unchanged (0 of 5,478 shifts differ) because every shift contains an hour that needs 2. **Spec issue**: the brief the author was given said ceil(load)+1; the implementation follows it. | Start at `max(1, ceil(load))` and increment once if `agents <= load`. Update the docstring. Add a test: `required_agents(1.0, 600, 0.8, 30) == 1` (load 0.167, SL at N=1 = 0.85). |
| 3 | LOW | `db/staffing.py:57-62` | `_service_level` returns negative values when `agents <= load`. Unreachable today. | Return 0.0 when `agents <= load` (or clamp to [0,1]). |
| 4 | LOW | `db/staffing.py:144-154` | `rollup_shifts` emits shifts with partial core hours at the window edge with a partial call total, and double-counts duplicate `hour_start` rows. Both unreachable via `staff_for_range`. | Assert `hour_start` is unique; either filter `core_hours == 8` or add an `is_complete` column. |
| 5 | LOW | `db/erlang.py:54` | Float `agents` raises `TypeError` from `range`. | Validate `isinstance(agents, int)` or cast with a `ValueError` message. |
| 6 | INFO | `db/aht.py` | `agent_sec` may exclude after-call work; max value 7200 exactly suggests a cap. Not verifiable from schema. | Confirm with the client whether wrap-up time exists and should be added to AHT. |
| 7 | INFO | model / tool output | Minimum predicted load is 0.109 erlangs and, with finding 2, `scheduled_needed` is ≥ 4 at every hour; 81% of shifts land on exactly 4. Not a math error, but the output is nearly constant by construction. | None required; worth stating in the README so users do not mistake a flat schedule for a model failure. |

No CRITICAL findings. Poisson objective, Erlang C, service-level formula, efficiency arithmetic and shift rollup all verified correct.
