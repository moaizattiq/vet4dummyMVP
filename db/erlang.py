"""Erlang C staffing math for the Vets4Warriors call volume tool.

Two public functions, nothing else. No database access, no imports from
other project files. Pure math so it can be unit-tested in isolation and
called from anywhere (forecaster, API, notebook).

    erlang_c(agents, load)          -> probability an arriving call waits
    required_agents(calls_per_hour,
                    aht_sec,
                    target_pct,
                    target_sec)     -> smallest agent count meeting the
                                       service-level target

Terminology:
    load (erlangs) = calls_per_hour * aht_sec / 3600
                   = average number of agents busy at once.
"""

from __future__ import annotations

import math

# Hard stop for the search loop in required_agents. Erlang C falls off
# quickly once agents > load, so hitting this means the inputs are absurd
# (e.g. target_pct = 1.0), not that we need more agents.
MAX_AGENTS = 1000


def erlang_c(agents: int, load: float) -> float:
    """Probability an arriving call has to wait (Erlang C), for `agents`
    servers and offered `load` in erlangs.

    Computed by walking the Erlang B recurrence

        B(0, A) = 1
        B(k, A) = A * B(k-1, A) / (k + A * B(k-1, A))

    up to k = agents, then converting

        C(N, A) = N * B / (N - A * (1 - B))

    The recurrence never forms a factorial, so it does not overflow.

    Edge cases:
        load <= 0            -> 0.0 (nobody waits when nobody calls)
        agents <= load       -> 1.0 (queue is unstable; everyone waits)
    """
    if load <= 0:
        return 0.0
    if agents <= 0 or agents <= load:
        return 1.0

    blocking = 1.0
    for k in range(1, agents + 1):
        blocking = load * blocking / (k + load * blocking)

    waiting = agents * blocking / (agents - load * (1.0 - blocking))
    # Floating-point can nudge the result a hair outside [0, 1].
    return min(1.0, max(0.0, waiting))


def required_agents(
    calls_per_hour: float,
    aht_sec: float,
    target_pct: float = 0.80,
    target_sec: float = 30,
) -> int:
    """Smallest number of agents such that at least `target_pct` of calls
    are answered within `target_sec` seconds.

    service_level = 1 - C(N, A) * exp(-(N - A) * target_sec / aht_sec)

    Search starts at ceil(load) + 1 (the first stable staffing level) and
    increments by one until the target is met.

    Returns 0 when calls_per_hour == 0.
    """
    if calls_per_hour == 0:
        return 0
    if calls_per_hour < 0:
        raise ValueError(f"calls_per_hour must be >= 0, got {calls_per_hour}")
    if aht_sec <= 0:
        raise ValueError(f"aht_sec must be > 0, got {aht_sec}")
    if not 0 < target_pct <= 1:
        raise ValueError(f"target_pct must be in (0, 1], got {target_pct}")
    if target_sec < 0:
        raise ValueError(f"target_sec must be >= 0, got {target_sec}")

    load = calls_per_hour * aht_sec / 3600.0
    agents = math.ceil(load) + 1

    while agents <= MAX_AGENTS:
        wait_prob = erlang_c(agents, load)
        service_level = 1.0 - wait_prob * math.exp(
            -(agents - load) * target_sec / aht_sec
        )
        if service_level >= target_pct:
            return agents
        agents += 1

    raise ValueError(
        f"no staffing level up to {MAX_AGENTS} agents reaches "
        f"{target_pct:.0%} within {target_sec}s for load {load:.2f} erlangs"
    )
