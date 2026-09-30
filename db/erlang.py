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


def _erlang_a_stationary(
    calls_per_hour: float,
    aht_sec: float,
    agents: int,
    patience_sec: float,
) -> list[float]:
    """Stationary state probabilities for an M/M/N+M (Erlang A) queue."""
    if calls_per_hour <= 0:
        return [1.0]
    arrival = calls_per_hour / 3600.0
    service = 1.0 / aht_sec
    abandon = 1.0 / patience_sec
    weights = [1.0]
    total = 1.0
    small_tail = 0
    for state in range(1, 10000):
        departure = min(state, agents) * service + max(state - agents, 0) * abandon
        weight = weights[-1] * arrival / departure
        weights.append(weight)
        total += weight
        ratio = arrival / departure
        if state > agents and ratio < 1.0 and weight < total * 1e-14:
            small_tail += 1
            if small_tail >= 8:
                break
        else:
            small_tail = 0
    return [weight / total for weight in weights]


def _erlang_a_answer_within(
    queue_ahead: int,
    agents: int,
    aht_sec: float,
    patience_sec: float,
    target_sec: float,
) -> float:
    """Probability a queued arrival is answered within the target.

    Uniformization evaluates the small phase-type chain for the arriving
    caller's position. Abandonment by callers ahead moves the caller forward;
    their own abandonment is an absorbing failure.
    """
    service_rate = agents / aht_sec
    abandon_rate = 1.0 / patience_sec
    uniform_rate = service_rate + (queue_ahead + 1) * abandon_rate
    x = uniform_rate * target_sec
    transient = [0.0] * (queue_ahead + 1)
    transient[queue_ahead] = 1.0
    answered = 0.0
    poisson = math.exp(-x)
    result = poisson * answered
    cumulative_poisson = poisson

    for step in range(1, 10000):
        nxt = [0.0] * len(transient)
        answered_next = answered
        for position, probability in enumerate(transient):
            if probability == 0.0:
                continue
            total_rate = service_rate + (position + 1) * abandon_rate
            nxt[position] += probability * (1.0 - total_rate / uniform_rate)
            if position == 0:
                answered_next += probability * service_rate / uniform_rate
            else:
                nxt[position - 1] += probability * (
                    service_rate + position * abandon_rate
                ) / uniform_rate
        transient = nxt
        answered = answered_next
        poisson *= x / step
        cumulative_poisson += poisson
        result += poisson * answered
        if 1.0 - cumulative_poisson < 1e-13:
            break
    return min(1.0, max(0.0, result))


def erlang_a_metrics(
    calls_per_hour: float,
    aht_sec: float,
    agents: int,
    patience_sec: float,
    target_sec: float = 30,
) -> dict:
    """Service level and abandonment probability for an Erlang A queue."""
    if calls_per_hour < 0:
        raise ValueError(f"calls_per_hour must be >= 0, got {calls_per_hour}")
    if aht_sec <= 0 or patience_sec <= 0:
        raise ValueError("aht_sec and patience_sec must be > 0")
    if agents <= 0:
        return {"service_level": 0.0, "abandonment_probability": 1.0 if calls_per_hour else 0.0}
    if calls_per_hour == 0:
        return {"service_level": 1.0, "abandonment_probability": 0.0}

    probabilities = _erlang_a_stationary(calls_per_hour, aht_sec, agents, patience_sec)
    service_level = 0.0
    expected_queue = 0.0
    for state, probability in enumerate(probabilities):
        if state < agents:
            service_level += probability
        else:
            queue_ahead = state - agents
            service_level += probability * _erlang_a_answer_within(
                queue_ahead, agents, aht_sec, patience_sec, target_sec
            )
        expected_queue += max(state - agents, 0) * probability

    arrival_rate = calls_per_hour / 3600.0
    abandonment_probability = (expected_queue / patience_sec) / arrival_rate
    return {
        "service_level": min(1.0, max(0.0, service_level)),
        "abandonment_probability": min(1.0, max(0.0, abandonment_probability)),
    }


def required_agents_a(
    calls_per_hour: float,
    aht_sec: float,
    patience_sec: float,
    target_pct: float = 0.80,
    target_sec: float = 30,
) -> int:
    """Smallest agent count meeting the target in an Erlang A queue."""
    if calls_per_hour == 0:
        return 0
    for agents in range(1, MAX_AGENTS + 1):
        metrics = erlang_a_metrics(
            calls_per_hour, aht_sec, agents, patience_sec, target_sec
        )
        if metrics["service_level"] >= target_pct:
            return agents
    raise ValueError(
        f"no staffing level up to {MAX_AGENTS} agents reaches "
        f"{target_pct:.0%} within {target_sec}s"
    )
