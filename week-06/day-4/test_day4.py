"""Week 6, Day 4 - what it costs.

Run me:  pytest week-06/day-4 -v

cost.py is in this folder.
"""

import pytest


class Clock:
    """A pretend clock: each call to the work advances it by the next duration."""

    def __init__(self, durations):
        self.now, self.durations, self.calls = 100.0, list(durations), 0

    def time(self):
        return self.now

    def work(self):
        self.now += self.durations[self.calls]
        self.calls += 1


def test_latency_percentiles_with_the_warm_up_left_out(load):
    # the first call is slow because the model is loading; it must not be counted
    clock = Clock([9.0] + [0.010 * i for i in range(1, 21)])          # then 10 ms, 20 ms, ... 200 ms
    result = load("cost.py").measure_latency(clock.work, runs=20, clock=clock.time, warmup=1)
    assert clock.calls == 21
    assert result["runs"] == 20
    assert result["p50_ms"] == pytest.approx(105.0)                   # halfway between the 10th and 11th
    assert result["p95_ms"] == pytest.approx(190.5)
    assert result["max_ms"] == pytest.approx(200.0)


def test_latency_does_not_depend_on_the_order_of_the_calls(load):
    measure = load("cost.py").measure_latency
    a = Clock([1.0, 0.3, 0.1, 0.2])
    b = Clock([1.0, 0.1, 0.2, 0.3])
    assert measure(a.work, 3, clock=a.time) == measure(b.work, 3, clock=b.time)
    assert measure(Clock([0.5]).work, 1, clock=Clock([0.5]).time, warmup=0)["runs"] == 1


def test_no_warm_up_when_asked_for_none(load):
    clock = Clock([0.1, 0.2, 0.3])
    result = load("cost.py").measure_latency(clock.work, runs=3, clock=clock.time, warmup=0)
    assert clock.calls == 3 and result["p50_ms"] == pytest.approx(200.0)


def test_hosted_cost(load):
    hosted_cost = load("cost.py").hosted_cost
    # 1,000 tokens in and 100 out, at $3 and $15 per MILLION tokens
    assert hosted_cost(1000, 100, price_in_per_m=3.0, price_out_per_m=15.0) == pytest.approx(0.0045)
    assert hosted_cost(0, 0, 3.0, 15.0) == 0.0


def test_monthly_cost(load):
    assert load("cost.py").monthly_cost(requests_per_day=2000, cost_per_request=0.0045) == pytest.approx(270.0)


def test_break_even(load):
    break_even = load("cost.py").break_even_requests
    # $50 of your time and GPU up front; a hosted call costs 0.45 cents, a local one 0.05
    assert break_even(fixed_cost=50.0, hosted_cost_per_request=0.0045, local_cost_per_request=0.0005) == pytest.approx(12500)
    assert break_even(0.0, 0.0045, 0.0005) == 0
    assert break_even(50.0, 0.0005, 0.0005) is None, "the same price per request never pays the fixed cost back"
    assert break_even(50.0, 0.0005, 0.0045) is None, "and a local model that costs MORE per request certainly does not"
