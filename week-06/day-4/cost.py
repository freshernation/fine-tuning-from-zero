"""Day 4 - latency and money, as arithmetic."""

import time


def measure_latency(call, runs, clock=time.perf_counter, warmup=1):
    """How long does call() take? Run it warmup + runs times; time only the last `runs`.

    The first request to a server is slow for reasons that have nothing to do with
    a typical request (the model is being loaded into memory), which is what the
    warm-up is for.

    clock   a function returning the current time in seconds; read it immediately
            before and immediately after each call.

    Returns {"runs": ..., "p50_ms": ..., "p95_ms": ..., "max_ms": ...}.
    Percentiles by linear interpolation between the sorted times: for fraction q the
    position is q * (runs - 1); if that falls between two times, take the weighted
    average of them.
    """
    raise NotImplementedError


def hosted_cost(input_tokens, output_tokens, price_in_per_m, price_out_per_m):
    """The price of one hosted call, in the currency the prices are quoted in.
    Prices are per MILLION tokens, which is how providers list them."""
    raise NotImplementedError


def monthly_cost(requests_per_day, cost_per_request):
    """Thirty days of it."""
    raise NotImplementedError


def break_even_requests(fixed_cost, hosted_cost_per_request, local_cost_per_request):
    """How many requests before a local model has paid back what it cost to build.

    fixed_cost   what you spent before the first request: your hours, GPU time.
    Each request then saves (hosted - local). If it saves nothing, or costs more,
    there is no such number: return None.
    """
    raise NotImplementedError
