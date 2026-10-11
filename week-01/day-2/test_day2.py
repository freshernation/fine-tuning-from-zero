"""Week 1, Day 2 - gradients.

Run me:  pytest week-01/day-2 -v
"""

import torch


def test_grad_by_hand_matches_the_formula(load):
    # loss = (w*x - y)**2, so d(loss)/dw = 2*x*(w*x - y)
    assert load("descent.py").grad_by_hand(w=2.0, x=3.0, y=10.0) == -24.0


def test_autograd_agrees_with_you(load):
    descent = load("descent.py")
    by_hand = descent.grad_by_hand(2.0, 3.0, 10.0)
    by_torch = descent.autograd_grad(2.0, 3.0, 10.0)
    assert isinstance(by_torch, float)
    assert abs(by_torch - by_hand) < 1e-5


def test_step_moves_against_the_gradient(load):
    assert load("descent.py").step(w=1.0, grad=4.0, lr=0.1) == 0.6


def test_fit_line_recovers_slope_and_intercept(load):
    torch.manual_seed(0)
    xs = torch.linspace(-1, 1, 50)
    ys = 3 * xs - 1
    w, b = load("descent.py").fit_line(xs, ys, lr=0.1, steps=500)
    assert abs(w - 3) < 0.05 and abs(b + 1) < 0.05, f"got w={w:.3f}, b={b:.3f}"
