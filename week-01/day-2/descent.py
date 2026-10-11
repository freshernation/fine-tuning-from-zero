"""Day 2 - one weight, one gradient, one step downhill.

The loss all day is the squared error of a one-weight model:

    prediction = w * x
    loss       = (prediction - y) ** 2
"""

import torch


def grad_by_hand(w, x, y):
    """The slope of the loss with respect to w, from the formula you derived on paper.

    Plain Python numbers in, a plain Python float out. No torch.
    """
    raise NotImplementedError


def autograd_grad(w, x, y):
    """The same slope, but make torch work it out.

    Build w as a tensor with requires_grad=True, compute the loss, call .backward(),
    and return w.grad as a plain float.
    """
    raise NotImplementedError


def step(w, grad, lr):
    """One step of gradient descent: the new value of w."""
    raise NotImplementedError


def fit_line(xs, ys, lr, steps):
    """Find w and b so that w * xs + b is as close to ys as possible.

    xs, ys: (n,) float tensors.  Returns (w, b) as plain floats.

    The loss is the MEAN squared error over all n points. Start w and b at 0.
    Use autograd for the gradients, then update w and b yourself, by hand:
    torch.optim is not allowed today. You will need torch.no_grad() for the
    update, and you will need to zero the gradients - find out what happens if
    you do not.
    """
    raise NotImplementedError
