"""JAX reward helpers ported from MetaWorld / dm_control."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

_DEFAULT_VALUE_AT_MARGIN = 0.1

SIGMOID_TYPE = Literal[
    "gaussian",
    "hyperbolic",
    "long_tail",
    "reciprocal",
    "cosine",
    "linear",
    "quadratic",
    "tanh_squared",
]


def _sigmoids(x, value_at_1: float, sigmoid: SIGMOID_TYPE):
    if sigmoid == "gaussian":
        scale = jnp.sqrt(-2 * jnp.log(value_at_1))
        return jnp.exp(-0.5 * (x * scale) ** 2)
    if sigmoid == "hyperbolic":
        scale = jnp.arccosh(1 / value_at_1)
        return 1 / jnp.cosh(x * scale)
    if sigmoid == "long_tail":
        scale = jnp.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)
    if sigmoid == "reciprocal":
        scale = 1 / value_at_1 - 1
        return 1 / (jnp.abs(x) * scale + 1)
    if sigmoid == "cosine":
        scale = jnp.arccos(2 * value_at_1 - 1) / jnp.pi
        scaled_x = x * scale
        return jnp.where(jnp.abs(scaled_x) < 1, (1 + jnp.cos(jnp.pi * scaled_x)) / 2, 0.0)
    if sigmoid == "linear":
        scale = 1 - value_at_1
        scaled_x = x * scale
        return jnp.where(jnp.abs(scaled_x) < 1, 1 - scaled_x, 0.0)
    if sigmoid == "quadratic":
        scale = jnp.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return jnp.where(jnp.abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)
    if sigmoid == "tanh_squared":
        scale = jnp.arctanh(jnp.sqrt(1 - value_at_1))
        return 1 - jnp.tanh(x * scale) ** 2
    raise ValueError(f"Unknown sigmoid type {sigmoid!r}.")


def tolerance(
    x,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin=0.0,
    sigmoid: SIGMOID_TYPE = "gaussian",
    value_at_margin: float = _DEFAULT_VALUE_AT_MARGIN,
):
    lower, upper = bounds
    in_bounds = jnp.logical_and(lower <= x, x <= upper)
    d = jnp.where(x < lower, lower - x, x - upper)
    scaled = jnp.where(margin > 0, d / margin, d)
    outside = jnp.where(margin > 0, _sigmoids(scaled, value_at_margin, sigmoid), 0.0)
    return jnp.where(in_bounds, 1.0, outside)


def hamacher_product(a, b):
    """T-norm product of values in [0, 1]."""
    denominator = a + b - (a * b)
    return jnp.where(denominator > 0, (a * b) / denominator, 0.0)


def rect_prism_tolerance(curr, zero, one):
    """Reward when ``curr`` lies inside the axis-aligned prism spanned by corners."""
    lo = jnp.minimum(zero, one)
    hi = jnp.maximum(zero, one)
    in_prism = jnp.all((curr >= lo) & (curr <= hi))
    diff = one - zero
    safe_diff = jnp.where(jnp.abs(diff) < 1e-8, 1.0, diff)
    scales = (curr - zero) / safe_diff
    inside_reward = scales[0] * scales[1] * scales[2]
    return jnp.where(in_prism, inside_reward, 1.0)
