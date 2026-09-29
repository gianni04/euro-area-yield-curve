"""Svensson term structure model, in the convention the ECB uses.

ECB series code SV_C_YM = Svensson / Continuous compounding / Yield error
Minimisation. Rates are continuously compounded, in percent. Converting with
exp(y) - 1 would add about 7 bp at 10 years, so a test checks the convention.
"""

from __future__ import annotations

import numpy as np

PARAM_ORDER = ("BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2")


def _decay(x: np.ndarray) -> np.ndarray:
    """(1 - exp(-x)) / x, continuous at x = 0 where the limit is 1.

    Written with expm1 rather than 1 - exp(-x). For a maturity near zero the two
    terms of that subtraction agree to fifteen digits and cancel, which costs
    roughly seven significant figures; expm1 computes the difference directly and
    keeps the short end accurate.
    """
    out = np.ones_like(x, dtype=float)
    nz = x != 0
    out[nz] = -np.expm1(-x[nz]) / x[nz]
    return out


def _as_array(tau) -> tuple[np.ndarray, bool]:
    scalar = np.ndim(tau) == 0
    arr = np.atleast_1d(np.asarray(tau, dtype=float))
    if np.any(arr < 0):
        raise ValueError("maturities must be non-negative")
    return arr, scalar


def instantaneous_forward(tau, beta0, beta1, beta2, beta3, tau1, tau2):
    """f(t) = b0 + b1*e^-x1 + b2*x1*e^-x1 + b3*x2*e^-x2, in percent."""
    arr, scalar = _as_array(tau)
    x1, x2 = arr / tau1, arr / tau2
    e1, e2 = np.exp(-x1), np.exp(-x2)
    out = beta0 + beta1 * e1 + beta2 * x1 * e1 + beta3 * x2 * e2
    return float(out[0]) if scalar else out


def spot_rate(tau, beta0, beta1, beta2, beta3, tau1, tau2):
    """Continuously compounded zero coupon rate, in percent.

    This is the average of the instantaneous forward curve over [0, tau], which
    the exponential form integrates in closed form.
    """
    arr, scalar = _as_array(tau)
    x1, x2 = arr / tau1, arr / tau2
    d1, d2 = _decay(x1), _decay(x2)
    e1, e2 = np.exp(-x1), np.exp(-x2)
    out = beta0 + beta1 * d1 + beta2 * (d1 - e1) + beta3 * (d2 - e2)
    return float(out[0]) if scalar else out


def discount_factor(tau, beta0, beta1, beta2, beta3, tau1, tau2):
    """DF(t) = exp(-y(t) * t), with y in percent converted to decimal."""
    arr, scalar = _as_array(tau)
    y = np.atleast_1d(spot_rate(arr, beta0, beta1, beta2, beta3, tau1, tau2))
    out = np.exp(-y / 100.0 * arr)
    return float(out[0]) if scalar else out


# Gauss-Legendre nodes on [-1, 1]. 64 points integrate the discount function to
# machine precision over 30 years and, unlike scipy.integrate.quad, vectorise
# across every maturity and every date at once.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)


def annuity(tau, beta0, beta1, beta2, beta3, tau1, tau2):
    """Continuous annuity factor, the integral of DF(t) over [0, tau]."""
    arr, scalar = _as_array(tau)
    params = (beta0, beta1, beta2, beta3, tau1, tau2)

    half = arr[:, None] / 2.0
    points = half * (_GL_NODES[None, :] + 1.0)
    values = np.exp(
        -spot_rate(points.ravel(), *params).reshape(points.shape) / 100.0 * points
    )
    out = half[:, 0] * (values * _GL_WEIGHTS[None, :]).sum(axis=1)
    return float(out[0]) if scalar else out


# The ECB replaced the discrete monthly annuity with the exact integral on
# 2023-05-19 and did not restate the earlier history, so both conventions are
# needed to reproduce the published par yield curve end to end. The difference
# scales with the level of rates: about 1 bp in 2011, near zero in 2020-21.
PAR_YIELD_METHOD_CHANGE = "2023-05-19"

LEGACY_STEPS_PER_YEAR = 12


def _par_yield_monthly(arr: np.ndarray, params: tuple) -> np.ndarray:
    """Legacy annuity: right Riemann sum of DF at monthly steps.

    sum(DF(j/12) * 1/12) for j = 1..12T, which understates the integral because
    the discount function is decreasing, and so overstates the par yield.
    """
    step = 1.0 / LEGACY_STEPS_PER_YEAR
    counts = np.rint(arr * LEGACY_STEPS_PER_YEAR).astype(int)
    if np.any(np.abs(arr * LEGACY_STEPS_PER_YEAR - counts) > 1e-9):
        raise ValueError("legacy par yield requires maturities on a monthly grid")

    nodes = np.arange(1, counts.max() + 1) * step
    discounts = np.atleast_1d(discount_factor(nodes, *params))
    cumulative = np.cumsum(discounts) * step

    index = counts - 1
    return 100.0 * (1.0 - discounts[index]) / cumulative[index]


def par_yield(tau, beta0, beta1, beta2, beta3, tau1, tau2, method: str = "integral"):
    """Continuously compounded par yield, in percent.

    The ECB technical notes define it against a *continuous* annuity,

        y_par(T) = C * (1 - DF(T)) / integral of DF(t) dt over [0, T].

    method="integral" evaluates that integral, which reproduces the published
    curve from 2023-05-19 onward. method="monthly" reproduces the discrete
    approximation used before that date.
    """
    if method not in {"integral", "monthly"}:
        raise ValueError("method must be 'integral' or 'monthly'")

    arr, scalar = _as_array(tau)
    params = (beta0, beta1, beta2, beta3, tau1, tau2)

    out = np.full(arr.shape, np.nan, dtype=float)
    live = arr > 0
    if live.any():
        if method == "monthly":
            out[live] = _par_yield_monthly(arr[live], params)
        else:
            numerator = 1.0 - np.atleast_1d(discount_factor(arr[live], *params))
            out[live] = 100.0 * numerator / np.atleast_1d(annuity(arr[live], *params))

    return float(out[0]) if scalar else out


def forward_rate(start, end, beta0, beta1, beta2, beta3, tau1, tau2):
    """Continuously compounded forward rate between two maturities, in percent.

    f(a, b) = (y(b)*b - y(a)*a) / (b - a). This is the rate locked in today for
    the period a to b, and the basis for roll down analysis.
    """
    a, b = float(start), float(end)
    if b <= a:
        raise ValueError("end maturity must exceed start maturity")
    params = (beta0, beta1, beta2, beta3, tau1, tau2)
    return (spot_rate(b, *params) * b - spot_rate(a, *params) * a) / (b - a)


def params_from_row(row) -> tuple[float, ...]:
    """Extract the six Svensson parameters from a parameter frame row, in order."""
    return tuple(float(row[name]) for name in PARAM_ORDER)
