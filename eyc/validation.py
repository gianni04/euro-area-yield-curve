"""Compare the rebuilt curves with the rates published by the ECB.

The ECB publishes both the parameters and the spot, forward and par yield
curves, so rebuilding the rates and taking the difference should give zero.
Any residual is either a bug or an inconsistency in the published data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import loader
from .svensson import (
    PAR_YIELD_METHOD_CHANGE,
    discount_factor,
    instantaneous_forward,
    par_yield,
    spot_rate,
)

KINDS = {"SR": spot_rate, "IF": instantaneous_forward, "PY": par_yield}

# Rates are published in percent, so one basis point is 0.01.
BP = 100.0

# A correct reconstruction lands far inside this. It is there to flag failures.
TOLERANCE_BP = 0.01


def reconstruct(params: pd.DataFrame, maturities, kind: str) -> pd.DataFrame:
    """Rebuild a rate surface (dates x maturities) from the Svensson parameters."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")

    taus = np.asarray(list(maturities), dtype=float)
    betas = [params[name].to_numpy(dtype=float) for name in loader.PARAMS]

    if kind == "PY":
        # The par yield integrates the discount curve, which does not broadcast
        # across dates, so it is evaluated one curve at a time. The convention
        # also changes partway through the sample.
        switch = pd.Timestamp(PAR_YIELD_METHOD_CHANGE)
        methods = np.where(params.index >= switch, "integral", "monthly")
        values = np.vstack(
            [
                par_yield(taus, *[b[i] for b in betas], method=methods[i])
                for i in range(len(params))
            ]
        )
    else:
        values = KINDS[kind](taus[None, :], *[b[:, None] for b in betas])

    return pd.DataFrame(values, index=params.index, columns=taus)


def compare(published: pd.DataFrame, rebuilt: pd.DataFrame) -> pd.DataFrame:
    """Signed reconstruction error in basis points, aligned on both axes."""
    common_dates = published.index.intersection(rebuilt.index)
    published_taus = set(published.columns)
    common_taus = [c for c in rebuilt.columns if c in published_taus]
    if not len(common_dates) or not common_taus:
        raise ValueError("published and rebuilt frames do not overlap")

    left = published.loc[common_dates, common_taus].astype(float)
    right = rebuilt.loc[common_dates, common_taus].astype(float)
    return (right - left) * BP


def summarise(errors: pd.DataFrame, label: str) -> dict:
    """Summary statistics of an error surface: max error, RMSE, dates above tolerance."""
    flat = errors.stack(future_stack=True).dropna()
    if flat.empty:
        raise ValueError(f"no overlapping observations for {label}")

    absolute = flat.abs()
    worst = absolute.idxmax()
    per_date = errors.abs().max(axis=1)
    breaches = per_date[per_date > TOLERANCE_BP]
    return {
        "series": label,
        "observations": int(flat.size),
        "dates": int(errors.shape[0]),
        "maturities": int(errors.shape[1]),
        "max_abs_bp": float(absolute.max()),
        "mean_abs_bp": float(absolute.mean()),
        "rmse_bp": float(np.sqrt((flat**2).mean())),
        "p99_abs_bp": float(absolute.quantile(0.99)),
        "dates_above_tolerance": int(breaches.size),
        "breach_dates": ", ".join(str(d)[:10] for d in breaches.index[:5]),
        "worst_date": str(worst[0])[:10],
        "worst_maturity": float(worst[1]),
    }


def validate(
    kind: str,
    curve: str = "ALL",
    maturities=None,
    start: str = loader.FIRST_DATE,
    end: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Download, rebuild and compare one rate curve. Returns errors and summary."""
    taus = list(loader.MATURITIES if maturities is None else maturities)
    params = loader.load_parameters(curve=curve, start=start, end=end)
    published = loader.load_rates(
        kind, curve=curve, maturities=taus, start=start, end=end
    )

    errors = compare(published, reconstruct(params, taus, kind))
    return errors, summarise(errors, f"{curve}/{kind}")


def check_invariants(params: pd.DataFrame, maturities=None) -> dict:
    """Internal consistency checks that need no external reference.

    A discount factor above one, or a discount curve that rises with maturity, is
    not flagged. Both follow from negative rates, which the euro area printed from
    2015 to 2022. The invariants checked are positivity and the model's own
    spot/forward identity.
    """
    from scipy.integrate import quad

    taus = np.asarray(list(loader.MATURITIES if maturities is None else maturities))
    betas = [params[name].to_numpy(dtype=float) for name in loader.PARAMS]
    column = [b[:, None] for b in betas]

    discounts = discount_factor(taus[None, :], *column)
    spots = spot_rate(taus[None, :], *column)

    # Spot must equal the average instantaneous forward over [0, tau]. Evaluated
    # on a sample of curves, since quad cannot be vectorised over dates.
    sample = np.linspace(0, len(params) - 1, 25, dtype=int)
    integration_error = 0.0
    for i in sample:
        row = [b[i] for b in betas]
        for t in (1.0, 5.0, 10.0, 30.0):
            average = quad(
                lambda s: instantaneous_forward(s, *row), 0.0, t, limit=200
            )[0] / t
            integration_error = max(
                integration_error, abs(average - spot_rate(t, *row)) * BP
            )

    return {
        "curves_tested": int(len(params)),
        "discount_non_positive": int((discounts <= 0).sum()),
        "discount_at_zero_error": float(
            abs(np.atleast_1d(discount_factor(0.0, *[b[-1] for b in betas])) - 1.0).max()
        ),
        "spot_forward_integration_max_bp": float(integration_error),
        "curves_with_negative_spot": int((spots < 0).any(axis=1).sum()),
    }
