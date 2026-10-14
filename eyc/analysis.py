"""Curve analytics: sovereign credit premium, factor structure, carry and roll.

- sovereign spread: all-issuers curve minus AAA curve
- PCA of daily spot changes (level, slope, curvature)
- carry and roll down over a fixed horizon, curve unchanged
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import loader
from .svensson import discount_factor, forward_rate, spot_rate

BP = 100.0


def sovereign_spread(
    start: str = loader.FIRST_DATE, end: str | None = None, maturities=None
) -> pd.DataFrame:
    """All-issuers minus AAA spot rates, in basis points.

    The ECB fits the same model to two bond samples that differ only by credit
    quality, so the difference isolates what the market charges for sovereign
    credit risk, with term structure effects already netted out.
    """
    taus = list(loader.MATURITIES if maturities is None else maturities)
    everyone = loader.load_rates(
        "SR", curve="ALL", maturities=taus, start=start, end=end
    )
    aaa = loader.load_rates("SR", curve="AAA", maturities=taus, start=start, end=end)

    common = everyone.index.intersection(aaa.index)
    return (everyone.loc[common] - aaa.loc[common]) * BP


def curve_pca(
    curve: str = "ALL",
    maturities=None,
    start: str = loader.FIRST_DATE,
    end: str | None = None,
    n_factors: int = 3,
    split: float = 0.5,
) -> dict:
    """Principal components of daily spot rate changes, with a holdout test.

    The loadings are estimated on the first part of the sample and scored on the
    rest. Fitting on the whole sample would flatter the result, and a factor
    structure that does not hold out of sample is of little use.
    """
    taus = list(loader.MATURITIES if maturities is None else maturities)
    rates = loader.load_rates("SR", curve=curve, maturities=taus, start=start, end=end)
    changes = rates.diff().dropna() * BP

    cut = int(len(changes) * split)
    train, test = changes.iloc[:cut], changes.iloc[cut:]

    centred = (train - train.mean()).to_numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centred, rowvar=False))

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    # Sign convention: each factor loads positive on average, so the columns are
    # comparable across samples and readable as level, slope and curvature.
    for k in range(eigenvectors.shape[1]):
        if eigenvectors[:, k].sum() < 0:
            eigenvectors[:, k] *= -1.0

    loadings = eigenvectors[:, :n_factors]
    demeaned = (test - test.mean()).to_numpy()
    residual = demeaned - (demeaned @ loadings) @ loadings.T

    return {
        "curve": curve,
        "observations": int(len(changes)),
        "train_dates": int(len(train)),
        "test_dates": int(len(test)),
        "explained_in_sample": float(
            eigenvalues[:n_factors].sum() / eigenvalues.sum()
        ),
        "explained_out_of_sample": float(
            1.0 - (residual**2).sum() / (demeaned**2).sum()
        ),
        "factor_shares": (eigenvalues[:n_factors] / eigenvalues.sum()).tolist(),
        "loadings": pd.DataFrame(
            loadings,
            index=taus,
            columns=["level", "slope", "curvature"][:n_factors],
        ),
    }


def carry_rolldown(params, maturity: float, horizon: float = 0.25) -> dict:
    """Decompose the return on a zero coupon position if the curve does not move.

    Holding a zero of maturity T for h years with the curve unchanged returns
    DF(T-h)/DF(T) - 1. Financing that position over the same horizon costs the
    short rate, 1/DF(h) - 1. What is left is roll down: the bond has become
    shorter, and on an upward sloping curve a shorter bond is discounted less.

    This is what the position earns if nothing changes, so it is checked before
    any forecast.
    """
    if horizon <= 0 or horizon >= maturity:
        raise ValueError("horizon must be positive and shorter than maturity")

    df_now = discount_factor(maturity, *params)
    df_then = discount_factor(maturity - horizon, *params)
    df_funding = discount_factor(horizon, *params)

    total = df_then / df_now - 1.0
    carry = 1.0 / df_funding - 1.0

    return {
        "maturity": maturity,
        "horizon": horizon,
        "total_return_bp": total * 1e4,
        "carry_bp": carry * 1e4,
        "rolldown_bp": (total - carry) * 1e4,
        "spot": spot_rate(maturity, *params),
        "forward_over_horizon": forward_rate(1e-9, horizon, *params),
    }


def rolldown_table(params, maturities=(2, 5, 10, 20, 30), horizon: float = 0.25):
    """Carry and roll down across the curve: the shape of a relative value screen."""
    rows = [carry_rolldown(params, float(m), horizon) for m in maturities]
    return pd.DataFrame(rows).set_index("maturity")
