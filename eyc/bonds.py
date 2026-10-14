"""Fixed coupon bond pricing and interest rate risk off a Svensson curve.

Two sets of risk measures:
- yield based: modified duration and convexity from the bond's own yield
- curve based: effective duration and DV01 from shifting the zero curve

On a sloped curve the two differ, and the gap is reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .svensson import discount_factor, spot_rate

# Curve risk is measured by a one basis point parallel shift. The Svensson spot
# rate is beta0 plus terms that do not involve beta0, so bumping beta0 moves
# every maturity by exactly the same amount: an exact parallel shift.
SHIFT_BP = 1.0


@dataclass(frozen=True)
class Bond:
    """A fully paid fixed coupon bullet bond.

    maturity: years to redemption
    coupon: annual coupon rate in percent
    frequency: coupon payments per year
    face: redemption value
    """

    maturity: float
    coupon: float
    frequency: int = 1
    face: float = 100.0

    def __post_init__(self) -> None:
        if self.maturity <= 0:
            raise ValueError("maturity must be positive")
        if self.frequency not in (1, 2, 4):
            raise ValueError("frequency must be 1, 2 or 4")
        if self.coupon < 0:
            raise ValueError("coupon must be non-negative")

    def schedule(self) -> tuple[np.ndarray, np.ndarray]:
        """Payment times in years and their amounts, seen from settlement.

        The schedule is anchored on redemption and steps backwards by one coupon
        period, so a bond with 9.7 years to run and annual coupons pays at 0.7,
        1.7, ... 9.7. Times are exact year fractions: this project prices off a
        continuous time zero curve, so no business day calendar is applied.
        """
        step = 1.0 / self.frequency
        count = int(np.ceil(round(self.maturity / step, 9)))
        times = self.maturity - step * np.arange(count - 1, -1, -1)
        times = times[times > 1e-12]

        amounts = np.full(times.shape, self.face * self.coupon / 100.0 * step)
        amounts[-1] += self.face
        return times, amounts

    def accrued(self) -> float:
        """Interest accrued since the last coupon, on the same year fraction basis."""
        step = 1.0 / self.frequency
        times, _ = self.schedule()
        remaining = times[0]
        elapsed = step - remaining if remaining < step else 0.0
        return self.face * self.coupon / 100.0 * elapsed


def dirty_price(bond: Bond, params) -> float:
    """Present value of every remaining cash flow, discounted off the curve."""
    times, amounts = bond.schedule()
    return float(np.dot(amounts, np.atleast_1d(discount_factor(times, *params))))


def clean_price(bond: Bond, params) -> float:
    """Dirty price less accrued interest: the price a screen quotes."""
    return dirty_price(bond, params) - bond.accrued()


def price_from_yield(bond: Bond, ytm: float) -> float:
    """Dirty price implied by a single redemption yield, compounded annually."""
    times, amounts = bond.schedule()
    return float(np.dot(amounts, (1.0 + ytm / 100.0) ** (-times)))


def yield_to_maturity(bond: Bond, dirty: float) -> float:
    """Redemption yield in percent that reprices the bond, solved by bisection."""
    if dirty <= 0:
        raise ValueError("dirty price must be positive")

    def objective(y: float) -> float:
        return price_from_yield(bond, y) - dirty

    return float(brentq(objective, -50.0, 200.0, xtol=1e-12, rtol=1e-15))


def _shifted(params, bump_bp: float) -> tuple:
    """Parallel shift of the zero curve by bump_bp, applied through beta0."""
    shifted = list(params)
    shifted[0] = shifted[0] + bump_bp / 100.0
    return tuple(shifted)


def dv01(bond: Bond, params) -> float:
    """Money made on a one basis point parallel fall in the zero curve.

    Quoted positive for a long position, as a desk would.
    """
    up = dirty_price(bond, _shifted(params, SHIFT_BP))
    down = dirty_price(bond, _shifted(params, -SHIFT_BP))
    return (down - up) / 2.0


def effective_duration(bond: Bond, params) -> float:
    """Curve based duration: relative price sensitivity to a parallel shift."""
    base = dirty_price(bond, params)
    up = dirty_price(bond, _shifted(params, SHIFT_BP))
    down = dirty_price(bond, _shifted(params, -SHIFT_BP))
    return (down - up) / (2.0 * base * SHIFT_BP / 10000.0)


def effective_convexity(bond: Bond, params) -> float:
    """Curve based convexity: the second order term a duration hedge misses."""
    base = dirty_price(bond, params)
    up = dirty_price(bond, _shifted(params, SHIFT_BP))
    down = dirty_price(bond, _shifted(params, -SHIFT_BP))
    return (up + down - 2.0 * base) / (base * (SHIFT_BP / 10000.0) ** 2)


# Tenors a rates desk buckets its risk into. Books are hedged bucket by bucket.
KEY_TENORS = (0.5, 2.0, 5.0, 10.0, 30.0)


def _tent_weights(times: np.ndarray, keys: tuple) -> np.ndarray:
    """Triangular basis on the key tenors, one row per key.

    Each row peaks at one at its own key and decays linearly to zero at the
    neighbouring keys, flat beyond the ends. Built by interpolating unit vectors,
    which guarantees the rows sum to one at every maturity: the sum of all key
    rate shocks is then exactly a parallel shift, and the sum of the key rate
    durations is exactly the effective duration. The tests check this identity.
    """
    grid = np.asarray(keys, dtype=float)
    rows = []
    for i in range(len(grid)):
        unit = np.zeros(len(grid))
        unit[i] = 1.0
        rows.append(np.interp(times, grid, unit))
    return np.vstack(rows)


def key_rate_durations(bond: Bond, params, keys: tuple = KEY_TENORS):
    """Sensitivity to a one basis point bump at each key tenor separately.

    A parallel DV01 says how much a book loses if the whole curve moves. It says
    nothing about *where* the risk sits, so it cannot tell you which instrument
    to hedge with. The key rate vector does: a thirty year bond carries almost
    all of its risk in the thirty year bucket, and hedging it with a ten year
    leaves most of that exposure open.
    """
    times, amounts = bond.schedule()
    rates = np.atleast_1d(spot_rate(times, *params))
    tents = _tent_weights(times, keys)

    base = float(np.dot(amounts, np.exp(-rates / 100.0 * times)))
    shift = SHIFT_BP / 100.0

    durations = []
    for tent in tents:
        up = np.dot(amounts, np.exp(-(rates + shift * tent) / 100.0 * times))
        down = np.dot(amounts, np.exp(-(rates - shift * tent) / 100.0 * times))
        durations.append((down - up) / (2.0 * base * SHIFT_BP / 10000.0))

    return pd.Series(durations, index=list(keys), name="key_rate_duration")


def macaulay_duration(bond: Bond, ytm: float) -> float:
    """Cash flow weighted average time to payment, in years."""
    times, amounts = bond.schedule()
    present = amounts * (1.0 + ytm / 100.0) ** (-times)
    return float(np.dot(times, present) / present.sum())


def modified_duration(bond: Bond, ytm: float) -> float:
    """Yield based duration, the Macaulay measure discounted one period."""
    return macaulay_duration(bond, ytm) / (1.0 + ytm / 100.0)


def risk_report(bond: Bond, params) -> dict:
    """Every price and risk figure for one bond, with both duration families."""
    dirty = dirty_price(bond, params)
    ytm = yield_to_maturity(bond, dirty)
    modified = modified_duration(bond, ytm)
    effective = effective_duration(bond, params)
    return {
        "maturity": bond.maturity,
        "coupon": bond.coupon,
        "clean_price": clean_price(bond, params),
        "dirty_price": dirty,
        "accrued": bond.accrued(),
        "ytm": ytm,
        "dv01": dv01(bond, params),
        "macaulay_duration": macaulay_duration(bond, ytm),
        "modified_duration": modified,
        "effective_duration": effective,
        "duration_gap": effective - modified,
        "effective_convexity": effective_convexity(bond, params),
    }
