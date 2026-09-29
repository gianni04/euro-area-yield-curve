"""Curve relative value: factor-neutral structures, signals and a backtest.

Only curve shape is traded (butterflies on the fitted curve), since the ECB
does not publish individual bond prices.

- weights are solved so each structure has zero exposure to the first two
  principal components (level and slope)
- loadings and z-scores use trailing windows only, and a position taken on
  day t earns the move from t to t+1 (no look-ahead)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .loader import PARAMS as PARAM_NAMES
from .svensson import spot_rate

BP = 100.0

# Round trip cost per unit of DV01 traded, in basis points of yield. Liquid euro
# curve points trade inside a basis point; half a basis point round trip is
# deliberately conservative for a study of this kind.
DEFAULT_COST_BP = 0.5

TRADING_DAYS = 256


def rolling_pca_loadings(changes: pd.DataFrame, window: int, n_factors: int = 3):
    """Factor loadings estimated on a trailing window, one set per date.

    Returns date -> (n_maturities, n_factors) array. The loadings for date t use
    changes strictly before t, so they are known when the position is taken.
    """
    values = changes.to_numpy()
    out = {}
    for i in range(window, len(changes)):
        sample = values[i - window : i]
        centred = sample - sample.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centred, rowvar=False))
        order = np.argsort(eigenvalues)[::-1][:n_factors]
        loadings = eigenvectors[:, order]
        for k in range(loadings.shape[1]):
            if loadings[:, k].sum() < 0:
                loadings[:, k] *= -1.0
        out[changes.index[i]] = loadings
    return out


def neutral_weights(loadings: np.ndarray, legs: list[int], normalise: int = 1):
    """Weights on three curve points orthogonal to the first two factors.

    With three legs and two neutrality constraints the solution is unique up to
    scale, so it is pinned by fixing the belly weight. The result is a genuine
    curvature trade: it makes nothing from a parallel move or a pure steepening,
    only from the belly moving against a factor-weighted blend of the wings.
    """
    if len(legs) != 3:
        raise ValueError("a butterfly needs exactly three legs")

    constraints = loadings[legs, :2].T  # (2 factors, 3 legs)
    # The null space of a 2x3 constraint matrix is one dimensional.
    _, singular, right = np.linalg.svd(constraints)
    weights = right[-1]

    if abs(weights[normalise]) < 1e-12:
        raise ValueError("degenerate neutral weights")
    return weights / weights[normalise]


def conditioning(loadings: np.ndarray, legs: list[int]) -> float:
    """Condition number of the neutrality constraint, per structure.

    The wings must be far enough apart, in factor space, to pin the belly. When
    the three legs load almost identically on level and slope the null space is
    nearly undetermined: hedge ratios blow up, the structure carries enormous
    gross risk for a tiny net view, and the trade is an artefact of the solver
    rather than a position. A 1s3s7s butterfly on this curve reaches weights of
    fourteen against one. Screen on this before trading, not after.
    """
    singular = np.linalg.svd(loadings[legs, :2].T, compute_uv=False)
    return float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")


# Gross DV01 above which a three leg structure is treated as untradeable: the
# hedge ratios are being driven by solver noise rather than by curve shape.
MAX_GROSS_DV01 = 6.0


def build_fly(
    rates: pd.DataFrame,
    legs: tuple[float, float, float],
    window: int = 250,
    n_factors: int = 3,
    reset_every: int = 21,
) -> pd.DataFrame:
    """Factor-neutral butterfly level through time, in basis points.

    Hedge ratios are re-solved every reset_every days and held constant in
    between, which is how a desk actually runs a book. Re-solving daily injects a
    jump into the level every time the weights move, and that jump is not a
    market move: it makes the series look like noise and destroys any estimate of
    mean reversion taken from it.

    The naive 2:-1:-1 version is returned alongside, so the difference between a
    structure that is actually neutral and one that only looks neutral is visible
    rather than asserted.
    """
    columns = list(rates.columns)
    indices = [columns.index(leg) for leg in legs]

    changes = rates.diff().dropna() * BP
    loadings = rolling_pca_loadings(changes, window, n_factors)

    levels = rates * BP
    records = []
    weights, condition, stale = None, np.nan, False

    for count, (date, matrix) in enumerate(loadings.items()):
        if weights is None or count % reset_every == 0:
            candidate = neutral_weights(matrix, indices)
            # A desk does not put on a hedge ratio of fourteen because the
            # solver said so. When the constraint degenerates, the previous
            # ratios stay in the book until the next reset. Without this, one
            # bad reset in 2011 produces an 840 bp single day move on the
            # 3s7s15s and dominates twenty years of P&L.
            if np.abs(candidate).sum() <= MAX_GROSS_DV01 or weights is None:
                weights = candidate
                condition = conditioning(matrix, indices)
                stale = False
            else:
                stale = True

        row = levels.loc[date].to_numpy()
        records.append(
            {
                "date": date,
                "fly": float(np.dot(weights, row[indices])),
                "naive": float(2 * row[indices[1]] - row[indices[0]] - row[indices[2]]),
                "w_wing1": float(weights[0]),
                "w_belly": float(weights[1]),
                "w_wing2": float(weights[2]),
                "gross": float(np.abs(weights).sum()),
                "condition": condition,
                "weights_stale": stale,
            }
        )

    return pd.DataFrame(records).set_index("date")


def structure_rolldown(
    params: pd.DataFrame,
    weights: pd.DataFrame,
    legs: tuple[float, float, float],
    horizon: float = 0.25,
) -> pd.Series:
    """Mechanical move of the structure over a horizon, on an unchanged curve.

    Each leg ages: a 10 year point held for three months becomes a 9 year 9 month
    point, and on a sloped curve that is a different rate even though nothing
    happened. The structure therefore drifts by

        roll = sum_i w_i * (y(T_i - h) - y(T_i))

    with no financing assumption and no repo model: it is a deterministic
    property of today's curve. This is the carry a rates desk checks before it
    looks at any signal.
    """
    if horizon <= 0 or horizon >= min(legs):
        raise ValueError("horizon must be positive and shorter than the front leg")

    aligned = weights.reindex(params.index).ffill()
    column = [params[name].to_numpy(dtype=float)[:, None] for name in PARAM_NAMES]

    taus = np.asarray(legs, dtype=float)
    now = spot_rate(taus[None, :], *column)
    aged = spot_rate((taus - horizon)[None, :], *column)

    drift = ((aged - now) * aligned.to_numpy()).sum(axis=1) * BP
    return pd.Series(drift, index=params.index)


def constant_weight_moments(
    rates: pd.DataFrame,
    weights: pd.DataFrame,
    legs: tuple[float, float, float],
    window: int = 120,
) -> pd.DataFrame:
    """Level, rolling mean, sigma, deviation and z, all at today's weights.

    The raw level jumps every time the hedge ratios reset, by up to 200 bp, and a
    z-score taken on that series fires on the jump rather than on the market. The
    fix is to score today's structure against what it *would* have been at
    today's weights, which needs no history rebuild: at fixed weights the mean is
    linear and the variance quadratic in the rolling moments of the legs,

        mu    = w . mean(y)
        sigma = sqrt(w' Cov(y) w)

    so both follow from the rolling mean and covariance of the three legs.
    """
    columns = list(legs)
    values = rates[columns] * BP
    aligned = weights.reindex(values.index).ffill()

    means = values.rolling(window).mean()
    weighted_mean = (aligned.to_numpy() * means.to_numpy()).sum(axis=1)

    variance = np.zeros(len(values))
    for i, first in enumerate(columns):
        for j, second in enumerate(columns):
            product = (values[first] * values[second]).rolling(window).mean()
            covariance = product.to_numpy() - means[first].to_numpy() * means[second].to_numpy()
            variance += aligned.to_numpy()[:, i] * aligned.to_numpy()[:, j] * covariance

    level = (aligned.to_numpy() * values.to_numpy()).sum(axis=1)
    sigma = np.sqrt(np.clip(variance, 1e-12, None))

    return pd.DataFrame(
        {"level": level, "mean": weighted_mean, "sigma": sigma,
         "deviation": level - weighted_mean, "z": (level - weighted_mean) / sigma},
        index=values.index,
    )


def constant_weight_zscore(rates, weights, legs, window: int = 120) -> pd.Series:
    """Just the z column of constant_weight_moments."""
    return constant_weight_moments(rates, weights, legs, window)["z"]


def reversion_fraction(half_life_days: float, horizon_days: float) -> float:
    """Share of a deviation that a mean reverting series gives back over a horizon.

    An exponential process closes 1 - 2^(-h/HL) of its gap. With a half life of
    78 days and a three month horizon that is 43 percent, not 100. Comparing a
    full deviation against a full horizon of carry is not comparing like with
    like, and it is how a carry adjustment ends up dominated by the carry term.
    """
    if half_life_days <= 0:
        return 1.0
    if not np.isfinite(half_life_days):
        # No mean reversion means no gap is closed over any horizon. Returning 1
        # here made the carry filter maximally permissive on exactly the
        # structures that never revert.
        return 0.0
    return float(1.0 - 2.0 ** (-horizon_days / half_life_days))


def carry_filter(
    position: pd.Series,
    deviation: pd.Series,
    roll: pd.Series,
    reversion: float,
) -> pd.Series:
    """Drop positions whose expected reversion does not cover the roll against them.

    Expected gain on a held position is

        position * (reversion * deviation - roll)

    because a long profits from the deviation closing and pays away the drift the
    structure earns by ageing. Trades with a negative expectation are declined.
    Framed as a filter rather than as a rescaled signal so the entry and exit
    thresholds stay exactly where they were: retuning them alongside a new term
    would make the comparison meaningless.
    """
    index = position.index
    expected = position * (
        reversion * deviation.reindex(index).fillna(0.0) - roll.reindex(index).fillna(0.0)
    )
    return position.where(expected > 0.0, 0.0)


def position_from_zscore(
    z: pd.Series, entry: float = 1.5, exit_at: float = 0.5
) -> pd.Series:
    """Discrete position with hysteresis, driven by a z-score computed elsewhere."""
    positions = np.zeros(len(z))
    state = 0.0
    for i, value in enumerate(np.nan_to_num(z.to_numpy())):
        if state == 0.0:
            if value > entry:
                state = 1.0
            elif value < -entry:
                state = -1.0
        elif (state > 0 and value < exit_at) or (state < 0 and value > -exit_at):
            state = 0.0
        positions[i] = state
    return pd.Series(positions, index=z.index)


def zscore_signal(
    level: pd.Series, window: int = 120, entry: float = 1.0, cap: float = 3.0
) -> pd.Series:
    """Mean reversion signal from a trailing z-score of the structure.

    Positive when the structure sits above its own recent history, so the
    position profits if it falls back. Mean and standard deviation use data up to
    and including the signal date; the position earns the next day's move.
    """
    mean = level.rolling(window).mean()
    deviation = level.rolling(window).std()
    z = (level - mean) / deviation

    position = z.clip(-cap, cap)
    position[z.abs() < entry] = 0.0
    return position.fillna(0.0)


def structure_change(
    rates: pd.DataFrame, weights: pd.DataFrame, legs: tuple[float, float, float]
) -> pd.Series:
    """Overnight move of the structure at *constant* weights, in basis points.

    This is not the first difference of the structure's level. The hedge ratios
    are re-solved every day, and differencing the level would book the change in
    weights as profit and loss, which is money nobody made: the position held
    overnight is yesterday's weights on today's market.

        change_t = sum_i w_i(t) * (y_i(t+1) - y_i(t))
    """
    moves = rates[list(legs)].diff().shift(-1) * BP  # t -> t+1, aligned on t
    aligned = weights.reindex(moves.index)
    product = (aligned.to_numpy() * moves.to_numpy()).sum(axis=1)
    return pd.Series(product, index=moves.index).dropna()


def half_life(level: pd.Series) -> float:
    """Mean reversion half life in trading days, from an AR(1) fit.

    Regressing the daily change on the previous level gives dx = k*x + noise with
    k negative for a mean reverting series; the half life is -ln(2)/ln(1+k). It
    sets the honest rebalancing frequency: trading a spread that halves its
    distance to fair value in forty days at daily frequency pays forty times the
    spread to capture the same move once.
    """
    lagged = level.shift(1)
    change = (level - lagged).dropna()
    lagged = lagged.reindex(change.index)

    slope = change.cov(lagged) / lagged.var()
    if slope >= 0:
        return float("inf")
    return float(-np.log(2.0) / np.log1p(slope))


def backtest(
    change: pd.Series,
    position: pd.Series,
    gross: pd.Series | None = None,
    cost_bp: float = DEFAULT_COST_BP,
    level: pd.Series | None = None,
) -> pd.DataFrame:
    """Daily P&L of a curve structure held in DV01 units.

    P&L is quoted in basis points of belly DV01: holding one unit while the
    structure falls one basis point earns one unit. Long the structure gains when
    its level falls, hence the negative sign.

    `change` must already be the constant-weight overnight move from
    structure_change. Costs charge the traded change in position scaled by the
    gross DV01, because rebalancing a butterfly trades all three legs.
    """
    if gross is None:
        gross = pd.Series(4.0, index=change.index)
    gross = gross.reindex(change.index).ffill()

    if level is None:
        level = pd.Series(np.nan, index=change.index)

    gross_pnl = -position * change
    turnover = position.diff().abs().fillna(position.abs())

    frame = pd.DataFrame(
        {
            "position": position,
            "level": level,
            "gross_pnl_bp": gross_pnl,
            "cost_bp": turnover * gross * cost_bp / 2.0,
            "turnover": turnover,
        }
    ).dropna(subset=["position", "gross_pnl_bp", "cost_bp"])
    frame["net_pnl_bp"] = frame["gross_pnl_bp"] - frame["cost_bp"]
    frame["equity_bp"] = frame["net_pnl_bp"].cumsum()
    frame["drawdown_bp"] = frame["equity_bp"] - frame["equity_bp"].cummax()
    return frame


def performance(result: pd.DataFrame, label: str = "") -> dict:
    """Headline statistics, including the figures that expose a weak strategy."""
    net = result["net_pnl_bp"]
    active = net[result["position"] != 0]
    volatility = net.std()

    years = len(net) / TRADING_DAYS
    sharpe = (
        float(net.mean() / volatility * np.sqrt(TRADING_DAYS))
        if volatility > 0
        else float("nan")
    )

    return {
        "strategy": label,
        "days": int(len(net)),
        # Sharpe times the square root of the sample length in years. Below about
        # two the strategy is not statistically distinguishable from zero, and a
        # Sharpe quoted without it is only half an answer.
        "t_stat": sharpe * np.sqrt(years),
        "days_in_market": int((result["position"] != 0).sum()),
        "total_bp": float(net.sum()),
        "gross_bp": float(result["gross_pnl_bp"].sum()),
        "cost_bp": float(result["cost_bp"].sum()),
        "sharpe": sharpe,
        "hit_rate": float((active > 0).mean()) if len(active) else float("nan"),
        "max_drawdown_bp": float(result["drawdown_bp"].min()),
        "annual_turnover": float(result["turnover"].sum() / len(net) * TRADING_DAYS),
    }


def hedge_residual(rates: pd.DataFrame, hedge_maturity: float) -> pd.DataFrame:
    """What a duration hedge leaves behind, maturity by maturity.

    A position hedged only for parallel risk keeps its slope and curvature
    exposure. Regressing each maturity's daily change on the hedge instrument
    gives the beta a desk would use; the residual volatility is the risk that
    survives the hedge, and it is not small away from the hedge point.
    """
    changes = rates.diff().dropna() * BP
    hedge = changes[hedge_maturity]

    rows = []
    for maturity in changes.columns:
        target = changes[maturity]
        beta = target.cov(hedge) / hedge.var()
        residual = target - beta * hedge
        rows.append(
            {
                "maturity": maturity,
                "beta_to_hedge": float(beta),
                "total_vol_bp": float(target.std()),
                "residual_vol_bp": float(residual.std()),
                "variance_hedged": float(1.0 - residual.var() / target.var()),
            }
        )
    return pd.DataFrame(rows).set_index("maturity")
