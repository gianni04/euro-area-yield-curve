"""Curve relative value study: screen structures, backtest them, draw the charts.

Run from the project root:  python scripts/run_trading.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyc import analysis, charts, loader, strategies  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Benchmark tenors, the points a rates desk actually quotes.
TENORS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]

CANDIDATES = [
    (2.0, 5.0, 10.0),
    (5.0, 10.0, 30.0),
    (3.0, 7.0, 15.0),
    (2.0, 10.0, 30.0),
    (1.0, 3.0, 7.0),
]

HEDGE_TENOR = 10.0

# Holding horizon used to price the carry of a structure, in years.
CARRY_HORIZON = 0.25

REPORT_COLUMNS = [
    "days_in_market",
    "gross_bp",
    "cost_bp",
    "total_bp",
    "sharpe",
    "t_stat",
    "hit_rate",
    "max_drawdown_bp",
    "annual_turnover",
]


def label(legs) -> str:
    return "s".join(f"{int(x)}" if x >= 1 else f"{x:g}" for x in legs) + "s"


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rates = loader.load_rates("SR", "ALL", maturities=TENORS)
    print(f"{len(rates)} curves, {rates.index[0].date()} to {rates.index[-1].date()}\n")

    print("=== Structure screen ===\n")
    built, screen = {}, []
    for legs in CANDIDATES:
        fly = strategies.build_fly(rates, legs)
        weights = fly[["w_wing1", "w_belly", "w_wing2"]]
        gross = float(fly["gross"].mean())
        tradeable = gross <= strategies.MAX_GROSS_DV01
        screen.append(
            {
                "structure": label(legs),
                "mean_gross_dv01": gross,
                "max_gross_dv01": float(fly["gross"].max()),
                "stale_weight_days": float(fly["weights_stale"].mean()),
                "condition_number": float(fly["condition"].mean()),
                "daily_vol_bp": float(
                    strategies.structure_change(rates, weights, legs).std()
                ),
                "half_life_days": strategies.half_life(
                    strategies.constant_weight_moments(rates, weights, legs)["level"]
                ),
                "tradeable": tradeable,
            }
        )
        if tradeable:
            built[legs] = fly

    screen_frame = pd.DataFrame(screen).set_index("structure")
    screen_frame.to_csv(OUTPUT_DIR / "fly_screen.csv")
    print(screen_frame.round(2).to_string())
    print(
        f"\nStructures above {strategies.MAX_GROSS_DV01} gross DV01 are rejected: "
        "their hedge ratios are solver noise, not curve shape."
    )

    print("\n=== Backtest, net of costs ===\n")
    results, rows = {}, []
    for legs, fly in built.items():
        weights = fly[["w_wing1", "w_belly", "w_wing2"]]
        change = strategies.structure_change(rates, weights, legs)

        # Both variants score the same constant-weight z, so the comparison
        # isolates one variable: continuous position against entry/exit bands.
        # Scoring the continuous arm on the raw level instead would fold the
        # monthly reset artefact back into the branch being compared against.
        moments = strategies.constant_weight_moments(rates, weights, legs)
        z = moments["z"]
        for name, position in (
            ("continuous z", z.clip(-3.0, 3.0).where(z.abs() >= 1.0, 0.0)),
            ("bands 1.5/0.5", strategies.position_from_zscore(z)),
        ):
            aligned = position.reindex(change.index).fillna(0.0)
            frame = strategies.backtest(
                change, aligned, fly["gross"], level=fly["fly"]
            )
            key = f"{label(legs)} {name}"
            rows.append(strategies.performance(frame, key))
            if name.startswith("bands"):
                results[key] = frame

    # Every structure that passed the screen, equally weighted. No name is
    # excluded: dropping the two that happened to perform worst, and calling the
    # survivors "the well conditioned ones", would be selection after the fact.
    # It is also false here: 2s10s30s has the best condition number of the five.
    def combine(keys, label):
        weight = 1.0 / len(keys)
        frame = pd.DataFrame(
            {
                column: sum(results[k][column] for k in keys) * weight
                for column in ("gross_pnl_bp", "cost_bp", "net_pnl_bp", "turnover")
            }
        )
        # Number of legs live, so days_in_market and hit rate stay comparable
        # with the single structure rows.
        frame["position"] = sum((results[k]["position"] != 0) for k in keys)
        frame["equity_bp"] = frame["net_pnl_bp"].cumsum()
        frame["drawdown_bp"] = frame["equity_bp"] - frame["equity_bp"].cummax()
        rows.append(strategies.performance(frame, label))
        return frame

    everything = list(results)
    results["portfolio, all 5"] = combine(everything, "portfolio, all 5")

    # Reported second and labelled for what it is: the same portfolio with the
    # worst conditioned structure removed. The threshold was chosen after seeing
    # the results, so this line is a diagnostic, not a result.
    survivors = [k for k in everything if not k.startswith("1s3s7s")]
    results["portfolio, ex-post drop"] = combine(
        survivors, "portfolio, ex-post drop (in-sample)"
    )

    performance = pd.DataFrame(rows).set_index("strategy")
    performance.to_csv(OUTPUT_DIR / "trading_performance.csv")
    print(performance[REPORT_COLUMNS].round(2).to_string())

    correlation = pd.DataFrame(
        {key: results[key]["net_pnl_bp"] for key in everything}
    ).corr()
    print("\ncorrelation of daily P&L between structures:")
    print(correlation.round(2).to_string())

    print("\n=== Carry: roll down of each structure, three month horizon ===\n")
    params_all = loader.load_parameters("ALL")
    carry_rows = []
    for legs, fly in built.items():
        weights = fly[["w_wing1", "w_belly", "w_wing2"]]
        moments = strategies.constant_weight_moments(rates, weights, legs)
        roll = strategies.structure_rolldown(params_all, weights, legs, CARRY_HORIZON)
        life = strategies.half_life(moments["level"])
        share = strategies.reversion_fraction(
            life, CARRY_HORIZON * strategies.TRADING_DAYS
        )

        change = strategies.structure_change(rates, weights, legs)
        base = strategies.position_from_zscore(moments["z"])
        filtered = strategies.carry_filter(base, moments["deviation"], roll, share)

        scores = {}
        for name, position in (("plain", base), ("filtered", filtered)):
            frame = strategies.backtest(
                change, position.reindex(change.index).fillna(0.0), fly["gross"]
            )
            scores[name] = strategies.performance(frame)["sharpe"]

        carry_rows.append(
            {
                "structure": label(legs),
                "median_roll_bp": float(roll.median()),
                "roll_p05_bp": float(roll.quantile(0.05)),
                "roll_p95_bp": float(roll.quantile(0.95)),
                "half_life_days": life,
                "reversion_share": share,
                "sharpe_plain": scores["plain"],
                "sharpe_filtered": scores["filtered"],
            }
        )

    carry_frame = pd.DataFrame(carry_rows).set_index("structure")
    carry_frame.to_csv(OUTPUT_DIR / "carry.csv")
    print(carry_frame.round(3).to_string())
    print(
        "\nRoll is small by construction: a structure built orthogonal to level and\n"
        "slope has little carry, because carry is mostly a function of those very\n"
        "shapes. Filtering on it declines trades that were profitable on average,\n"
        "so it is reported and left off rather than switched on."
    )

    print("\n=== Duration hedge residual ===\n")
    residual = strategies.hedge_residual(rates, HEDGE_TENOR)
    print(residual.round(3).to_string())

    print("\n=== Figures ===\n")
    pca = analysis.curve_pca(maturities=TENORS)
    params = loader.load_parameters("ALL")
    latest = tuple(params.iloc[-1][name] for name in loader.PARAMS)
    headline = next(iter(built))
    headline_z = strategies.constant_weight_zscore(
        rates, built[headline][["w_wing1", "w_belly", "w_wing2"]], headline
    )

    written = [
        charts.plot_curve_history(rates, FIGURE_DIR / "01_curve_history.png"),
        charts.plot_curve_heatmap(rates, FIGURE_DIR / "02_rate_surface.png"),
        charts.plot_pca_loadings(
            pca["loadings"], pca["factor_shares"], FIGURE_DIR / "03_factors.png"
        ),
        charts.plot_spread(
            analysis.sovereign_spread(maturities=TENORS)[10.0],
            FIGURE_DIR / "04_sovereign_spread.png",
        ),
        charts.plot_structure(
            headline_z,
            strategies.position_from_zscore(headline_z),
            FIGURE_DIR / "05_butterfly.png",
            label(headline),
        ),
        charts.plot_equity(results, FIGURE_DIR / "06_performance.png"),
        charts.plot_cost_bridge(performance, FIGURE_DIR / "07_cost_bridge.png"),
        charts.plot_carry_roll(
            analysis.rolldown_table(latest), FIGURE_DIR / "08_carry_roll.png"
        ),
        charts.plot_hedge_residual(
            residual, HEDGE_TENOR, FIGURE_DIR / "09_hedge_residual.png"
        ),
    ]
    for path in written:
        print(f"  {path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
