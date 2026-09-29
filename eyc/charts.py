"""Figures for the curve and the trading study.

P&L charts show the drawdown under the equity line.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#1a1a1a"
MUTED = "#8a8a8a"
ACCENT = "#c0392b"
BLUE = "#2c5f8a"
GREEN = "#2d7a4f"
GRID = "#e3e3e3"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
    }
)

MATURITY_TICKS = [0.25, 1, 2, 5, 10, 30]
MATURITY_LABELS = ["3M", "1Y", "2Y", "5Y", "10Y", "30Y"]


def _finish(fig, path: Path, title: str, subtitle: str = "") -> Path:
    fig.suptitle(title, fontsize=12, fontweight="bold", x=0.01, ha="left", y=1.0)
    if subtitle:
        fig.text(0.01, 0.955, subtitle, fontsize=8.5, color=MUTED, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.90 if subtitle else 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_curve_history(rates: pd.DataFrame, path: Path, every: int = 250) -> Path:
    """Every curve in the sample, faded, with the most recent one picked out."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    taus = np.asarray(rates.columns, dtype=float)

    sample = rates.iloc[::every]
    shades = plt.cm.viridis(np.linspace(0.15, 0.9, len(sample)))
    for colour, (_, row) in zip(shades, sample.iterrows()):
        ax.plot(taus, row.to_numpy(), color=colour, linewidth=0.9, alpha=0.75)

    ax.plot(taus, rates.iloc[-1].to_numpy(), color=ACCENT, linewidth=2.2, label="latest")
    ax.axhline(0, color=INK, linewidth=0.8, linestyle=":")
    ax.set_xlabel("maturity (years)")
    ax.set_ylabel("zero coupon rate (%)")
    ax.legend(loc="lower right")

    return _finish(
        fig,
        path,
        "Euro area zero coupon curve",
        f"one curve per {every} business days, {sample.index[0].date()} to "
        f"{rates.index[-1].date()}; colour runs dark (old) to light (recent)",
    )


def plot_curve_heatmap(rates: pd.DataFrame, path: Path) -> Path:
    """The whole history as a surface: rate level by maturity and date."""
    fig, ax = plt.subplots(figsize=(8, 4))
    taus = np.asarray(rates.columns, dtype=float)
    dates = mdates.date2num(rates.index.to_pydatetime())

    limit = float(np.nanmax(np.abs(rates.to_numpy())))
    mesh = ax.pcolormesh(
        dates, taus, rates.to_numpy().T, cmap="RdBu_r",
        vmin=-limit, vmax=limit, shading="auto",
    )
    ax.set_yscale("log")
    ax.set_yticks(MATURITY_TICKS)
    ax.set_yticklabels(MATURITY_LABELS)
    ax.set_ylabel("maturity")
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, label="rate (%)", pad=0.015)

    return _finish(
        fig,
        path,
        "Rate surface",
        "blue is negative: the euro area printed sub-zero yields for years, "
        "which is data and not a defect",
    )


def plot_pca_loadings(loadings: pd.DataFrame, shares, path: Path) -> Path:
    """The three factors, with the share of daily variance each explains."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    taus = np.asarray(loadings.index, dtype=float)

    for colour, name, share in zip([BLUE, ACCENT, GREEN], loadings.columns, shares):
        ax.plot(taus, loadings[name].to_numpy(), color=colour, linewidth=1.8,
                label=f"{name} ({share:.1%})")

    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks(MATURITY_TICKS)
    ax.set_xticklabels(MATURITY_LABELS)
    ax.set_xlabel("maturity")
    ax.set_ylabel("loading")
    ax.legend(loc="best")

    return _finish(
        fig,
        path,
        "Curve factors",
        "principal components of daily spot changes; level, slope and curvature "
        "are read off the shapes, not assumed",
    )


def plot_spread(spread: pd.Series, path: Path) -> Path:
    """Sovereign credit premium through the crisis and out the other side."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.fill_between(spread.index, 0, spread.to_numpy(), color=BLUE, alpha=0.18)
    ax.plot(spread.index, spread.to_numpy(), color=BLUE, linewidth=1.1)

    peak = spread.idxmax()
    ax.scatter([peak], [spread.max()], color=ACCENT, zorder=5, s=22)
    ax.annotate(
        f"{spread.max():.0f} bp\n{peak.date()}",
        xy=(peak, spread.max()), xytext=(12, -6), textcoords="offset points",
        fontsize=8, color=ACCENT,
    )
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_ylabel("basis points")

    return _finish(
        fig,
        path,
        "Sovereign credit premium, 10 year",
        "all euro area issuers minus AAA only, same model and same day, so term "
        "structure effects net out",
    )


def plot_structure(
    z: pd.Series, position: pd.Series, path: Path, label: str,
    entry: float = 1.5, exit_at: float = 0.5,
) -> Path:
    """The traded signal with its bands, and when the strategy was in the market.

    The signal, not the raw level: hedge ratios reset monthly and the raw level
    jumps by up to 200 bp when they do, which is not a market move and is not
    what the strategy responds to.
    """
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    top.plot(z.index, z.to_numpy(), color=BLUE, linewidth=0.8)
    for level, style, colour in (
        (entry, "--", ACCENT), (-entry, "--", ACCENT),
        (exit_at, ":", MUTED), (-exit_at, ":", MUTED),
    ):
        top.axhline(level, color=colour, linestyle=style, linewidth=1.0)
    top.axhline(0, color=INK, linewidth=0.8)
    top.set_ylabel("z-score at constant weights")
    top.text(0.005, 0.96, f"entry +/-{entry}, exit +/-{exit_at}", transform=top.transAxes,
             fontsize=8, color=ACCENT, va="top")

    active = position.reindex(z.index).fillna(0.0).to_numpy()
    bottom.fill_between(z.index, 0, active, where=active > 0, color=GREEN,
                        alpha=0.55, step="pre", label="long the fly")
    bottom.fill_between(z.index, 0, active, where=active < 0, color=ACCENT,
                        alpha=0.55, step="pre", label="short the fly")
    bottom.set_ylabel("position")
    bottom.legend(loc="upper left", ncol=2)

    return _finish(
        fig,
        path,
        f"{label} butterfly signal",
        "scored against what the structure would have been at today's hedge "
        "ratios, so a monthly reset does not read as a trading opportunity",
    )


def plot_equity(results: dict, path: Path) -> Path:
    """Cumulative P&L with the drawdown underneath, never without it."""
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    palette = [BLUE, ACCENT, GREEN, "#8e44ad", "#d4820a", "#0f7f8f"]

    for i, (label, frame) in enumerate(results.items()):
        headline = "portfolio" in label.lower()
        colour = INK if headline else palette[i % len(palette)]
        top.plot(
            frame.index, frame["equity_bp"], color=colour,
            linewidth=2.4 if headline else 1.1,
            alpha=1.0 if headline else 0.75, label=label, zorder=5 if headline else 2,
        )
        bottom.fill_between(
            frame.index, 0, frame["drawdown_bp"], color=colour,
            alpha=0.35 if headline else 0.16,
        )

    top.axhline(0, color=INK, linewidth=0.8)
    top.set_ylabel("cumulative P&L\n(bp of belly DV01)")
    top.legend(loc="upper left")
    bottom.set_ylabel("drawdown (bp)")

    return _finish(
        fig,
        path,
        "Strategy performance, net of costs",
        "quoted per unit of belly DV01; costs charged on gross DV01 traded",
    )


def plot_cost_bridge(performance: pd.DataFrame, path: Path) -> Path:
    """Gross edge against the cost of harvesting it, side by side."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    index = np.arange(len(performance))
    width = 0.38

    ax.bar(index - width / 2, performance["gross_bp"], width, color=BLUE, label="gross")
    ax.bar(index + width / 2, -performance["cost_bp"], width, color=ACCENT,
           label="costs")
    ax.plot(index, performance["total_bp"], "o", color=INK, markersize=6, label="net")

    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(index)
    ax.set_xticklabels(
        [name.replace(" 1.5/0.5", "") for name in performance.index],
        fontsize=7.5, rotation=35, ha="right",
    )
    ax.set_ylabel("basis points")
    ax.legend(loc="best")

    return _finish(
        fig,
        path,
        "Where the edge goes",
        "a curve trade lives or dies on turnover: the gross number is not the result",
    )


def plot_carry_roll(table: pd.DataFrame, path: Path) -> Path:
    """Carry and roll down across the curve, stacked."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    index = np.arange(len(table))

    ax.bar(index, table["carry_bp"], 0.55, color=BLUE, label="carry")
    ax.bar(index, table["rolldown_bp"], 0.55, bottom=table["carry_bp"], color=GREEN,
           label="roll down")
    ax.plot(index, table["total_return_bp"], "o-", color=INK, linewidth=1.2,
            markersize=5, label="total")

    ax.set_xticks(index)
    ax.set_xticklabels([f"{int(m)}Y" for m in table.index])
    ax.set_ylabel("basis points, 3 month horizon")
    ax.legend(loc="best")

    return _finish(
        fig,
        path,
        "Carry and roll down",
        "what the position pays if the curve does not move at all, before any "
        "forecast",
    )


def plot_key_rate_durations(table: pd.DataFrame, path: Path) -> Path:
    """Where each bond's risk actually sits along the curve."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    index = np.arange(len(table))
    bottom = np.zeros(len(table))
    shades = plt.cm.viridis(np.linspace(0.15, 0.85, table.shape[1]))

    for colour, tenor in zip(shades, table.columns):
        values = table[tenor].to_numpy()
        ax.bar(index, values, 0.6, bottom=bottom, color=colour,
               label=f"{tenor:g}Y bucket")
        bottom += values

    ax.set_xticks(index)
    ax.set_xticklabels([f"{int(m)}Y bond" for m in table.index])
    ax.set_ylabel("key rate duration (years)")
    ax.legend(loc="upper left", ncol=2)

    return _finish(
        fig,
        path,
        "Where the risk sits",
        "the bars sum to the effective duration by construction; a 30Y bond keeps "
        "almost all of its risk in the 30Y bucket, so a 10Y hedge misses it",
    )


def plot_hedge_residual(residual: pd.DataFrame, hedge: float, path: Path) -> Path:
    """Risk that survives a duration hedge, maturity by maturity."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    taus = np.asarray(residual.index, dtype=float)

    ax.plot(taus, residual["total_vol_bp"], color=MUTED, linewidth=1.6,
            label="unhedged daily vol")
    ax.plot(taus, residual["residual_vol_bp"], color=ACCENT, linewidth=1.8,
            label=f"after {hedge:.0f}Y duration hedge")
    ax.fill_between(taus, residual["residual_vol_bp"], residual["total_vol_bp"],
                    color=BLUE, alpha=0.15, label="risk removed")

    ax.set_xscale("log")
    ax.set_xticks(MATURITY_TICKS)
    ax.set_xticklabels(MATURITY_LABELS)
    ax.set_xlabel("maturity")
    ax.set_ylabel("daily volatility (bp)")
    ax.legend(loc="best")

    return _finish(
        fig,
        path,
        "What a duration hedge leaves behind",
        "a hedge neutralises the level factor only; slope and curvature survive, "
        "and away from the hedge point they are most of the risk",
    )
