"""Sovereign credit premium, curve factors, bond risk, carry and roll down.

Run from the project root:  python scripts/run_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyc import analysis, bonds, charts, loader  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

PORTFOLIO = [
    bonds.Bond(2.0, 2.5),
    bonds.Bond(5.0, 3.0),
    bonds.Bond(10.0, 3.0),
    bonds.Bond(30.0, 2.0),
]

REPORT_MATURITIES = [0.25, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    params_frame = loader.load_parameters("ALL")
    latest_date = params_frame.index[-1]
    latest = tuple(params_frame.iloc[-1][name] for name in loader.PARAMS)
    print(f"Curve date: {latest_date.date()}  (all euro area issuers)\n")

    print("Sovereign credit premium, all issuers minus AAA (bp)\n")
    spread = analysis.sovereign_spread()
    spread.to_csv(OUTPUT_DIR / "sovereign_spread.csv")
    ten_year = spread[10.0]
    print(
        f"10Y spread: mean {ten_year.mean():.1f}, min {ten_year.min():.1f}, "
        f"max {ten_year.max():.1f} on {ten_year.idxmax().date()}, "
        f"latest {ten_year.iloc[-1]:.1f}"
    )
    print(f"negative readings: {int((ten_year < 0).sum())} of {len(ten_year)}")
    print("\nannual average, 10Y:")
    print(ten_year.resample("YE").mean().round(1).to_string())

    print("\nCurve factors, daily spot changes\n")
    pca = analysis.curve_pca()
    pca["loadings"].to_csv(OUTPUT_DIR / "pca_loadings.csv")
    shares = ", ".join(f"{s:.1%}" for s in pca["factor_shares"])
    print(f"three factors explain {pca['explained_in_sample']:.2%} in sample")
    print(f"                      {pca['explained_out_of_sample']:.2%} out of sample")
    print(f"individual shares: {shares}")
    print(f"train {pca['train_dates']} days, holdout {pca['test_dates']} days")
    print("\nloadings:")
    print(pca["loadings"].loc[REPORT_MATURITIES].round(3).to_string())

    print("\nBond risk on the latest curve\n")
    risk = pd.DataFrame([bonds.risk_report(b, latest) for b in PORTFOLIO])
    risk.to_csv(OUTPUT_DIR / "bond_risk.csv", index=False)
    print(risk.round(4).to_string(index=False))

    print("\nKey rate durations: where the risk sits\n")
    krd = pd.DataFrame(
        {bond.maturity: bonds.key_rate_durations(bond, latest) for bond in PORTFOLIO}
    ).T
    krd.index.name = "maturity"
    krd.to_csv(OUTPUT_DIR / "key_rate_durations.csv")
    print(krd.round(3).to_string())

    check = pd.Series(
        {bond.maturity: bonds.effective_duration(bond, latest) for bond in PORTFOLIO}
    )
    gap = float((krd.sum(axis=1) - check).abs().max())
    print(f"\nsum of buckets vs effective duration, max gap: {gap:.2e} years")
    print(
        "A 10Y hedge on the 30Y bond neutralises only its 10Y bucket, "
        f"{krd.loc[30.0, 10.0]:.2f} of {check[30.0]:.2f} years."
    )

    charts.plot_key_rate_durations(
        krd, OUTPUT_DIR / "figures" / "10_key_rate_durations.png"
    )

    print("\nCarry and roll down, three month horizon, curve unchanged\n")
    roll = analysis.rolldown_table(latest)
    roll.to_csv(OUTPUT_DIR / "rolldown.csv")
    print(roll.round(2).to_string())

    print(f"\nWrote CSV output to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
