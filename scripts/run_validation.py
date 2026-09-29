"""Rebuild every published ECB curve from raw parameters and report the error.

Run from the project root:  python scripts/run_validation.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eyc import loader, validation  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

TOLERANCE_BP = validation.TOLERANCE_BP

# Three dates on which the ECB's own published AAA series are mutually
# inconsistent. The all-issuers curve reconstructs exactly on the same dates, and
# the surrounding business days are exact to machine precision, so these are data
# findings rather than implementation defects.
#
#   2020-07-15  spot reconstructs to 0.000000 bp, but the published forward is
#               off by up to 12.6 bp at 30Y. Both tau parameters collapse onto
#               each other (1.83 vs 1.98), which makes the Svensson factors
#               collinear and the fit degenerate.
#   2022-06-23  the spot curve itself disagrees, by up to 5.6 bp at 30Y. Same
#               degenerate signature: tau1 1.29 vs tau2 1.45, beta2 +13.9
#               against beta3 -13.4.
#   2007-05-23  a milder version, 0.007 bp on spot and 0.014 bp on the forward.
#               Not explained by published precision, which carries 14 or more
#               significant digits on that date.
KNOWN_ECB_INCONSISTENCIES = {"2007-05-23", "2020-07-15", "2022-06-23"}

COLUMNS = [
    "dates",
    "maturities",
    "observations",
    "max_abs_bp",
    "rmse_bp",
    "dates_above_tolerance",
    "breach_dates",
    "seconds",
]


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    summaries = []
    failures = []

    for curve in ("AAA", "ALL"):
        for kind in ("SR", "IF", "PY"):
            started = time.perf_counter()
            errors, summary = validation.validate(kind, curve=curve)
            summary["seconds"] = round(time.perf_counter() - started, 1)
            summaries.append(summary)
            errors.to_csv(OUTPUT_DIR / f"errors_{curve}_{kind}.csv")

            per_date = errors.abs().max(axis=1)
            unexplained = [
                str(d)[:10]
                for d in per_date[per_date > TOLERANCE_BP].index
                if str(d)[:10] not in KNOWN_ECB_INCONSISTENCIES
            ]
            if unexplained:
                failures.append((summary["series"], unexplained[:5]))

    table = pd.DataFrame(summaries).set_index("series")
    pd.set_option("display.width", 200)
    print("\n=== Reconstruction vs published ECB rates (errors in bp) ===\n")
    print(
        table[COLUMNS].to_string(
            formatters={
                "max_abs_bp": "{:.2e}".format,
                "rmse_bp": "{:.2e}".format,
                "worst_maturity": "{:.2f}".format,
            }
        )
    )
    table.to_csv(OUTPUT_DIR / "validation_summary.csv")

    print("\n=== Internal invariants ===\n")
    for curve in ("AAA", "ALL"):
        checks = validation.check_invariants(loader.load_parameters(curve=curve))
        print(f"{curve}: {checks}")

    if failures:
        print(f"\nFAILED: unexplained breaches above {TOLERANCE_BP} bp")
        for series, dates in failures:
            print(f"  {series}: {dates}")
        return 1

    print(
        f"\nAll curves reconstructed within {TOLERANCE_BP} bp, except on "
        f"{sorted(KNOWN_ECB_INCONSISTENCIES)} where the ECB's own AAA "
        "parameters and rates disagree."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
