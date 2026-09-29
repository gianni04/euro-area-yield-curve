"""US Treasury curve data, used as a benchmark for the calibration.

Two sources:
- US Treasury constant maturity par yields (on-the-run, fixed tenors)
- Fed Gurkaynak-Sack-Wright dataset: Svensson parameters fitted to off-the-run
  bonds, plus the curves they generate

Fitting the Fed's own par yields tests whether the optimiser recovers known
parameters. Fitting Treasury yields is a different problem, so the parameters
are not expected to match the Fed's.
"""

from __future__ import annotations

import io
import urllib.request

import pandas as pd

from .loader import CACHE_DIR, USER_AGENT

TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
    "?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
)

GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"

# The Fed file carries nine lines of preamble before the header row.
GSW_SKIPROWS = 9

TREASURY_TENORS = {
    "1 Mo": 1 / 12,
    "2 Mo": 2 / 12,
    "3 Mo": 0.25,
    "4 Mo": 4 / 12,
    "6 Mo": 0.5,
    "1 Yr": 1.0,
    "2 Yr": 2.0,
    "3 Yr": 3.0,
    "5 Yr": 5.0,
    "7 Yr": 7.0,
    "10 Yr": 10.0,
    "20 Yr": 20.0,
    "30 Yr": 30.0,
}

GSW_PARAMS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]


def _download(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read().decode("utf-8", errors="replace")


def treasury_par_yields(years, use_cache: bool = True) -> pd.DataFrame:
    """Constant maturity par yields in percent, columns labelled in years."""
    years = list(years)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"ust_par_{min(years)}_{max(years)}.csv"
    if use_cache and cache.exists():
        cached = pd.read_csv(cache, index_col=0, parse_dates=True)
        # A round trip through CSV turns the float maturities into strings.
        cached.columns = cached.columns.astype(float)
        return cached

    frames = []
    for year in years:
        raw = pd.read_csv(io.StringIO(_download(TREASURY_URL.format(year=year))))
        raw["Date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y")
        keep = [c for c in TREASURY_TENORS if c in raw.columns]
        frame = raw.set_index("Date")[keep]
        frame.columns = [TREASURY_TENORS[c] for c in keep]
        frames.append(frame)

    combined = pd.concat(frames).sort_index()
    combined.index.name = "date"
    combined.to_csv(cache)
    return combined


def gsw(start: str = "2015-01-01", use_cache: bool = True) -> pd.DataFrame:
    """Fed Svensson parameters, zero yields and par yields, from `start` onward.

    Zero yields (SVENYxx) are continuously compounded; par yields (SVENPYxx) are
    coupon-equivalent, meaning a semiannual coupon schedule.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"gsw_{start}.csv"
    if use_cache and cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    raw = pd.read_csv(
        io.StringIO(_download(GSW_URL)),
        skiprows=GSW_SKIPROWS,
        na_values=["NA"],
        low_memory=False,
    )
    raw["Date"] = pd.to_datetime(raw["Date"])
    frame = raw.set_index("Date").sort_index().loc[start:]

    wanted = GSW_PARAMS + [
        column
        for column in frame.columns
        if column.startswith(("SVENY", "SVENPY")) and column[-2:].isdigit()
    ]
    frame = frame[wanted].astype(float)
    frame.index.name = "date"
    frame.to_csv(cache)
    return frame


def gsw_curve(frame: pd.DataFrame, prefix: str, maturities=range(1, 31)):
    """Extract SVENY or SVENPY columns as a frame indexed by maturity in years."""
    columns = {float(m): f"{prefix}{m:02d}" for m in maturities}
    available = {tau: name for tau, name in columns.items() if name in frame.columns}
    out = frame[list(available.values())].copy()
    out.columns = list(available.keys())
    return out
