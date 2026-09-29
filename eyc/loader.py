"""ECB Data Portal client for the euro area yield curve dataset (YC).

The dataset has, for every TARGET business day since 2004-09-06, both the
Svensson parameters and the published rates, which allows an exact
reconstruction check.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://data-api.ecb.europa.eu/service/data/YC"

# The portal rejects requests without a browser user agent.
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

# G_N_A covers issuers rated AAA only; G_N_C covers all issuers, all ratings.
# Their difference is the euro area sovereign credit premium.
CURVES = {"AAA": "G_N_A", "ALL": "G_N_C"}

# Svensson, continuous compounding, yield error minimisation.
MODEL = "SV_C_YM"

PARAMS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]

FIRST_DATE = "2004-09-06"


def default_maturities() -> list[float]:
    """Published maturities: 3 and 6 months, then every 3 months out to 30 years."""
    taus = [0.25, 0.5, 0.75]
    taus += [y + q / 4 for y in range(1, 30) for q in range(4)]
    taus += [30.0]
    return taus


MATURITIES = default_maturities()


def maturity_code(tau: float) -> str:
    """Encode a maturity in years as an ECB series suffix (e.g. 1.25 -> 1Y3M)."""
    months = round(tau * 12)
    years, rem = divmod(months, 12)
    if years and rem:
        return f"{years}Y{rem}M"
    if years:
        return f"{years}Y"
    return f"{rem}M"


def rate_codes(kind: str, maturities=None) -> list[str]:
    """Series codes for spot (SR), instantaneous forward (IF) or par yield (PY)."""
    if kind not in {"SR", "IF", "PY"}:
        raise ValueError(f"unknown rate kind: {kind}")
    taus = MATURITIES if maturities is None else maturities
    return [f"{kind}_{maturity_code(t)}" for t in taus]


def _fetch_raw(series_key: str, start: str, end: str | None, attempts: int = 4) -> str:
    """One SDMX request, retried on transport failure with linear backoff.

    The portal drops connections intermittently. Without retries a single flaky
    request kills a run that has already spent minutes downloading, which is a
    poor trade for the few seconds a retry costs.
    """
    url = f"{BASE_URL}/{series_key}?format=csvdata&startPeriod={start}"
    if end:
        url += f"&endPeriod={end}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)
    raise RuntimeError("unreachable")


def _cache_path(curve: str, items: list[str], start: str, end: str | None) -> Path:
    head = "-".join(items[:2])
    tag = f"{curve}_{head}_{len(items)}_{start}_{end or 'latest'}"
    return CACHE_DIR / f"{tag}.csv"


# The SDMX endpoint accepts multiple codes in one key, but very long keys are
# rejected by the gateway, so requests are split.
CHUNK_SIZE = 25


def fetch(
    items: list[str],
    curve: str = "ALL",
    start: str = FIRST_DATE,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download one or more YC series and return them as a date-indexed frame.

    Columns are the requested item codes; rows are TARGET business days. Missing
    observations are left as NaN rather than filled, so gaps stay visible to the
    validation layer instead of being silently interpolated.
    """
    if not items:
        raise ValueError("items must not be empty")
    if curve not in CURVES:
        raise ValueError(f"curve must be one of {sorted(CURVES)}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(curve, items, start, end)
    if use_cache and cache_file.exists():
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    if len(items) > CHUNK_SIZE:
        chunks = [
            _fetch_one(items[i : i + CHUNK_SIZE], curve, start, end)
            for i in range(0, len(items), CHUNK_SIZE)
        ]
        frame = pd.concat(chunks, axis=1).reindex(columns=items)
        frame.to_csv(cache_file)
        return frame

    frame = _fetch_one(items, curve, start, end)
    frame.to_csv(cache_file)
    return frame


def _fetch_one(
    items: list[str], curve: str, start: str, end: str | None
) -> pd.DataFrame:
    """Fetch a single SDMX request whose key holds at most CHUNK_SIZE codes."""
    series_key = f"B.U2.EUR.4F.{CURVES[curve]}.{MODEL}.{'+'.join(items)}"
    try:
        payload = _fetch_raw(series_key, start, end)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"ECB request failed ({exc.code}) for {series_key}") from exc

    raw = pd.read_csv(io.StringIO(payload), low_memory=False)
    if raw.empty:
        raise RuntimeError(f"ECB returned no observations for {series_key}")

    frame = raw.pivot_table(
        index="TIME_PERIOD",
        columns="DATA_TYPE_FM",
        values="OBS_VALUE",
        aggfunc="first",
    )
    missing = [item for item in items if item not in frame.columns]
    if missing:
        raise RuntimeError(f"ECB did not return series: {missing}")

    frame = frame.reindex(columns=items).sort_index()
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "date"
    return frame


def load_parameters(curve: str = "ALL", **kwargs) -> pd.DataFrame:
    """Daily Svensson parameters: BETA0..BETA3, TAU1, TAU2."""
    return fetch(PARAMS, curve=curve, **kwargs)


def load_rates(kind: str, curve: str = "ALL", maturities=None, **kwargs) -> pd.DataFrame:
    """Daily published rates, with float maturities in years as column labels."""
    taus = list(MATURITIES if maturities is None else maturities)
    frame = fetch(rate_codes(kind, taus), curve=curve, **kwargs)
    frame.columns = taus
    return frame
