"""
br_bonds/vna.py
───────────────
Daily VNA (Valor Nominal Atualizado) series for NTN-B bonds,
following ANBIMA methodology.

ANBIMA VNA rules
----------------
- Base: VNA = R$ 1,000.00 on July 17, 2000 (first aniversário —
  July 15, 2000 was a Saturday → next ANBIMA bizday = July 17).
- Aniversário dates: 15th of each month, adjusted to next ANBIMA bizday.
- Each aniversário d_n incorporates the IPCA of month M-1, where M is
  the month of d_n:
      ratio_n = SIDRA_index(M-1) / SIDRA_index(M-2)
- Between consecutive aniversários d_{n-1} and d_n (exclusive start,
  inclusive end):
      VNA(t) = VNA(d_{n-1}) × ratio_n ^ (du(d_{n-1}, t) / du(d_{n-1}, d_n))
  where du() counts ANBIMA business days.

Projection for unpublished months
----------------------------------
When the realized IPCA for a given interval is not yet in SIDRA:
  1. Use projected_ipca[month] if provided (% per month, e.g. 0.52
     for 0.52%, indexed by month-start Timestamp).
  2. Otherwise fall back to the previous month's realized ratio
     (ANBIMA default rule for interim periods).

IPCA source
-----------
SIDRA table 1737, variable 2266 (Número-índice, 13 decimal places).
URL: https://apisidra.ibge.gov.br/values/t/1737/n1/all/v/2266/p/all/d/v2266%2013

Public API
----------
fetch_ipca_index()
    Fetch IPCA Número-índice from SIDRA (monthly Series).

build_vna(end_date, ipca_index, projected_ipca, verbose)
    Build daily VNA Series from 2000-07-17 to end_date.
"""

from __future__ import annotations

import requests
import numpy as np
import pandas as pd
from bizdays import Calendar

_CAL          = Calendar.load('ANBIMA')
VNA_BASE      = 1_000.0
VNA_BASE_DATE = pd.Timestamp('2000-07-17')   # July 15 was Saturday → first bizday


# ── IPCA index from SIDRA ─────────────────────────────────────────────────────

def fetch_ipca_index() -> pd.Series:
    """
    Fetch IPCA Número-índice (base: Dec 1993 = 100) from SIDRA table 1737.

    Returns
    -------
    pd.Series
        Monthly Series indexed by month-start Timestamp, values = index
        level with 13 decimal places.
    """
    url = (
        'https://apisidra.ibge.gov.br/values/t/1737/n1/all/v/2266'
        '/p/all/d/v2266%2013'
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    rows = []
    for item in data[1:]:
        period = item['D3C']
        value  = item['V']
        if not value or value in ('-', '...'):
            continue
        rows.append({
            'date' : pd.Timestamp(f"{period[:4]}-{period[4:6]}-01"),
            'index': float(value),
        })

    return (
        pd.DataFrame(rows)
        .set_index('date')['index']
        .sort_index()
    )


# ── aniversário date helpers ──────────────────────────────────────────────────

def _anniv_date(year: int, month: int) -> pd.Timestamp:
    """15th of (year, month) adjusted to next ANBIMA business day."""
    d = pd.Timestamp(year, month, 15)
    return pd.Timestamp(_CAL.adjust_next(d)) if not _CAL.isbizday(d) else d


def _build_anniv_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[pd.Timestamp]:
    """
    Sorted list of aniversário dates from the month of *start* through *end*.
    VNA_BASE_DATE is always prepended as the initial anchor.
    """
    dates = [VNA_BASE_DATE]
    y, m  = start.year, start.month
    while pd.Timestamp(y, m, 1) <= end:
        dates.append(_anniv_date(y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return sorted(set(dates))


# ── VNA construction ──────────────────────────────────────────────────────────

def build_vna(
    end_date: pd.Timestamp | None = None,
    ipca_index: pd.Series | None = None,
    projected_ipca: pd.Series | None = None,
    verbose: bool = True,
) -> pd.Series:
    """
    Build a daily VNA series (ANBIMA methodology) from 2000-07-17 to end_date.

    Parameters
    ----------
    end_date : pd.Timestamp, optional
        Last date of the series. Defaults to today.
    ipca_index : pd.Series, optional
        Pre-fetched SIDRA IPCA index (from ``fetch_ipca_index()``).
        Fetched automatically if None.
    projected_ipca : pd.Series, optional
        Projected monthly IPCA rates (% per month, e.g. 0.52 for 0.52%),
        indexed by month-start Timestamp. Used for months not yet published
        in SIDRA. Falls back to the previous month's realized ratio when
        neither realized nor projected data is available (ANBIMA default).
    verbose : bool
        Print progress messages.

    Returns
    -------
    pd.Series
        Daily VNA in R$, indexed by ANBIMA business day Timestamps.
    """
    if end_date is None:
        end_date = pd.Timestamp.today().normalize()
    if ipca_index is None:
        if verbose:
            print("Fetching IPCA index from SIDRA...")
        ipca_index = fetch_ipca_index()
        if verbose:
            print(
                f"  {len(ipca_index)} monthly obs "
                f"({ipca_index.index[0].date()} to {ipca_index.index[-1].date()})"
            )

    annivs    = _build_anniv_dates(VNA_BASE_DATE, end_date)
    bdays     = pd.DatetimeIndex(_CAL.seq(VNA_BASE_DATE, end_date))
    vna       = pd.Series(np.nan, index=bdays, dtype=float)
    vna.iloc[0] = VNA_BASE                      # anchor: 2000-07-17 = 1000.0

    last_sidra = ipca_index.index[-1]

    for i in range(1, len(annivs)):
        d_prev = annivs[i - 1]
        d_curr = annivs[i]

        # d_curr is the 15th of month M (adjusted).
        # The interval uses IPCA of M-1: ratio = SIDRA(M-1) / SIDRA(M-2).
        _m1        = d_curr - pd.DateOffset(months=1)
        ipca_month = pd.Timestamp(_m1.year, _m1.month, 1)   # M-1
        _m2        = ipca_month - pd.DateOffset(months=1)
        prev_month = pd.Timestamp(_m2.year, _m2.month, 1)   # M-2

        # ── resolve ratio ────────────────────────────────────────────────────
        if ipca_month in ipca_index.index and prev_month in ipca_index.index:
            ratio = ipca_index[ipca_month] / ipca_index[prev_month]

        elif projected_ipca is not None and ipca_month in projected_ipca.index:
            ratio = 1.0 + projected_ipca[ipca_month] / 100.0

        else:
            # ANBIMA fallback: repeat the previous month's realized ratio
            _ps = last_sidra - pd.DateOffset(months=1)
            prev_sidra = pd.Timestamp(_ps.year, _ps.month, 1)
            if last_sidra in ipca_index.index and prev_sidra in ipca_index.index:
                ratio = ipca_index[last_sidra] / ipca_index[prev_sidra]
            else:
                continue

        du_total = _CAL.bizdays(d_prev, d_curr)
        if du_total == 0:
            continue

        vna_prev = vna.get(d_prev)
        if pd.isna(vna_prev):
            continue

        # ── vectorized fill ──────────────────────────────────────────────────
        # days_in_interval are consecutive bizdays d_prev+1 … d_curr,
        # so CAL.bizdays(d_prev, t) = 1, 2, …, du_total in order.
        mask             = (bdays > d_prev) & (bdays <= d_curr)
        days_in_interval = bdays[mask]
        if len(days_in_interval) == 0:
            continue

        du_arr       = np.arange(1, len(days_in_interval) + 1, dtype=float)
        vna[days_in_interval] = vna_prev * (ratio ** (du_arr / du_total))

    return vna.dropna().sort_index()
