"""
br_bonds/derivatives/copom.py
──────────────────────────────
Implied COPOM decisions extracted from the DI1 futures curve.

Given a DI1Curve, CopomCurve fits a vector of meeting-by-meeting rate changes
(in basis points) whose cumulative sum, when compounded daily, best replicates
the market DI1 forward structure. A smoothness penalty prevents erratic paths.

Rates convention
----------------
DI1Curve stores all rates in decimal (0.105 = 10.5%).
CopomCurve.decisions returns:
  - bps         : int-rounded basis-point change per meeting
  - bps_cum     : cumulative bps from overnight
  - cdi_pct     : implied CDI level in % p.a.  (overnight + bps_cum / 100)
  - selic_pct   : implied Selic = CDI + 0.10% (BCB convention)

Fetch utilities
---------------
fetch_di1()              — live B3 DI1 quotes (requires internet)
fetch_overnight(refdate) — CDI overnight from BCB SGS series 12 (requires bcb lib)
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.request
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from bizdays import Calendar

from .di1 import DI1Curve

# ── COPOM meeting schedule ─────────────────────────────────────────────────────
# Update annually. Dates sourced from BCB official calendar.

COPOM_DATES: list[pd.Timestamp] = sorted([
    pd.Timestamp(datetime.strptime(d, "%d/%m/%Y").date()) for d in [
        "04/12/2030", "16/10/2030", "28/08/2030", "17/07/2030", "29/05/2030",
        "17/04/2030", "27/02/2030", "16/01/2030",
        "05/12/2029", "17/10/2029", "29/08/2029", "18/07/2029", "30/05/2029",
        "18/04/2029", "28/02/2029", "17/01/2029",
        "06/12/2028", "18/10/2028", "30/08/2028", "19/07/2028", "31/05/2028",
        "19/04/2028", "01/03/2028", "19/01/2028",
        "08/12/2027", "20/10/2027", "01/09/2027", "21/07/2027", "02/06/2027",
        "21/04/2027", "03/03/2027", "20/01/2027",
        "09/12/2026", "04/11/2026", "16/09/2026", "05/08/2026", "17/06/2026",
        "29/04/2026", "18/03/2026", "28/01/2026",
        "10/12/2025", "05/11/2025", "17/09/2025", "30/07/2025", "19/06/2025",
        "07/05/2025", "19/03/2025", "29/01/2025",
        "11/12/2024", "06/11/2024", "18/09/2024", "30/07/2024", "18/06/2024",
        "07/05/2024", "19/03/2024", "30/01/2024",
        "12/12/2023", "31/10/2023", "19/09/2023", "01/08/2023", "20/06/2023",
        "02/05/2023", "21/03/2023", "31/01/2023",
    ]
])

# ── fetch utilities ────────────────────────────────────────────────────────────

def _get_previous_business_day(cal: Calendar) -> datetime.date:
    today = datetime.today()
    if cal.isbizday(today):
        return pd.to_datetime(cal.adjust_previous(today - timedelta(days=1))).date()
    return pd.to_datetime(cal.adjust_previous(today)).date()


def _get_price(s: dict):
    cur = s["SctyQtn"].get("curPrc")
    if cur is not None:
        return cur
    buy  = s.get("buyOffer",  {}).get("price")
    sell = s.get("sellOffer", {}).get("price")
    if buy is not None and sell is not None:
        return round((buy + sell) / 2, 4)
    return buy or sell or None


def fetch_di1() -> pd.DataFrame:
    """
    Fetch live DI1 quotes from B3.

    Returns
    -------
    DataFrame with columns:
        maturity_date : pd.Timestamp
        rate          : float — implied CDI rate in % p.a.
        open_contracts: int
    """
    url = "https://cotacao.b3.com.br/mds/api/v1/DerivativeQuotation/DI1"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())

    futures = sorted(
        [s for s in data["Scty"] if s["asset"]["code"] == "DI1" and s["mkt"]["cd"] == "FUT"],
        key=lambda x: x["asset"]["AsstSummry"]["mtrtyCode"],
    )

    rows = []
    for s in futures:
        price = _get_price(s)
        if price is None:
            continue
        rows.append({
            "maturity_date" : pd.Timestamp(s["asset"]["AsstSummry"]["mtrtyCode"]),
            "rate"          : float(price),
            "open_contracts": s["asset"]["AsstSummry"].get("opnCtrcts", np.nan),
        })

    return pd.DataFrame(rows)


def fetch_overnight(refdate=None) -> float:
    """
    Fetch CDI overnight rate from BCB SGS series 12.

    Returns
    -------
    float — annualised CDI in % p.a.  (e.g. 10.5 for 10.5%)

    Requires the ``bcb`` package (pip install python-bcb).
    """
    try:
        from bcb import sgs
    except ImportError as e:
        raise ImportError("Install python-bcb to fetch overnight: pip install python-bcb") from e

    df_cdi    = sgs.get({'cdi': 12}, last=5)
    cdi_daily = float(df_cdi['cdi'].iloc[-1])
    cdi_anual = ((1 + cdi_daily / 100) ** 252 - 1) * 100
    return cdi_anual


# ── internal helpers ───────────────────────────────────────────────────────────

def _flatfwd_batch(du_verts: np.ndarray, df_verts: np.ndarray, du_grid: np.ndarray) -> np.ndarray:
    """
    Vectorised flat-forward discount factors for a du grid.

    du_grid must be within [du_verts[0], du_verts[-1]]; values outside are
    clipped to the nearest segment (mirrors flatfwd_df boundary behaviour).
    """
    idx    = np.searchsorted(du_verts, du_grid, side='right') - 1
    idx    = np.clip(idx, 0, len(du_verts) - 2)
    du1    = du_verts[idx];  du2 = du_verts[idx + 1]
    df1    = df_verts[idx];  df2 = df_verts[idx + 1]
    w      = (du_grid - du1) / (du2 - du1)
    return df1 * (df2 / df1) ** w


def _objective(
    C: np.ndarray,
    du_verts: np.ndarray,
    df_verts: np.ndarray,
    overnight: float,
    bdays_meeting: np.ndarray,
    du_max: int,
    lamb: float,
) -> float:
    """
    Least-squares distance between market DI1 accumulation and COPOM model.

    Parameters
    ----------
    C             : decision vector (decimal, e.g. 0.0025 for +25 bps)
    du_verts      : DI1Curve business-day vertices
    df_verts      : DI1Curve discount factors
    overnight     : CDI overnight rate (decimal)
    bdays_meeting : business days to each COPOM meeting from refdate
    du_max        : horizon (business days) for the comparison
    lamb          : smoothness penalty weight
    """
    C        = np.asarray(C)
    j_idx    = np.searchsorted(bdays_meeting, np.arange(1, du_max), side='right')
    C_cumsum = np.insert(np.cumsum(C), 0, 0.0)

    # model: daily CDI path implied by COPOM decisions
    futures     = np.empty(du_max)
    futures[0]  = overnight
    futures[1:] = overnight + C_cumsum[j_idx]
    prod_model  = np.cumprod((1.0 + futures) ** (1.0 / 252.0))

    # market: accumulation implied by DI1 flat-forward curve
    du_grid     = np.arange(1, du_max + 1, dtype=float)
    df_market   = _flatfwd_batch(du_verts, df_verts, du_grid)
    prod_market = 1.0 / df_market

    # smoothness penalty — stronger in the mid-curve (logistic weights)
    first_diff = np.diff(C, prepend=C[0])
    x          = np.linspace(0, 1, len(C))
    weights    = 2.0 / (1.0 + np.exp(-12.0 * (x - 0.45)))
    penalty    = lamb * np.sum(weights * first_diff ** 2)

    return float(np.sum((prod_market - prod_model) ** 2) + penalty)


# ── CopomCurve ─────────────────────────────────────────────────────────────────

class CopomCurve:
    """
    Implied COPOM decisions extracted from a DI1Curve.

    Fits a meeting-by-meeting rate-change vector (bps) whose cumulative sum,
    when compounded daily, best replicates the flat-forward DI1 structure.

    Usage
    -----
    >>> cal   = Calendar.load('ANBIMA')
    >>> curve = DI1Curve.from_data(df_di1, overnight_pct, refdate, cal)
    >>> copom = CopomCurve(curve, COPOM_DATES, cal).fit()
    >>> copom.decisions
    >>>
    >>> # One-shot from live B3 data:
    >>> copom = CopomCurve.from_b3(COPOM_DATES, cal).fit()
    >>> copom.print_table()
    """

    MAX_MEETINGS = 44  # cap to avoid sparse long-end overfitting

    def __init__(
        self,
        di1_curve: DI1Curve,
        meeting_dates,
        cal: Calendar,
        lamb: float = 1e-2,
    ) -> None:
        self._di1  = di1_curve
        self._cal  = cal
        self._lamb = lamb

        refdate  = di1_curve.refdate
        meetings = pd.DatetimeIndex(sorted(meeting_dates))
        meetings = meetings[meetings >= refdate]
        if len(meetings) > self.MAX_MEETINGS:
            meetings = meetings[: self.MAX_MEETINGS]
        self._meetings   = meetings
        self._decisions_df: pd.DataFrame | None = None

    # ── fitting ───────────────────────────────────────────────────────────────

    def fit(self) -> 'CopomCurve':
        """Run L-BFGS-B optimisation and store implied decisions. Returns self."""
        refdate       = self._di1.refdate
        bdays_meeting = np.array(
            [self._cal.bizdays(refdate, m) for m in self._meetings], dtype=int
        )

        # horizon: first DI1 vertex at or beyond last meeting
        last_du    = int(bdays_meeting[-1])
        candidates = self._di1._du[self._di1._du >= last_du]
        du_max     = int(candidates[0]) if len(candidates) else self._di1.du_max

        C0     = np.zeros(len(self._meetings))
        bounds = [(-0.02, 0.02)] * len(C0)  # ±200 bps per meeting
        result = minimize(
            _objective,
            C0,
            args=(self._di1._du, self._di1._df,
                  self._di1.overnight, bdays_meeting, du_max, self._lamb),
            method='L-BFGS-B',
            bounds=bounds,
        )
        if not result.success:
            warnings.warn(f"CopomCurve: optimisation did not converge — {result.message}")

        C        = result.x                        # decimal
        bps      = np.round(C * 1e4, 2)           # basis points
        bps_cum  = np.cumsum(bps)
        cdi_pct  = self._di1.overnight * 100 + bps_cum / 100
        selic_pct = cdi_pct + 0.10

        self._decisions_df = pd.DataFrame({
            'refdate'     : refdate,
            'meeting_date': self._meetings,
            'bps'         : bps,
            'bps_cum'     : bps_cum,
            'cdi_pct'     : cdi_pct,
            'selic_pct'   : selic_pct,
        }).reset_index(drop=True)

        return self

    # ── output ────────────────────────────────────────────────────────────────

    @property
    def decisions(self) -> pd.DataFrame:
        """Implied COPOM decisions. Columns: meeting_date, bps, bps_cum, cdi_pct, selic_pct."""
        if self._decisions_df is None:
            raise RuntimeError("Call .fit() first")
        return self._decisions_df

    def print_table(self) -> None:
        """Print a formatted summary table."""
        df       = self.decisions
        refdate  = self._di1.refdate.date()
        overnight = self._di1.overnight * 100
        print(f"\nPRICING COPOM  |  refdate: {refdate}  |  CDI base: {overnight:.2f}% a.a.")
        print(f"{'Reunião':<14} {'Decisão (bps)':>14} {'CDI impl. (%)':>14} {'Selic impl. (%)':>16}")
        print("-" * 62)
        for _, row in df.iterrows():
            print(
                f"{str(row['meeting_date'].date()):<14} "
                f"{row['bps']:>14.2f} "
                f"{row['cdi_pct']:>14.2f} "
                f"{row['selic_pct']:>16.2f}"
            )

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_b3(
        cls,
        meeting_dates=None,
        cal: Calendar | None = None,
        lamb: float = 1e-2,
    ) -> 'CopomCurve':
        """
        Fetch live DI1 data from B3 + CDI overnight from BCB and return an
        unfitted CopomCurve. Call .fit() to run the optimisation.

        Parameters
        ----------
        meeting_dates : array-like of dates — defaults to COPOM_DATES
        cal           : bizdays.Calendar — defaults to ANBIMA
        lamb          : smoothness penalty
        """
        if cal is None:
            cal = Calendar.load('ANBIMA')
        if meeting_dates is None:
            meeting_dates = COPOM_DATES

        refdate   = _get_previous_business_day(cal)
        df_di1    = fetch_di1()
        overnight = fetch_overnight(refdate)
        curve     = DI1Curve.from_data(df_di1, overnight, refdate, cal)

        return cls(curve, meeting_dates, cal, lamb)

    def __repr__(self) -> str:
        fitted = self._decisions_df is not None
        return (
            f"CopomCurve(refdate={self._di1.refdate.date()}, "
            f"meetings={len(self._meetings)}, "
            f"lamb={self._lamb}, fitted={fitted})"
        )
