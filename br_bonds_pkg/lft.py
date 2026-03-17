"""
br_bonds/lft.py
───────────────
Pricing and yield-curve tools for Brazilian LFT bonds (Selic-linked floating
rate), ANBIMA BUS/252 convention.

LFT key conventions
--------------------
- Zero-coupon Selic-linked bond; no intermediate cash flows.
- VNA (Valor Nominal Atualizado) = accumulated Selic factor from 2000-07-01,
  starting at R$1000.00. Published daily by ANBIMA.
- PU (R$) = VNA / (1 + r)^(du/252)
  where r = yield (taxa de deságio), usually in [−0.01, +0.02].
- Maturity adjustment: adjust_next if not bizday (same rule as LTN).

Public API
----------
price_lft(ytm, du, vna)
    PU in R$ given yield and VNA.

ytm_lft(pu, du, vna)
    Analytical yield inversion.

fetch_vna_selic(start, end)
    Download daily VNA Selic from BCB API (requires requests).

LFTCurve
    Panel-based Selic yield curve: flat-forward interpolation.
    Supports optional VNA series for PU conversion.
"""

from __future__ import annotations

import warnings
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from bizdays import Calendar

from ._interpolation import interp_yield


# ── standalone pricing ────────────────────────────────────────────────────────

def price_lft(ytm: float, du: float, vna: float) -> float:
    """
    Mark-to-market PU of an LFT in R$, ANBIMA BUS/252 convention.

    Parameters
    ----------
    ytm : yield (taxa de deságio), decimal (e.g. 0.002 for 0.20% p.a.)
    du  : business days to maturity
    vna : VNA Selic on settlement date (R$); base = R$1000 on 2000-07-01

    Returns
    -------
    PU in R$.
    """
    return vna / (1.0 + ytm) ** (du / 252.0)


def ytm_lft(pu: float, du: float, vna: float) -> float:
    """
    Analytical inversion: PU → yield (taxa de deságio).

    Parameters
    ----------
    pu  : dirty price in R$
    du  : business days to maturity
    vna : VNA Selic on settlement date (R$)

    Returns
    -------
    Annual yield (decimal, BUS/252).
    """
    return (vna / pu) ** (252.0 / du) - 1.0


# ── VNA Selic data helper ─────────────────────────────────────────────────────

def fetch_vna_selic(
    start: str = '01/07/2000',
    end: str | None = None,
) -> pd.Series:
    """
    Download daily VNA Selic from BCB API and compute as cumulative product.

    Uses BCB open data series 11 (daily Selic rate, % a.a.).  The VNA is
    reconstructed by compounding the daily factor starting at R$1000.00 on
    2000-07-01 (Tesouro Nacional base date).

    Requires: ``requests`` (pip install requests).

    Parameters
    ----------
    start : first date to request, DD/MM/YYYY format (default '01/07/2000')
    end   : last date, DD/MM/YYYY; None → today

    Returns
    -------
    pd.Series indexed by pd.Timestamp (business days only), values = VNA in R$.

    Notes
    -----
    The series starts at 1000.00 on 2000-07-03 (first bizday on/after 2000-07-01).
    Save the output to a parquet for offline use:
        vna_selic = fetch_vna_selic()
        vna_selic.to_frame('vna').to_parquet('vna_selic.parquet')
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "fetch_vna_selic requires 'requests': pip install requests"
        )

    if end is None:
        from datetime import date
        end = date.today().strftime('%d/%m/%Y')

    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados"
        f"?formato=json&dataInicial={start}&dataFinal={end}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    raw = pd.DataFrame(resp.json())
    raw['data']  = pd.to_datetime(raw['data'], dayfirst=True)
    raw['valor'] = raw['valor'].astype(float)
    raw = raw.sort_values('data').reset_index(drop=True)

    # Convert annualised Selic (% a.a.) to daily compounding factor
    # daily_factor = (1 + annual_rate/100)^(1/252)
    raw['daily_factor'] = (1.0 + raw['valor'] / 100.0) ** (1.0 / 252.0)

    # Cumulative VNA: starts at 1000 on 2000-07-01 (seed on the day before first obs)
    raw['vna'] = 1000.0 * raw['daily_factor'].cumprod()

    # Align: VNA on date t reflects Selic accrual up to and including t-1.
    # Shift forward so that vna.loc[t] = price to be paid on settlement date t.
    raw['vna'] = raw['vna'].shift(1)
    raw = raw.dropna(subset=['vna'])

    return raw.set_index('data')['vna'].rename('vna_selic')


# ── LFTCurve ──────────────────────────────────────────────────────────────────

class LFTCurve:
    """
    Selic yield curve interpolator for LFT bonds, ANBIMA BUS/252 convention.

    LFT has no coupons — the yield is simultaneously the zero rate and the
    par yield.  The curve is a flat-forward (or linear) interpolation of
    observed deságio yields across maturities.

    Parameters
    ----------
    panel : pd.DataFrame
        Required columns: ['date', 'maturity', <yield_col>].
        Yields must be decimal (e.g. 0.002 for 0.20% p.a.).
    cal : Calendar
        ANBIMA business-day calendar.
    method : {'flatfwd', 'linear'}
        Interpolation method. 'flatfwd' is ANBIMA standard.
    yield_col : str
        Column with per-bond yields (default 'yield_base').
    vna_series : pd.Series or None
        Daily VNA Selic indexed by pd.Timestamp.
        Required for pu() and build_series(). See fetch_vna_selic().

    Examples
    --------
    >>> from br_bonds import LFTCurve
    >>> curve = LFTCurve(panel, cal, vna_series=vna_selic)
    >>> curve.ytm(date, 252)          # 1Y deságio yield
    >>> curve.pu(date, 252)           # PU in R$
    >>> curve.zero_curve(date)        # trivial: zero_rate = ytm (no coupons)
    >>> curve.build_series(dates, 252)
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        cal: Calendar,
        method: Literal['flatfwd', 'linear'] = 'flatfwd',
        yield_col: str = 'yield_base',
        vna_series: pd.Series | None = None,
    ) -> None:
        if method not in ('flatfwd', 'linear'):
            raise ValueError("method must be 'flatfwd' or 'linear'")

        self.method    = method
        self.yield_col = yield_col
        self._cal      = cal
        self._vna      = vna_series
        self._index: dict = {}
        self._build(panel, cal, yield_col)

    # ── construction ──────────────────────────────────────────────────────────

    def _build(self, panel: pd.DataFrame, cal: Calendar, yield_col: str) -> None:
        needed  = ['date', 'maturity', yield_col]
        missing = [c for c in needed if c not in panel.columns]
        if missing:
            raise ValueError(f"panel is missing columns: {missing}")

        df = panel[needed].dropna(subset=['date', 'maturity', yield_col]).copy()

        pairs  = df[['date', 'maturity']].drop_duplicates()
        du_map = {}
        for row in pairs.itertuples(index=False):
            if row.maturity > row.date:
                mat_adj = (cal.adjust_next(row.maturity)
                           if not cal.isbizday(row.maturity) else row.maturity)
                du_map[(row.date, row.maturity)] = cal.bizdays(row.date, mat_adj)
            else:
                du_map[(row.date, row.maturity)] = 0
        df['du'] = [du_map[(r.date, r.maturity)] for r in df.itertuples(index=False)]
        df = df[df['du'] > 0]

        for date, grp in df.groupby('date', sort=False):
            grp = grp.sort_values('du')
            du  = grp['du'].to_numpy(dtype=np.float64)
            ytm = grp[yield_col].to_numpy(dtype=np.float64)
            dfs = (1.0 + ytm) ** (-du / 252.0)
            self._index[date] = (du, ytm, dfs)

    # ── public interface ───────────────────────────────────────────────────────

    def ytm(self, date: pd.Timestamp, du_target: int) -> float | None:
        """
        Flat-forward (or linear) interpolated deságio yield at ``du_target`` bdays.
        Returns None if date not in index or du_target out of range.
        """
        entry = self._index.get(date)
        if entry is None:
            return None
        return interp_yield(entry[0], entry[1], entry[2], float(du_target), self.method)

    def pu(self, date: pd.Timestamp, du_target: int) -> float | None:
        """
        Mark-to-market PU in R$.  PU = VNA / (1 + ytm)^(du/252).
        Requires vna_series at construction; returns None if VNA missing.
        """
        y = self.ytm(date, du_target)
        if y is None:
            return None
        if self._vna is None:
            warnings.warn(
                "LFTCurve.pu(): vna_series was not supplied. "
                "Pass vna_series= at construction to enable PU conversion.",
                UserWarning, stacklevel=2,
            )
            return None
        v = self._vna.get(date)
        if v is None:
            return None
        return price_lft(y, float(du_target), v)

    def zero_curve(self, date: pd.Timestamp) -> pd.DataFrame | None:
        """
        Zero (spot) rate curve on ``date``.

        For LFT (zero-coupon) the zero rate equals the YTM at every node —
        no bootstrap required.  Returns the same schema as NTNBCurve and
        PrefixadoCurve for API consistency.

        Returns
        -------
        pd.DataFrame with columns ['du', 'bond_type', 'zero_rate',
                                    'discount_factor'].
        None if date is not in the index.
        """
        entry = self._index.get(date)
        if entry is None:
            return None

        du  = entry[0]
        ytm = entry[1]
        dfs = entry[2]
        zero_rates = dfs ** (-252.0 / du) - 1.0   # = ytm (identity for zero-coupon)

        return pd.DataFrame({
            'du':              du,
            'bond_type':       'LFT',
            'zero_rate':       zero_rates,
            'discount_factor': dfs,
        })

    def build_series(
        self,
        dates: Sequence[pd.Timestamp],
        du_target: int,
    ) -> pd.DataFrame:
        """
        Compute (ytm, vna, pu) for all ``dates`` at fixed ``du_target`` bdays.

        ``vna`` and ``pu`` are NaN when ``vna_series`` was not provided.

        Returns
        -------
        pd.DataFrame indexed by dates with columns ['ytm', 'vna', 'pu'].
        """
        n    = len(dates)
        ytms = np.empty(n, dtype=np.float64)
        du_t = float(du_target)

        for i, d in enumerate(dates):
            entry = self._index.get(d)
            if entry is None:
                ytms[i] = np.nan
                continue
            y = interp_yield(entry[0], entry[1], entry[2], du_t, self.method)
            ytms[i] = y if y is not None else np.nan

        if self._vna is not None:
            vna_vals = np.array([self._vna.get(d, np.nan) for d in dates],
                                dtype=np.float64)
        else:
            vna_vals = np.full(n, np.nan)

        pu_vals = np.where(
            np.isfinite(ytms) & np.isfinite(vna_vals),
            vna_vals / (1.0 + ytms) ** (du_t / 252.0),
            np.nan,
        )

        return pd.DataFrame({
            'ytm': ytms,
            'vna': vna_vals,
            'pu':  pu_vals,
        }, index=pd.DatetimeIndex(dates))

    # ── dunder helpers ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        dates = list(self._index)
        if not dates:
            return "LFTCurve(empty)"
        has_vna = f"vna={'yes' if self._vna is not None else 'no'}"
        return (
            f"LFTCurve(method='{self.method}', "
            f"dates={len(dates)}, "
            f"range={min(dates).date()}…{max(dates).date()}, "
            f"{has_vna})"
        )
