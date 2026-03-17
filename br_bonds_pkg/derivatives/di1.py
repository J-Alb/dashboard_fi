"""
br_bonds/derivatives/di1.py
────────────────────────────
B3 DI1 futures — pricing functions and flat-forward term structure.

Pricing convention
------------------
DI1 price = 100 000 / (1 + CDI_fut)^(du/252)

All rates are in decimal (0.105 = 10.5% p.a., BUS/252 ANBIMA calendar).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .._interpolation import flatfwd_df


@dataclass
class DI1Contract:
    """
    A single B3 DI1 futures contract.

    Attributes
    ----------
    rate : float — implied CDI rate (decimal, ANBIMA BUS/252)
    du   : int   — business days from settlement to expiry
    """
    rate: float
    du: int

    @property
    def price(self) -> float:
        """Contract price in R$ (notional 100 000)."""
        return di1_price(self.rate, self.du)

    @property
    def dv01(self) -> float:
        """DV01 in R$ per 1bp per contract."""
        return di1_dv01(self.rate, self.du)


def di1_price(rate: float, du: int) -> float:
    """
    DI1 futures price.

    Parameters
    ----------
    rate : implied CDI rate (decimal, BUS/252)
    du   : business days to expiry

    Returns
    -------
    float — price in R$ (notional 100 000)
    """
    return 100_000.0 / (1.0 + rate) ** (du / 252.0)


def di1_dv01(rate: float, du: int) -> float:
    """
    DI1 DV01 — change in contract price per 1bp change in implied rate.

    DV01 = price × (du/252) / (1 + rate) × 1e-4

    Parameters
    ----------
    rate : implied CDI rate (decimal, BUS/252)
    du   : business days to expiry

    Returns
    -------
    float — DV01 in R$ per bp per contract (positive)
    """
    p = di1_price(rate, du)
    return p * (du / 252.0) / (1.0 + rate) * 1e-4


class DI1Curve:
    """
    Flat-forward DI1 term structure for a single reference date.

    Rates are in decimal (0.105 = 10.5% p.a., BUS/252 ANBIMA).
    Interface mirrors PrefixadoCurve for compatibility with future DI1 work.

    Parameters
    ----------
    refdate    : pd.Timestamp — pricing reference date
    du_verts   : array-like of int — business days to each vertex (sorted, du=1 first)
    rate_verts : array-like of float — spot CDI rates (decimal) at each vertex
    """

    def __init__(self, refdate, du_verts, rate_verts) -> None:
        self.refdate  = pd.Timestamp(refdate)
        self._du      = np.asarray(du_verts,   dtype=float)
        self._rates   = np.asarray(rate_verts, dtype=float)
        self._df      = (1.0 + self._rates) ** (-self._du / 252.0)

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def overnight(self) -> float:
        """Overnight CDI rate (decimal) — du=1 vertex."""
        return float(self._rates[0])

    @property
    def du_min(self) -> int:
        return int(self._du[0])

    @property
    def du_max(self) -> int:
        return int(self._du[-1])

    # ── interpolation ─────────────────────────────────────────────────────────

    def ytm(self, du: int) -> float | None:
        """Flat-forward interpolated spot CDI rate at du (decimal). None if out of range."""
        du_f = float(du)
        if du_f < self._du[0] or du_f > self._du[-1]:
            return None
        df = flatfwd_df(self._du, self._df, du_f)
        return float(df ** (-252.0 / du_f) - 1.0)

    def forward(self, du1: int, du2: int) -> float | None:
        """Period forward rate from du1 to du2 (decimal). None if out of range."""
        if du1 >= du2:
            raise ValueError("du1 must be < du2")
        df1 = flatfwd_df(self._du, self._df, float(du1))
        df2 = flatfwd_df(self._du, self._df, float(du2))
        span = float(du2 - du1)
        return float((df1 / df2) ** (252.0 / span) - 1.0)

    def df_at(self, du: int) -> float:
        """Discount factor at du via flat-forward interpolation."""
        return flatfwd_df(self._du, self._df, float(du))

    # ── output frames ─────────────────────────────────────────────────────────

    def spot_curve(self, step: int = 21) -> pd.DataFrame:
        """Spot CDI rates on a du grid (step ≈ 1 month). Rates in decimal."""
        grid  = np.arange(self.du_min, self.du_max + 1, step, dtype=float)
        rates = np.array([self.ytm(int(d)) for d in grid])
        return pd.DataFrame({'du': grid.astype(int), 'rate': rates})

    def forward_curve(self) -> pd.DataFrame:
        """Forward rates between consecutive DI1 vertices. Rates in decimal."""
        rows = []
        for i in range(1, len(self._du)):
            du1, du2 = int(self._du[i - 1]), int(self._du[i])
            rows.append({'du_start': du1, 'du_end': du2, 'forward': self.forward(du1, du2)})
        return pd.DataFrame(rows)

    # ── constructor from panel ────────────────────────────────────────────────

    @classmethod
    def from_data(
        cls,
        df: pd.DataFrame,
        overnight_pct: float,
        refdate,
        cal,
    ) -> 'DI1Curve':
        """
        Build a DI1Curve from a B3-style panel.

        Parameters
        ----------
        df            : DataFrame with columns [maturity_date, rate] where rate is % p.a.
        overnight_pct : CDI overnight rate in % p.a. (e.g. 10.5 for 10.5%)
        refdate       : reference date (date or Timestamp)
        cal           : bizdays.Calendar
        """
        refdate_ts = pd.Timestamp(refdate)
        df = df.copy()
        df['du'] = df['maturity_date'].apply(lambda m: cal.bizdays(refdate_ts, m))
        df = df[df['du'] > 1].sort_values('du').reset_index(drop=True)

        du_verts   = np.concatenate([[1],                      df['du'].values])
        rate_verts = np.concatenate([[overnight_pct / 100.0], df['rate'].values / 100.0])

        return cls(refdate_ts, du_verts, rate_verts)

    def __repr__(self) -> str:
        return (
            f"DI1Curve(refdate={self.refdate.date()}, "
            f"vertices={len(self._du)}, "
            f"du=[{self.du_min}..{self.du_max}], "
            f"overnight={self.overnight * 100:.2f}%)"
        )
