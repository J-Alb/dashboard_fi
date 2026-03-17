"""
br_bonds/derivatives/ddi.py
────────────────────────────
B3 DDI futures — Futuro de Cupom Cambial (implied dollar coupon).

Pricing convention
------------------
DDI uses simple (linear) interest on an actual/360 basis:

    PU = 100 000 / (1 + rate * dc/360)

where:
  rate = annual dollar coupon rate (decimal, linear act/360)
  dc   = calendar days from settlement to expiry

This differs from DI1/DAP (compound BUS/252).

Rates are in decimal (0.045 = 4.5% p.a.).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type

import numpy as np
import pandas as pd
from bizdays import Calendar


@dataclass
class DDIContract:
    """
    A single B3 DDI futures contract.

    Attributes
    ----------
    rate : float — implied dollar coupon rate (decimal, linear act/360)
    dc   : int   — calendar days from settlement to expiry
    """
    rate: float
    dc: int

    @property
    def price(self) -> float:
        """Contract price in R$ (notional 100 000)."""
        return ddi_price(self.rate, self.dc)

    @property
    def dv01(self) -> float:
        """DV01 in R$ per 1bp per contract."""
        return ddi_dv01(self.rate, self.dc)


def ddi_price(rate: float, dc: int) -> float:
    """
    DDI futures price (simple interest, act/360).

    Parameters
    ----------
    rate : implied dollar coupon rate (decimal, linear act/360)
    dc   : calendar days to expiry

    Returns
    -------
    float — price in R$ (notional 100 000)
    """
    return 100_000.0 / (1.0 + rate * dc / 360.0)


def ddi_dv01(rate: float, dc: int) -> float:
    """
    DDI DV01 — price sensitivity to a 1bp increase in rate.

    Returns
    -------
    float — R$ change per 1bp per contract (always negative)
    """
    denom = 1.0 + rate * dc / 360.0
    return -(dc / 360.0) * 100_000.0 / (denom ** 2) * 1e-4


def ddi_rate(price: float, dc: int) -> float:
    """
    Invert DDI price → implied dollar coupon rate.

    Parameters
    ----------
    price : contract price in R$ (notional 100 000)
    dc    : calendar days to expiry

    Returns
    -------
    float — annual dollar coupon rate (decimal, linear act/360)
    """
    return (100_000.0 / price - 1.0) * 360.0 / dc


def ddi_from_df(
    df: pd.DataFrame,
    date_col: str = 'DATA',
    mat_col: str = 'VENCIMENTO',
    rate_col: str = 'AJUSTE_ATUAL',
    cal: Calendar | None = None,
) -> pd.DataFrame:
    """
    Enrich a DDI settlement DataFrame with calendar days to maturity,
    price, and DV01.

    Parameters
    ----------
    df       : DataFrame with columns [date_col, mat_col, rate_col].
               rate_col values are expected in % p.a. (e.g. 4.50 for 4.50%).
    date_col : settlement date column (default 'DATA').
    mat_col  : maturity date column (default 'VENCIMENTO').
    rate_col : settlement rate column in % p.a. (default 'AJUSTE_ATUAL').
    cal      : ANBIMA Calendar (unused for dc, kept for API consistency).

    Returns
    -------
    Input DataFrame with added columns: dc, rate (decimal), price, dv01.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out[mat_col]  = pd.to_datetime(out[mat_col])

    # DDI uses calendar days (actual/360), not business days
    out['dc']    = (out[mat_col] - out[date_col]).dt.days
    out['rate']  = out[rate_col] / 100.0
    out['price'] = out.apply(lambda r: ddi_price(r['rate'], r['dc']), axis=1)
    out['dv01']  = out.apply(lambda r: ddi_dv01(r['rate'], r['dc']),  axis=1)
    return out
