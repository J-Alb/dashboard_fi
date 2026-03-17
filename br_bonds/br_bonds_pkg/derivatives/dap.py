"""
br_bonds/derivatives/dap.py
────────────────────────────
B3 DAP futures — Futuro de Cupom de IPCA (implied real rate).

Pricing convention
------------------
Same as DI1, BUS/252 ANBIMA calendar:

    PU = 100 000 / (1 + rate)^(du/252)

where rate is the annual real (IPCA coupon) rate in decimal.
Maturities fall on the 15th of each month, adjusted to next ANBIMA bizday
(same dates as NTN-B aniversários).

Rates are in decimal (0.065 = 6.5% p.a.).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from bizdays import Calendar


@dataclass
class DAPContract:
    """
    A single B3 DAP futures contract.

    Attributes
    ----------
    rate : float — implied IPCA coupon rate (decimal, BUS/252)
    du   : int   — business days from settlement to expiry
    """
    rate: float
    du: int

    @property
    def price(self) -> float:
        """Contract price in R$ (notional 100 000)."""
        return dap_price(self.rate, self.du)

    @property
    def dv01(self) -> float:
        """DV01 in R$ per 1bp per contract."""
        return dap_dv01(self.rate, self.du)


def dap_price(rate: float, du: int) -> float:
    """
    DAP futures price.

    Parameters
    ----------
    rate : implied IPCA coupon rate (decimal, BUS/252)
    du   : business days to expiry

    Returns
    -------
    float — price in R$ (notional 100 000)
    """
    return 100_000.0 / (1.0 + rate) ** (du / 252.0)


def dap_dv01(rate: float, du: int) -> float:
    """
    DAP DV01 — price sensitivity to a 1bp increase in rate.

    Returns
    -------
    float — R$ change per 1bp per contract (always negative)
    """
    return -(du / 252.0) * dap_price(rate, du) / (1.0 + rate) * 1e-4


def dap_rate(price: float, du: int) -> float:
    """
    Invert DAP price → implied IPCA coupon rate.

    Parameters
    ----------
    price : contract price in R$ (notional 100 000)
    du    : business days to expiry

    Returns
    -------
    float — annual real rate (decimal, BUS/252)
    """
    return (100_000.0 / price) ** (252.0 / du) - 1.0


def dap_from_df(
    df: pd.DataFrame,
    date_col: str = 'DATA',
    mat_col: str = 'VENCIMENTO',
    rate_col: str = 'AJUSTE_ATUAL',
    cal: Calendar | None = None,
) -> pd.DataFrame:
    """
    Enrich a DAP settlement DataFrame with business days to maturity,
    price, and DV01.

    Parameters
    ----------
    df       : DataFrame with columns [date_col, mat_col, rate_col].
               rate_col values are expected in % p.a. (e.g. 6.50 for 6.50%).
    date_col : settlement date column (default 'DATA').
    mat_col  : maturity date column (default 'VENCIMENTO').
    rate_col : settlement rate column in % p.a. (default 'AJUSTE_ATUAL').
    cal      : ANBIMA Calendar (loaded automatically if None).

    Returns
    -------
    Input DataFrame with added columns: du, rate (decimal), price, dv01.
    """
    if cal is None:
        cal = Calendar.load('ANBIMA')

    out = df.copy()
    out[date_col]  = pd.to_datetime(out[date_col])
    out[mat_col]   = pd.to_datetime(out[mat_col])

    def _mat_adj(mat: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(cal.adjust_next(mat)) if not cal.isbizday(mat) else mat

    out['du']    = out.apply(
        lambda r: cal.bizdays(r[date_col], _mat_adj(r[mat_col])), axis=1
    )
    out['rate']  = out[rate_col] / 100.0
    out['price'] = out.apply(lambda r: dap_price(r['rate'], r['du']), axis=1)
    out['dv01']  = out.apply(lambda r: dap_dv01(r['rate'], r['du']),  axis=1)
    return out
