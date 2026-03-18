"""
br_bonds/derivatives/_common.py
────────────────────────────────
Shared helpers for compound BUS/252 futures pricing (DI1, DAP).

DI1 and DAP use the same zero-coupon pricing formula:
    PU = 100 000 / (1 + rate)^(du/252)
and the same DV01 magnitude — only the sign convention differs
(DI1 reports positive sensitivity; DAP reports negative).
"""

from __future__ import annotations


def _zcbond_price(rate: float, du: int) -> float:
    """
    Zero-coupon futures price (BUS/252 compound, notional 100 000).

    Parameters
    ----------
    rate : implied rate (decimal, BUS/252)
    du   : business days to expiry

    Returns
    -------
    float — price in R$
    """
    return 100_000.0 / (1.0 + rate) ** (du / 252.0)


def _zcbond_dv01_magnitude(rate: float, du: int) -> float:
    """
    DV01 magnitude — absolute price change per 1 bp in implied rate.

    Both DI1 and DAP share this formula; callers apply their own sign.

    Returns
    -------
    float — positive R$ per bp per contract
    """
    p = _zcbond_price(rate, du)
    return p * (du / 252.0) / (1.0 + rate) * 1e-4
