"""
br_bonds/analytics.py
─────────────────────
Duration, total-return, and constant-duration index tools for prefixado bonds.

Public API
----------
bond_duration(ytm, du, coupon, face, freq)
    Macaulay duration, modified duration, and DV01 for a hypothetical bond.

add_duration(cm, du_target)
    Annotate a constant-maturity DataFrame with mac_dur / mod_dur / dv01 columns.

cm_tri(prices, ytms)
    NaN-aware total return index for a constant-maturity rolling position.
    Includes daily accrual (1+ytm)^(1/252) to capture carry.

ret_index(prices, ytms, rf_rates, cash_gaps)
    Cumulative total return (or excess return when rf_rates supplied) index,
    rebased to 100. Includes daily accrual on the bond's own yield.
    cash_gaps=True (default): NaN days are treated as cash at rf, making
    strategies with different gap frequencies directly comparable.

du_for_mod_dur(date, curve, dur_target, coupon, face, freq)
    Solve for the maturity (in business days) at which a hypothetical bond
    achieves ``dur_target`` modified duration on ``date``.

build_cd_series(dates, curve, dur_target)
    Build a daily constant-duration price series by rebalancing maturity
    every day so that modified duration ≈ dur_target.

Notes
-----
Constant maturity ≠ constant duration:
  - Constant maturity fixes du_target — duration drifts with the yield level.
  - Constant duration solves du* each day so that mod_dur(ytm(du*), du*) =
    D_target. Duration is approximately preserved; maturity floats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .prefixado import price_ltn, price_ntnf


# ── duration and DV01 ─────────────────────────────────────────────────────────

def bond_duration(
    ytm: float,
    du: int,
    coupon: float = 0.10,
    face: float = 1000.0,
    freq: int = 2,
) -> tuple[float, float, float]:
    """
    Macaulay duration, modified duration, and DV01 for a hypothetical bond.

    Uses uniform 252//freq bday coupon spacing (price_ntnf convention).

    Parameters
    ----------
    ytm    : annual yield (decimal, ANBIMA BUS/252)
    du     : business days to maturity
    coupon : annual coupon rate (decimal); 0.0 for LTN
    face   : face value (default 1000)
    freq   : coupon payments per year (default 2)

    Returns
    -------
    (mac_dur, mod_dur, dv01)
        mac_dur : Macaulay duration in business years (= du/252 for LTN)
        mod_dur : modified duration = mac_dur / (1 + ytm/freq)  [years]
        dv01    : mod_dur × price × 1e-4  [R$ per bp per face unit]
    """
    if coupon == 0.0:
        mac = du / 252.0
        mod = mac / (1.0 + ytm)
        p   = price_ltn(ytm, du, face)
    else:
        step = 252 // freq
        c    = face * ((1.0 + coupon) ** (1.0 / freq) - 1.0)
        cf_du = np.arange(step, du + step, step, dtype=np.float64)
        cf_du = cf_du[cf_du <= du + 0.5]
        if len(cf_du) == 0 or cf_du[-1] < du:
            cf_du = np.append(cf_du, float(du))
        cashflows      = np.full(len(cf_du), c)
        cashflows[-1] += face
        t  = cf_du / 252.0
        pv = cashflows / (1.0 + ytm) ** t
        p  = pv.sum()
        mac = (t * pv).sum() / p
        mod = mac / (1.0 + ytm / freq)
    dv01 = mod * p * 1e-4
    return mac, mod, dv01


def add_duration(cm: pd.DataFrame, du_target: int) -> pd.DataFrame:
    """
    Annotate a constant-maturity DataFrame with duration and DV01 columns.

    Adds columns ['mac_dur', 'mod_dur', 'dv01'] in-place and returns the df.

    Parameters
    ----------
    cm        : DataFrame with column 'ytm' (as returned by build_series)
    du_target : constant business-day maturity used when building cm
    """
    rows = [
        bond_duration(y, du_target) if not np.isnan(y) else (np.nan, np.nan, np.nan)
        for y in cm['ytm'].values
    ]
    cm[['mac_dur', 'mod_dur', 'dv01']] = pd.DataFrame(rows, index=cm.index)
    return cm


# ── total return index (constant maturity) ────────────────────────────────────

def cm_tri(prices: np.ndarray, ytms: np.ndarray) -> np.ndarray:
    """
    NaN-aware total return index for a constant-maturity rolling position.

    Daily total return:
        TR[t] = (P_t / P_{t-1}) × (1 + ytm_{t-1})^(1/252)

    The accrual term captures the daily carry earned on the hypothetical bond
    before it is rolled back to du_target. Without it, the index earns zero
    return when yields are stable, understating performance.

    Gap handling: at re-entry after NaN, price return is computed vs the last
    valid price; accrual uses ytm[t-1] if valid, else 1.0.

    Parameters
    ----------
    prices : 1-D array of bond prices (may contain NaN)
    ytms   : 1-D array of annual YTMs aligned with prices (decimal)

    Returns
    -------
    np.ndarray — TRI rebased to 100 at the first valid observation.
    """
    prices = np.asarray(prices, dtype=float)
    ytms   = np.asarray(ytms,   dtype=float)
    tri    = np.full_like(prices, np.nan)
    valid  = ~np.isnan(prices)
    if not np.any(valid):
        return tri
    fi = int(np.argmax(valid))
    tri[fi] = 100.0
    prev_p, prev_t = prices[fi], 100.0
    for i in range(fi + 1, len(prices)):
        if valid[i]:
            accrual = (1.0 + ytms[i - 1]) ** (1.0 / 252.0) if not np.isnan(ytms[i - 1]) else 1.0
            tri[i]  = prev_t * (prices[i] / prev_p) * accrual
            prev_p, prev_t = prices[i], tri[i]
    return tri


# ── total / excess return index ───────────────────────────────────────────────

def ret_index(
    prices: np.ndarray,
    ytms: np.ndarray,
    rf_rates: np.ndarray | None = None,
    cash_gaps: bool = True,
) -> np.ndarray:
    """
    Cumulative return index, rebased to 100.

    Total return per day (including accrual at the bond's own yield):
        TR[t] = (P_t / P_{t-1}) × (1 + ytm_{t-1})^(1/252)

    If rf_rates supplied → excess return index:
        ER[t] = TR[t] / (1 + rf_{t-1})^(1/252)

    Without rf_rates → absolute total return index (TRI).

    The (1+ytm)^(1/252) accrual captures the daily carry embedded in the
    bond yield. Without it, the index earns zero return when yields are
    stable, which understates performance vs the risk-free rate.

    Gap handling (cash_gaps=True, default):
        On NaN days (no active position), the strategy is assumed to hold
        cash at the risk-free rate. TRI accrues rf; ERI stays flat (tr=1).
        This makes strategies with different gap frequencies directly
        comparable, because both accumulate the same rf denominator.
        Set cash_gaps=False to revert to the old behaviour (skip NaN days).

    Parameters
    ----------
    prices    : 1-D array of bond prices (may contain NaN)
    ytms      : 1-D array of annual YTMs aligned with prices (decimal)
    rf_rates  : 1-D array of risk-free annual rates aligned with prices (optional)
    cash_gaps : if True (default), NaN days earn rf in TRI and are flat in ERI.
                if False, NaN days are skipped (index carries forward unchanged).

    Returns
    -------
    np.ndarray — return index rebased to 100 at the first valid observation.
    """
    prices = np.asarray(prices, dtype=float)
    ytms   = np.asarray(ytms,   dtype=float)
    rf     = np.asarray(rf_rates, dtype=float) if rf_rates is not None else None
    idx    = np.full_like(prices, np.nan)
    n      = len(prices)

    fi = next((i for i in range(n) if not np.isnan(prices[i])), None)
    if fi is None:
        return idx

    idx[fi] = 100.0
    prev    = 100.0

    for i in range(fi + 1, n):
        inactive = np.isnan(prices[i]) or np.isnan(ytms[i - 1])

        if inactive:
            if cash_gaps and rf is not None and not np.isnan(rf[i - 1]):
                # TRI: earn rf; ERI: flat (tr = 1 after rf/rf cancellation)
                prev   = prev * (1.0 + rf[i - 1]) ** (1.0 / 252.0)
                idx[i] = prev
            else:
                idx[i] = prev
            continue

        accrual   = (1.0 + ytms[i - 1]) ** (1.0 / 252.0)
        price_ret = prices[i] / prices[i - 1] if not np.isnan(prices[i - 1]) else 1.0
        tr        = price_ret * accrual
        if rf is not None and not np.isnan(rf[i - 1]):
            tr /= (1.0 + rf[i - 1]) ** (1.0 / 252.0)
        idx[i] = prev * tr
        prev   = idx[i]

    return idx


# ── constant-duration solver ──────────────────────────────────────────────────

def du_for_mod_dur(
    date,
    curve,
    dur_target: float,
    coupon: float = 0.10,
    face: float = 1000.0,
    freq: int = 2,
) -> float:
    """
    Solve for du* such that mod_dur(curve.ytm(date, du*), du*) = dur_target.

    Uses the curve's actual interpolable range on ``date`` as the brentq
    bracket. Returns NaN if the target is outside the achievable range
    or if the date is not in the curve index.

    Parameters
    ----------
    date       : pricing date (must be in curve._index)
    curve      : PrefixadoCurve instance
    dur_target : target modified duration (business years)
    coupon     : coupon rate used in duration calculation (default 0.10)
    face       : face value (default 1000)
    freq       : payments per year (default 2)
    """
    try:
        du_arr = curve._index[date][0]
    except KeyError:
        return np.nan

    du_min, du_max = int(du_arr[0]) + 1, int(du_arr[-1])

    def obj(du):
        ytm = curve.ytm(date, int(round(du)))
        if ytm is None or np.isnan(ytm):
            raise ValueError('outside interpolable range')
        _, mod, _ = bond_duration(ytm, int(round(du)), coupon, face, freq)
        return mod - dur_target

    try:
        v_lo = obj(du_min)
        v_hi = obj(du_max)
        if v_lo > 0 or v_hi < 0:   # target outside achievable range on this date
            return np.nan
        return float(brentq(obj, du_min, du_max, xtol=0.5))
    except Exception:
        return np.nan


def build_cd_series(
    dates,
    curve,
    dur_target: float,
    coupon: float = 0.10,
    face: float = 1000.0,
    freq: int = 2,
) -> pd.DataFrame:
    """
    Build a daily constant-duration price series.

    Each day, solves for the maturity du* such that mod_dur(ytm(du*), du*) =
    dur_target, then prices a hypothetical NTN-F at that maturity. When no
    solution exists (date not in curve, or target outside range), the row
    is filled with NaN.

    Parameters
    ----------
    dates      : iterable of pd.Timestamp (business days)
    curve      : PrefixadoCurve instance
    dur_target : target modified duration in years
    coupon     : coupon rate for hypothetical bond (default 0.10)
    face       : face value (default 1000)
    freq       : payments per year (default 2)

    Returns
    -------
    pd.DataFrame indexed by date with columns ['du', 'ytm', 'price', 'mod_dur'].
    """
    records = []
    for date in dates:
        du_f = du_for_mod_dur(date, curve, dur_target, coupon, face, freq)
        if np.isnan(du_f):
            records.append((date, np.nan, np.nan, np.nan, np.nan))
        else:
            du  = int(round(du_f))
            ytm = curve.ytm(date, du)
            if ytm is None or np.isnan(ytm):
                records.append((date, np.nan, np.nan, np.nan, np.nan))
            else:
                price     = price_ntnf(ytm, du, coupon, face, freq)
                _, mod, _ = bond_duration(ytm, du, coupon, face, freq)
                records.append((date, du, ytm, price, mod))
    df = pd.DataFrame(records, columns=['date', 'du', 'ytm', 'price', 'mod_dur'])
    return df.set_index('date')
