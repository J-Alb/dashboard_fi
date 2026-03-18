"""
br_bonds/_interpolation.py
──────────────────────────
Flat-forward and linear YTM interpolation (ANBIMA BUS/252 standard).

Both PrefixadoCurve and NTNBCurve share the same interpolation logic;
keeping it here avoids duplication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def flatfwd_df(
    du_verts: np.ndarray,
    df_verts: np.ndarray,
    du_t: float,
) -> float:
    """
    Flat-forward discount factor at ``du_t`` given known (du, df) vertices.

    Extrapolates flat beyond the ends — needed for stub-period coupons that
    fall before the first bootstrapped zero vertex.

    Parameters
    ----------
    du_verts : sorted array of business-day vertices (e.g. from zero_curve)
    df_verts : corresponding discount factors
    du_t     : target business days

    Returns
    -------
    float — discount factor at du_t
    """
    n = len(du_verts)
    if n == 0:
        return 1.0

    # Before first vertex: B(0)=1, flat forward from 0 to du_verts[0]
    if du_t <= du_verts[0]:
        return float(df_verts[0] ** (du_t / du_verts[0]))

    # Beyond last vertex: flat forward from last segment
    if du_t >= du_verts[-1]:
        if n == 1:
            return float(df_verts[0] ** (du_t / du_verts[0]))
        du1, du2 = du_verts[-2], du_verts[-1]
        df1, df2 = df_verts[-2], df_verts[-1]
        return float(df1 * (df2 / df1) ** ((du_t - du1) / (du2 - du1)))

    idx = min(int(np.searchsorted(du_verts, du_t, side='right')) - 1, n - 2)
    du1, du2 = du_verts[idx], du_verts[idx + 1]
    df1, df2 = df_verts[idx], df_verts[idx + 1]
    return float(df1 * (df2 / df1) ** ((du_t - du1) / (du2 - du1)))


def flatfwd_batch(
    du_verts: np.ndarray,
    df_verts: np.ndarray,
    du_grid: np.ndarray,
) -> np.ndarray:
    """
    Vectorised flat-forward discount factors for an array of du values.

    Values outside [du_verts[0], du_verts[-1]] are clipped to the nearest
    boundary segment (mirrors flatfwd_df boundary behaviour).

    Parameters
    ----------
    du_verts : sorted array of business-day vertices
    df_verts : discount factors at each vertex
    du_grid  : target business-day values (array)

    Returns
    -------
    np.ndarray — discount factors at each du in du_grid
    """
    idx  = np.searchsorted(du_verts, du_grid, side='right') - 1
    idx  = np.clip(idx, 0, len(du_verts) - 2)
    du1  = du_verts[idx];  du2 = du_verts[idx + 1]
    df1  = df_verts[idx];  df2 = df_verts[idx + 1]
    w    = (du_grid - du1) / (du2 - du1)
    return df1 * (df2 / df1) ** w


def _mat_adj(d, cal) -> pd.Timestamp:
    """
    Adjust a date to the next ANBIMA business day if it falls on a holiday/weekend.

    Parameters
    ----------
    d   : date-like (Timestamp, date, str)
    cal : bizdays.Calendar

    Returns
    -------
    pd.Timestamp — adjusted date
    """
    return pd.Timestamp(cal.adjust_next(d) if not cal.isbizday(d) else d)


def interp_yield(
    du: np.ndarray,
    ytm: np.ndarray,
    dfs: np.ndarray,
    du_t: float,
    method: str,
) -> float | None:
    """
    Flat-forward or linear YTM interpolation — hot path, no pandas.

    Returns None when ``du_t`` is outside the range [du[0], du[-1]].
    """
    if du_t < du[0] or du_t > du[-1]:
        return None
    if method == 'linear':
        return float(np.interp(du_t, du, ytm))
    idx = min(int(np.searchsorted(du, du_t, side='right')) - 1, len(du) - 2)
    du1, du2 = du[idx], du[idx + 1]
    df1, df2 = dfs[idx], dfs[idx + 1]
    w = (du_t - du1) / (du2 - du1)
    return float((df1 * (df2 / df1) ** w) ** (-252.0 / du_t) - 1.0)
