"""
br_bonds/prefixado.py
─────────────────────
Pricing and yield-curve tools for Brazilian prefixado bonds (LTN + NTN-F),
ANBIMA BUS/252 convention.

Public API
----------
price_ltn(ytm, du, face)
    Zero-coupon LTN pricing.

price_ntnf(ytm, du, coupon, face, freq)
    Coupon NTN-F pricing (uniform 126-bday coupon spacing).

build_tri(prices, is_coupon, coupon_amount, base)
    Cumulative total-return index from a price series with coupon events.

PrefixadoCurve
    Panel-based yield curve: flat-forward interpolation + zero-curve bootstrap.
    Expensive pandas/bizdays work done once at construction; per-date queries
    are O(log n) pure-NumPy.

NTNFCurve
    Backward-compatible alias for PrefixadoCurve.
"""

from __future__ import annotations

import warnings
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from bizdays import Calendar

from scipy.optimize import brentq

from ._interpolation import flatfwd_df, interp_yield
from ._schedules import price_from_schedule, bond_cashflow_schedule


# ── standalone pricing ────────────────────────────────────────────────────────

def price_ltn(ytm: float, du: int, face: float = 1000.0) -> float:
    """
    Price a zero-coupon LTN (or any zero-coupon prefixado bond).

    PU = face / (1 + ytm)^(du/252)

    Parameters
    ----------
    ytm  : annual yield (decimal, ANBIMA BUS/252)
    du   : business days to maturity (use adjusted maturity date)
    face : face value in R$ (default 1000)
    """
    return face / (1.0 + ytm) ** (du / 252.0)


def price_ntnf(
    ytm: float,
    du: int,
    coupon: float = 0.10,
    face: float = 1000.0,
    freq: int = 2,
) -> float:
    """
    Price a coupon NTN-F (or any coupon prefixado bond).

    Uses uniform 252//freq bday coupon spacing. For exact ANBIMA schedules
    with real coupon dates, use PrefixadoCurve which accepts coupon_dates.

    Parameters
    ----------
    ytm    : annual yield (decimal, ANBIMA BUS/252)
    du     : business days to maturity
    coupon : annual coupon rate (decimal); default 0.10 (10% p.a.)
    face   : face value (default 1000)
    freq   : coupon payments per year (default 2 — semi-annual)
    """
    if coupon == 0.0:
        return price_ltn(ytm, du, face)

    step = 252 // freq
    c    = face * ((1.0 + coupon) ** (1.0 / freq) - 1.0)

    cf_du = np.arange(step, du + step, step, dtype=np.float64)
    cf_du = cf_du[cf_du <= du + 0.5]
    if len(cf_du) == 0 or cf_du[-1] < du:
        cf_du = np.append(cf_du, float(du))

    cashflows      = np.full(len(cf_du), c)
    cashflows[-1] += face

    return float(np.sum(cashflows / (1.0 + ytm) ** (cf_du / 252.0)))


def ytm_ltn(pu: float, du: float, face: float = 1000.0) -> float:
    """
    Invert LTN price → yield (analytical).

    ytm = (face / pu)^(252 / du) − 1

    Parameters
    ----------
    pu   : dirty price in R$
    du   : business days to maturity (adjusted)
    face : face value (default 1000)
    """
    return (face / pu) ** (252.0 / du) - 1.0


def ytm_ntnf(
    pu: float,
    du: float,
    date: pd.Timestamp,
    mat: pd.Timestamp,
    cal: Calendar,
    coupon_dates: np.ndarray,
    coupon: float = 0.10,
    face: float = 1000.0,
) -> float:
    """
    Invert NTN-F price → yield using the exact ANBIMA coupon schedule.

    Parameters
    ----------
    pu           : dirty price in R$
    du           : business days to maturity (adjusted)
    date         : settlement date
    mat          : contractual maturity (unadjusted — 5-day buffer handles adj)
    cal          : ANBIMA calendar
    coupon_dates : ANBIMA-adjusted Jan 1 / Jul 1 coupon dates (datetime64[D])
    coupon       : annual coupon rate (default 0.10)
    face         : face value (default 1000)

    Returns
    -------
    float — annual yield (decimal, BUS/252)
    """
    cf_du, cf = bond_cashflow_schedule(
        float(du), coupon, face, 2,
        date_np=np.datetime64(date.date(), 'D'),
        mat_np=np.datetime64(mat.date(), 'D'),
        cal=cal, coupon_dates=coupon_dates,
    )
    return brentq(
        lambda y: price_from_schedule(y, cf_du, cf) - pu,
        0.0001, 0.90, xtol=1e-8,
    )


# ── total return index ─────────────────────────────────────────────────────────

def build_tri(
    prices: np.ndarray,
    is_coupon: np.ndarray,
    coupon_amount: float,
    base: float | None = None,
) -> np.ndarray:
    """
    Cumulative Total Return Index from a daily price series.

    Return on day t:
        non-coupon :  prices[t] / prices[t-1]
        coupon date:  (prices[t] + coupon_amount) / prices[t-1]

    NaN prices → return = 1.0 (no change) for that step.

    Parameters
    ----------
    prices        : 1-D float array (may contain NaN)
    is_coupon     : bool array aligned with prices
    coupon_amount : cash coupon per period (e.g. 48.809 for NTN-F 10%)
    base          : rebase value; defaults to first non-NaN price

    Returns
    -------
    np.ndarray aligned with prices
    """
    n     = len(prices)
    ret   = np.ones(n, dtype=np.float64)
    valid = ~np.isnan(prices)

    for i in range(1, n):
        if not valid[i] or not valid[i - 1]:
            continue
        c      = coupon_amount if is_coupon[i] else 0.0
        ret[i] = (prices[i] + c) / prices[i - 1]

    tri   = np.cumprod(ret)
    first = prices[valid][0] if base is None else base
    return tri / tri[0] * first


# ── PrefixadoCurve ─────────────────────────────────────────────────────────────

class PrefixadoCurve:
    """
    Yield curve interpolator and zero-curve bootstrapper for Brazilian
    prefixado bonds (LTN + NTN-F), ANBIMA BUS/252 convention.

    Expensive work (bizdays loop, groupby, DF pre-computation) is done ONCE
    at construction. Per-date queries are O(log n) pure-NumPy calls.

    Parameters
    ----------
    panel : pd.DataFrame
        Required columns: ['date', 'maturity', <yield_col>].
        Optional:         [<coupon_col>] — per-bond annual coupon (decimal).
        Yields must be decimal (0.12 for 12%).
    cal : Calendar
        ANBIMA business-day calendar.
    method : {'flatfwd', 'linear'}
        Interpolation method. 'flatfwd' is the ANBIMA standard.
    yield_col : str
        Column with per-bond yields (default 'yield_base').
    coupon_col : str or None
        Column with per-bond annual coupon rates as decimals
        (0.0 for LTN, 0.10 for NTN-F). Required for accurate zero_curve().
    coupon_dates : array-like of dates or None
        ANBIMA-adjusted semiannual coupon payment dates (Jan 1 / Jul 1,
        shifted to next business day). When supplied, zero_curve() uses
        the real payment schedule for each NTN-F, correctly handling stub
        periods. If None, falls back to uniform 126-bday spacing.

    Examples
    --------
    >>> from bizdays import Calendar
    >>> cal = Calendar.load('ANBIMA')
    >>> cdates = pd.date_range('2000-01-01', '2040-01-01', freq='6MS')
    >>> cdates_adj = cdates.map(lambda d: cal.adjust_next(d) if not cal.isbizday(d) else d)
    >>> panel['coupon'] = panel['bond_type'].map({'LTN': 0.0, 'NTN-F': 0.10})
    >>> curve = PrefixadoCurve(panel, cal, coupon_col='coupon',
    ...                        coupon_dates=cdates_adj)
    >>> curve.zero_curve(date)
    >>> curve.build_series(dates, du_target=1260)
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        cal: Calendar,
        method: Literal['flatfwd', 'linear'] = 'flatfwd',
        yield_col: str = 'yield_base',
        coupon_col: str | None = None,
        coupon_dates=None,
    ) -> None:
        if method not in ('flatfwd', 'linear'):
            raise ValueError("method must be 'flatfwd' or 'linear'")

        self.method     = method
        self.yield_col  = yield_col
        self.coupon_col = coupon_col
        self._cal       = cal

        if coupon_dates is not None:
            self._cpn_dates = np.sort(
                pd.to_datetime(coupon_dates).values.astype('datetime64[D]')
            )
        else:
            self._cpn_dates = None

        # Main index: date -> (du, ytm, dfs [, cpn, mat])
        self._index: dict = {}
        self._build(panel, cal, yield_col, coupon_col)

    # ── construction ──────────────────────────────────────────────────────────

    def _build(self, panel: pd.DataFrame, cal: Calendar,
               yield_col: str, coupon_col: str | None) -> None:
        needed = ['date', 'maturity', yield_col]
        if coupon_col:
            needed.append(coupon_col)
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
            sort_cols = ['du', coupon_col] if coupon_col else ['du']
            grp = grp.sort_values(sort_cols)
            du  = grp['du'].to_numpy(dtype=np.float64)
            ytm = grp[yield_col].to_numpy(dtype=np.float64)
            dfs = (1.0 + ytm) ** (-du / 252.0)

            if coupon_col and self._cpn_dates is not None:
                cpn = grp[coupon_col].to_numpy(dtype=np.float64)
                mat = grp['maturity'].to_numpy(dtype='datetime64[D]')
                self._index[date] = (du, ytm, dfs, cpn, mat)
            elif coupon_col:
                cpn = grp[coupon_col].to_numpy(dtype=np.float64)
                self._index[date] = (du, ytm, dfs, cpn)
            else:
                self._index[date] = (du, ytm, dfs)

    # ── public interface ──────────────────────────────────────────────────────

    def ytm(self, date: pd.Timestamp, du_target: int) -> float | None:
        """
        Flat-forward or linear interpolated YTM at ``du_target`` bdays.
        Returns None if date not in index or du_target out of range.
        """
        entry = self._index.get(date)
        if entry is None:
            return None
        return interp_yield(entry[0], entry[1], entry[2], float(du_target), self.method)

    def price(
        self,
        date: pd.Timestamp,
        du_target: int,
        coupon: float = 0.10,
        face: float = 1000.0,
        freq: int = 2,
    ) -> float | None:
        """Price a synthetic coupon bond at ``du_target`` bdays."""
        y = self.ytm(date, du_target)
        return None if y is None else price_ntnf(y, du_target, coupon, face, freq)

    def build_series(
        self,
        dates: Sequence[pd.Timestamp],
        du_target: int,
        coupon: float = 0.10,
        face: float = 1000.0,
        freq: int = 2,
    ) -> pd.DataFrame:
        """
        Compute (ytm, price) for all ``dates`` at fixed ``du_target``.
        Returns DataFrame(ytm, price).
        """
        du_t   = float(du_target)
        n      = len(dates)
        ytms   = np.empty(n, dtype=np.float64)
        prices = np.empty(n, dtype=np.float64)

        for i, d in enumerate(dates):
            entry = self._index.get(d)
            if entry is None:
                ytms[i] = prices[i] = np.nan
                continue
            y = interp_yield(entry[0], entry[1], entry[2], du_t, self.method)
            if y is None:
                ytms[i] = prices[i] = np.nan
            else:
                ytms[i]   = y
                prices[i] = price_ntnf(y, du_target, coupon, face, freq)

        return pd.DataFrame({'ytm': ytms, 'price': prices},
                            index=pd.DatetimeIndex(dates))

    # ── zero-curve bootstrap ──────────────────────────────────────────────────

    def zero_curve(
        self,
        date: pd.Timestamp,
        coupon: float = 0.10,
        face: float = 1000.0,
        freq: int = 2,
    ) -> pd.DataFrame | None:
        """
        Bootstrap the zero (spot) rate curve on ``date``.

        Bond treatment
        --------------
        LTN  (coupon = 0):        exact — B(T) = (1+ytm)^(−T/252).
        NTN-F, single cash flow:  exact — tagged 'NTN-F (zero)'.
        NTN-F, first coupon bond: seed  — flat-curve approx, tagged 'NTN-F (seed)'.
        NTN-F, multiple CFs:      bootstrapped via
                                  B(T) = [P − Σ Cᵢ · B(tᵢ)] / Cₙ

        Returns
        -------
        pd.DataFrame with columns ['du', 'bond_type', 'zero_rate',
                                    'discount_factor'].
        None if date is not in the index.
        """
        entry = self._index.get(date)
        if entry is None:
            return None

        bond_dus  = entry[0]
        bond_ytms = entry[1]
        n_bonds   = len(bond_dus)

        bond_cpns = entry[3] if len(entry) >= 4 else np.full(n_bonds, coupon)
        bond_mats = entry[4] if len(entry) == 5 else None

        z_du    = np.empty(n_bonds, dtype=np.float64)
        z_df    = np.empty(n_bonds, dtype=np.float64)
        z_types = []
        z_len   = 0

        date_np = np.datetime64(date.date(), 'D')

        for j in range(n_bonds):
            T    = bond_dus[j]
            ytm  = bond_ytms[j]
            cpn  = bond_cpns[j]

            if cpn == 0.0:
                B_T    = (1.0 + ytm) ** (-T / 252.0)
                b_type = 'LTN'

            else:
                cf_du, cashflows = self._ntnf_schedule(
                    date, date_np, T,
                    bond_mats[j] if bond_mats is not None else None,
                    cpn, face, freq,
                )

                if len(cf_du) == 1:
                    B_T    = (1.0 + ytm) ** (-T / 252.0)
                    b_type = 'NTN-F (zero)'

                elif z_len == 0:
                    B_T    = (1.0 + ytm) ** (-T / 252.0)
                    b_type = 'NTN-F (seed)'

                else:
                    P      = price_from_schedule(ytm, cf_du, cashflows)
                    pv_int = sum(
                        cashflows[k] * flatfwd_df(z_du[:z_len], z_df[:z_len], cf_du[k])
                        for k in range(len(cf_du) - 1)
                    )
                    residual = P - pv_int
                    if residual <= 0.0:
                        warnings.warn(
                            f"Bootstrap: negative residual at du={T:.0f} on "
                            f"{date.date()} ({residual:.6f}). Falling back to YTM.",
                            RuntimeWarning, stacklevel=2,
                        )
                        B_T = (1.0 + ytm) ** (-T / 252.0)
                    else:
                        B_T = residual / cashflows[-1]
                    b_type = 'NTN-F'

            if z_len > 0 and T in z_du[:z_len]:
                continue

            z_du[z_len]  = T
            z_df[z_len]  = B_T
            z_types.append(b_type)
            z_len       += 1

        zero_rates = z_df[:z_len] ** (-252.0 / z_du[:z_len]) - 1.0

        return pd.DataFrame({
            'du':              z_du[:z_len],
            'bond_type':       z_types,
            'zero_rate':       zero_rates,
            'discount_factor': z_df[:z_len],
        })

    # ── coupon schedule helper ────────────────────────────────────────────────

    def _ntnf_schedule(
        self,
        date: pd.Timestamp,
        date_np: np.datetime64,
        du_mat: float,
        mat_np,
        coupon: float,
        face: float,
        freq: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (cf_du, cashflows) for an NTN-F bond.

        Uses actual ANBIMA coupon dates when coupon_dates was supplied,
        otherwise falls back to uniform 252//freq bday spacing.
        """
        c = face * ((1.0 + coupon) ** (1.0 / freq) - 1.0)

        if self._cpn_dates is not None and mat_np is not None:
            mat_d64 = (mat_np.astype('datetime64[D]')
                       if mat_np.dtype != np.dtype('datetime64[D]') else mat_np)
            mat_buf = mat_d64 + np.timedelta64(5, 'D')
            mask    = (self._cpn_dates > date_np) & (self._cpn_dates <= mat_buf)
            cpn_dt  = self._cpn_dates[mask]

            if len(cpn_dt) == 0:
                return np.array([du_mat]), np.array([c + face])

            cf_du = np.array(
                [self._cal.bizdays(date, pd.Timestamp(str(d))) for d in cpn_dt],
                dtype=np.float64,
            )
            cf_du[-1] = du_mat
        else:
            step  = 252 // freq
            cf_du = np.arange(step, du_mat + step, step, dtype=np.float64)
            cf_du = cf_du[cf_du <= du_mat + 0.5]
            if len(cf_du) == 0 or cf_du[-1] < du_mat:
                cf_du = np.append(cf_du, du_mat)

        cashflows      = np.full(len(cf_du), c)
        cashflows[-1] += face
        return cf_du, cashflows

    # ── dunder helpers ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        dates = list(self._index)
        if not dates:
            return f"{type(self).__name__}(empty)"
        has_cpn = f"coupon_col={'yes' if self.coupon_col else 'no'}"
        has_cpd = f"coupon_dates={'yes' if self._cpn_dates is not None else 'no'}"
        return (
            f"{type(self).__name__}(method='{self.method}', "
            f"dates={len(dates)}, "
            f"range={min(dates).date()}…{max(dates).date()}, "
            f"{has_cpn}, {has_cpd})"
        )


# backward-compatible alias
NTNFCurve = PrefixadoCurve
