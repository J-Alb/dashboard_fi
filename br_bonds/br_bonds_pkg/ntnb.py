"""
br_bonds/ntnb.py
────────────────
Pricing and yield-curve tools for Brazilian NTN-B bonds (IPCA-linked real),
ANBIMA BUS/252 convention.

NTN-B key conventions
---------------------
- Price convention: *cotação* (% of VNA, face = 100)
  PU (R$) = cotação × VNA / 100
- VNA (Valor Nominal Atualizado): IPCA-accreted face value, published daily
  by ANBIMA. Cancels out inside yield/zero-rate calculations.
- Coupon schedule: same day-of-month as maturity, every 6 months
  (e.g. Aug-15 bond → coupons on Feb 15 / Aug 15).
- Fixed semi-annual coupon: C = face × [(1+coupon_real)^(1/freq) − 1]
  Same amount for all periods including stubs (do NOT pro-rate).
- Yields are *real* annual rates (IPCA + spread), BUS/252.

Public API
----------
price_ntnb(ytm_real, du, coupon_real, face, freq)
    Cotação (% of VNA) using uniform coupon spacing.

ntnb_tri(pus, vna, is_coupon, coupon_real, freq, face)
    Buy-and-hold Total Return Index for a single NTN-B bond.
    Coupon cash flows are added explicitly on payment dates.

NTNBCurve
    Panel-based real yield curve: flat-forward interpolation + zero-curve bootstrap.
    Supports optional VNA series for PU conversion.
"""

from __future__ import annotations

import warnings
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from bizdays import Calendar

from scipy.optimize import brentq

from ._interpolation import flatfwd_df, interp_yield
from ._schedules import price_from_schedule, ntnb_cashflow_schedule
from .prefixado import price_ltn


# ── standalone pricing ────────────────────────────────────────────────────────

def price_ntnb(
    ytm_real:    float,
    du:          int,
    coupon_real: float = 0.06,
    face:        float = 100.0,
    freq:        int   = 2,
) -> float:
    """
    Price an NTN-B expressed as *cotação* (% of VNA), BUS/252 convention.

    Uses uniform coupon spacing (252//freq bdays). For exact ANBIMA coupon
    schedules with real payment dates, use NTNBCurve which derives the
    schedule analytically from the bond maturity date.

    Parameters
    ----------
    ytm_real    : real annual yield (decimal, e.g. 0.06 for 6%)
    du          : business days to maturity
    coupon_real : real annual coupon rate (decimal); 0.0 for NTN-B Principal
    face        : face value in cotação space (default 100)
    freq        : payments per year (default 2 — semi-annual)

    Returns
    -------
    cotação (float) — multiply by VNA / 100 to obtain PU in R$.
    """
    if coupon_real == 0.0:
        return price_ltn(ytm_real, du, face)

    step = 252 // freq
    c    = face * ((1.0 + coupon_real) ** (1.0 / freq) - 1.0)

    cf_du = np.arange(step, du + step, step, dtype=np.float64)
    cf_du = cf_du[cf_du <= du + 0.5]
    if len(cf_du) == 0 or cf_du[-1] < du:
        cf_du = np.append(cf_du, float(du))

    cashflows      = np.full(len(cf_du), c)
    cashflows[-1] += face

    return float(np.sum(cashflows / (1.0 + ytm_real) ** (cf_du / 252.0)))


def ytm_ntnb(
    pu: float,
    vna: float,
    date: pd.Timestamp,
    mat: pd.Timestamp,
    cal: Calendar,
    coupon_real: float = 0.06,
    face: float = 100.0,
) -> float:
    """
    Invert NTN-B PU (R$) → real yield using the exact ANBIMA coupon schedule.

    Parameters
    ----------
    pu          : dirty price in R$ (= cotação × VNA / 100)
    vna         : VNA on settlement date (R$)
    date        : settlement date
    mat         : contractual maturity date
    cal         : ANBIMA calendar
    coupon_real : annual real coupon rate (default 0.06)
    face        : face value in cotação space (default 100)

    Returns
    -------
    float — real annual yield (decimal, BUS/252)
    """
    cot = pu * 100.0 / vna
    cf_du, cf = ntnb_cashflow_schedule(date, mat, cal, coupon_real, face)
    return brentq(
        lambda y: price_from_schedule(y, cf_du, cf) - cot,
        0.0001, 0.90, xtol=1e-8,
    )


# ── buy-and-hold total return index ──────────────────────────────────────────

def ntnb_tri(
    pus: np.ndarray,
    vna: np.ndarray,
    is_coupon: np.ndarray,
    coupon_real: float = 0.06,
    freq: int = 2,
    face: float = 100.0,
) -> np.ndarray:
    """
    Buy-and-hold Total Return Index for a single NTN-B bond, rebased to 100.

    Daily return:
        non-coupon :  PU_t / PU_{t-1}
        coupon date:  (PU_t + C_t) / PU_{t-1}

    where C_t = face × [(1 + coupon_real)^(1/freq) − 1] × VNA_t / 100

    The PU ratio already embeds both IPCA accrual (via VNA growth) and
    roll-down (the bond ages one business day). No additional accrual
    term is needed — this is the key difference from ret_index / cm_tri,
    which are designed for constant-maturity rolling strategies.

    NaN handling: if either PU_t or PU_{t-1} is NaN the return for that
    day is set to 1.0 (no change). The index carries forward at the last
    valid level until data resumes.

    Parameters
    ----------
    pus         : daily PU series in R$ (= cotação × VNA / 100)
    vna         : daily VNA series in R$ (used to compute coupon cash flows)
    is_coupon   : bool array aligned with pus/vna; True on payment dates
    coupon_real : annual real coupon rate (decimal, default 0.06)
    freq        : coupon payments per year (default 2 — semi-annual)
    face        : face value in cotação space (default 100)

    Returns
    -------
    np.ndarray — TRI rebased to 100 at the first valid observation.
                 NaN before the first valid PU.
    """
    pus       = np.asarray(pus,       dtype=float)
    vna       = np.asarray(vna,       dtype=float)
    is_coupon = np.asarray(is_coupon, dtype=bool)

    c_per_vna = face * ((1.0 + coupon_real) ** (1.0 / freq) - 1.0) / 100.0

    n     = len(pus)
    valid = ~np.isnan(pus)
    idx   = np.full(n, np.nan)

    fi = next((i for i in range(n) if valid[i]), None)
    if fi is None:
        return idx

    idx[fi] = 100.0
    prev    = 100.0

    for i in range(fi + 1, n):
        if not valid[i] or not valid[i - 1]:
            continue
        c      = c_per_vna * vna[i] if is_coupon[i] and not np.isnan(vna[i]) else 0.0
        idx[i] = prev * (pus[i] + c) / pus[i - 1]
        prev   = idx[i]

    return idx


# ── NTNBCurve ─────────────────────────────────────────────────────────────────

class NTNBCurve:
    """
    Real yield curve interpolator and zero-curve bootstrapper for NTN-B bonds
    (IPCA-linked), ANBIMA BUS/252 convention. Corresponds to ANBIMA ETTJ IPCA.

    Supports NTN-B (semi-annual real coupon) and NTN-B Principal (zero-coupon).
    All prices are expressed as *cotação* (% of VNA, face = 100). VNA is
    optional — supply it to enable PU conversion.

    Parameters
    ----------
    panel : pd.DataFrame
        Required columns: ['date', 'maturity', <yield_col>].
        Optional:         [<coupon_col>] — per-bond annual real coupon
                          (0.0 for NTN-B Principal, 0.06 for NTN-B).
        Yields must be real decimal (0.06 for 6%).
    cal : Calendar
        ANBIMA business-day calendar.
    method : {'flatfwd', 'linear'}
        Interpolation for ytm() / build_series(). 'flatfwd' is ANBIMA standard.
    yield_col : str
        Column with per-bond real yields (default 'yield_base').
    coupon_col : str or None
        Column with per-bond annual coupon rates as decimal (0.0 / 0.06).
        Required for accurate zero_curve().
    vna_series : pd.Series or None
        Daily VNA values indexed by pd.Timestamp.
        When provided, pu() and build_series() return PU in R$.

    Examples
    --------
    >>> from br_bonds import NTNBCurve, ntnb_coupon_dates
    >>> panel['coupon'] = panel['bond_type'].map({'NTN-B': 0.06, 'NTN-B-P': 0.0})
    >>> curve = NTNBCurve(panel, cal, coupon_col='coupon', vna_series=vna)
    >>> curve.zero_curve(date)           # real zero rates (ETTJ IPCA)
    >>> curve.build_series(dates, 1260)  # 5Y cotação + PU time series
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        cal: Calendar,
        method: Literal['flatfwd', 'linear'] = 'flatfwd',
        yield_col: str = 'yield_base',
        coupon_col: str | None = None,
        vna_series: pd.Series | None = None,
    ) -> None:
        if method not in ('flatfwd', 'linear'):
            raise ValueError("method must be 'flatfwd' or 'linear'")

        self.method     = method
        self.yield_col  = yield_col
        self.coupon_col = coupon_col
        self._cal       = cal
        self._vna       = vna_series
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

            if coupon_col:
                cpn = grp[coupon_col].to_numpy(dtype=np.float64)
                mat = grp['maturity'].to_numpy(dtype='datetime64[D]')
                self._index[date] = (du, ytm, dfs, cpn, mat)
            else:
                self._index[date] = (du, ytm, dfs)

    # ── public interface ───────────────────────────────────────────────────────

    def ytm(self, date: pd.Timestamp, du_target: int) -> float | None:
        """
        Flat-forward or linear interpolated real YTM at ``du_target`` bdays.
        Returns None if date not in index or du_target out of range.
        """
        entry = self._index.get(date)
        if entry is None:
            return None
        return interp_yield(entry[0], entry[1], entry[2], float(du_target), self.method)

    def cotacao(
        self,
        date: pd.Timestamp,
        du_target: int,
        coupon: float = 0.06,
        face: float = 100.0,
        freq: int = 2,
        mat: pd.Timestamp | None = None,
    ) -> float | None:
        """
        Cotação (% of VNA) at ``du_target`` bdays.

        Pass ``mat`` (bond maturity date) for the exact ANBIMA coupon schedule.
        When omitted, uses price_ntnb (uniform 126-bday spacing).
        """
        y = self.ytm(date, du_target)
        if y is None:
            return None
        if mat is not None:
            cf_du, cf = ntnb_cashflow_schedule(date, mat, self._cal, coupon, face, freq)
            return float(np.sum(cf / (1.0 + y) ** (cf_du / 252.0)))
        return price_ntnb(y, du_target, coupon, face, freq)

    def pu(
        self,
        date: pd.Timestamp,
        du_target: int,
        coupon: float = 0.06,
        face: float = 100.0,
        freq: int = 2,
        mat: pd.Timestamp | None = None,
    ) -> float | None:
        """
        Mark-to-market PU in R$ at ``du_target`` bdays.
        PU = cotação × VNA / 100.
        Requires vna_series at construction; returns None if VNA missing.
        """
        cot = self.cotacao(date, du_target, coupon, face, freq, mat)
        if cot is None:
            return None
        if self._vna is None:
            warnings.warn(
                "NTNBCurve.pu(): vna_series was not supplied. "
                "Pass vna_series= at construction to enable PU conversion.",
                UserWarning, stacklevel=2,
            )
            return None
        v = self._vna.get(date)
        if v is None:
            return None
        return cot * v / 100.0

    def build_series(
        self,
        dates: Sequence[pd.Timestamp],
        du_target: int,
        coupon: float = 0.06,
        face: float = 100.0,
        freq: int = 2,
    ) -> pd.DataFrame:
        """
        Compute (ytm, cotacao, vna, pu) for all ``dates`` at fixed ``du_target``.

        ``vna`` and ``pu`` are NaN when ``vna_series`` was not provided.

        Returns
        -------
        pd.DataFrame indexed by dates with columns ['ytm', 'cotacao', 'vna', 'pu'].
        """
        n       = len(dates)
        ytms    = np.empty(n, dtype=np.float64)
        cotacos = np.empty(n, dtype=np.float64)
        du_t    = float(du_target)

        for i, d in enumerate(dates):
            entry = self._index.get(d)
            if entry is None:
                ytms[i] = cotacos[i] = np.nan
                continue
            y = interp_yield(entry[0], entry[1], entry[2], du_t, self.method)
            if y is None:
                ytms[i] = cotacos[i] = np.nan
            else:
                ytms[i]    = y
                cotacos[i] = price_ntnb(y, du_target, coupon, face, freq)

        if self._vna is not None:
            vna_vals = np.array([self._vna.get(d, np.nan) for d in dates],
                                dtype=np.float64)
        else:
            vna_vals = np.full(n, np.nan)

        return pd.DataFrame({
            'ytm':     ytms,
            'cotacao': cotacos,
            'vna':     vna_vals,
            'pu':      cotacos * vna_vals / 100.0,
        }, index=pd.DatetimeIndex(dates))

    # ── zero-curve bootstrap ──────────────────────────────────────────────────

    def zero_curve(
        self,
        date: pd.Timestamp,
        coupon: float = 0.06,
        face: float = 100.0,
        freq: int = 2,
    ) -> pd.DataFrame | None:
        """
        Bootstrap the real zero (spot) rate curve on ``date``.

        Bond treatment
        --------------
        NTN-B-P (coupon = 0):     exact — B(T) = (1+ytm_real)^(−T/252).
        NTN-B, single cash flow:  exact — tagged 'NTN-B (zero)'.
        NTN-B, first coupon bond: seed  — flat-curve approx; tagged 'NTN-B (seed)'.
        NTN-B, multiple CFs:      bootstrapped via
                                  B(T) = [cotação − Σ Cᵢ · B(tᵢ)] / Cₙ

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

        for j in range(n_bonds):
            T   = bond_dus[j]
            ytm = bond_ytms[j]
            cpn = bond_cpns[j]

            if cpn == 0.0:
                B_T    = (1.0 + ytm) ** (-T / 252.0)
                b_type = 'NTN-B-P'

            else:
                mat_np_j = bond_mats[j] if bond_mats is not None else None
                cf_du, cashflows = self._ntnb_schedule(date, mat_np_j, cpn, face, freq)

                if len(cf_du) == 1:
                    B_T    = (1.0 + ytm) ** (-T / 252.0)
                    b_type = 'NTN-B (zero)'

                elif z_len == 0:
                    B_T    = (1.0 + ytm) ** (-T / 252.0)
                    b_type = 'NTN-B (seed)'

                else:
                    P      = price_from_schedule(ytm, cf_du, cashflows)
                    pv_int = sum(
                        cashflows[k] * flatfwd_df(z_du[:z_len], z_df[:z_len], cf_du[k])
                        for k in range(len(cf_du) - 1)
                    )
                    residual = P - pv_int
                    if residual <= 0.0:
                        warnings.warn(
                            f"NTNBCurve bootstrap: negative residual at "
                            f"du={T:.0f} on {date.date()} ({residual:.6f}). "
                            "Falling back to YTM.",
                            RuntimeWarning, stacklevel=2,
                        )
                        B_T = (1.0 + ytm) ** (-T / 252.0)
                    else:
                        B_T = residual / cashflows[-1]
                    b_type = 'NTN-B'

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

    # ── coupon schedule helper ─────────────────────────────────────────────────

    def _ntnb_schedule(
        self,
        date: pd.Timestamp,
        mat_np,
        coupon: float,
        face: float,
        freq: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Delegate to ntnb_cashflow_schedule (derived analytically from maturity)."""
        if mat_np is not None:
            mat_ts = pd.Timestamp(str(mat_np))
            return ntnb_cashflow_schedule(date, mat_ts, self._cal, coupon, face, freq)
        raise ValueError(
            "NTNBCurve._ntnb_schedule: mat_np is None. "
            "Ensure 'maturity' column is present in the panel."
        )

    # ── dunder helpers ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        dates = list(self._index)
        if not dates:
            return "NTNBCurve(empty)"
        has_cpn = f"coupon_col={'yes' if self.coupon_col else 'no'}"
        has_vna = f"vna={'yes' if self._vna is not None else 'no'}"
        return (
            f"NTNBCurve(method='{self.method}', "
            f"dates={len(dates)}, "
            f"range={min(dates).date()}…{max(dates).date()}, "
            f"{has_cpn}, {has_vna})"
        )
