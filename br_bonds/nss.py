"""
br_bonds/nss.py
───────────────
Nelson-Siegel-Svensson yield curve (ANBIMA Diebold-Li convention).

Public API
----------
NSSResult            — fitted parameters + convenience methods (ytm, df, curve)
nss_ytm(du, beta, lam)  — evaluate NSS zero rate
fit_nss(du, zero_rates) — fit NSS to bootstrapped zero rates (WLS + Nelder-Mead)
fit_nss_bonds(...)      — fit NSS to bond prices (1/Duration-weighted)
fit_nss_prefixado(...)  — ANBIMA ETTJ PREF: bootstrap zeros → fit_nss
fit_nss_ntnb(...)       — ANBIMA ETTJ IPCA: fit_nss_bonds on NTN-B prices
fit_nss_anbima(...)     — backward-compatible alias for fit_nss_prefixado

ANBIMA convention
-----------------
  r(t) = β₁ + β₂·φ(λ₁t) + β₃·ψ(λ₁t) + β₄·ψ(λ₂t)
  t    = du / 252   (business years)
  λ₁ > λ₂           (identification: β₃ hump at shorter maturity)

λ is a decay *rate* (Diebold-Li), NOT a time constant (Svensson).
Using x = t/λ instead of x = λt causes ~177 bp errors at short end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from bizdays import Calendar

from ._schedules import bond_cashflow_schedule, ntnb_cashflow_schedule


# ── NSS result ────────────────────────────────────────────────────────────────

@dataclass
class NSSResult:
    """
    Fitted NSS parameters and diagnostics.

    Attributes
    ----------
    beta : ndarray, shape (4,)  — [β₁, β₂, β₃, β₄]
    lam  : ndarray, shape (2,)  — [λ₁, λ₂] in business years⁻¹ (decay rates)
    rmse : float — RMSE (decimal; on zero rates when fit via fit_nss,
                          on prices when fit via fit_nss_bonds)
    r2   : float — R² on the fitted quantity
    n_obs: int   — number of observations used in fit
    """
    beta : np.ndarray
    lam  : np.ndarray
    rmse : float
    r2   : float
    n_obs: int

    def ytm(self, du: float | np.ndarray) -> float | np.ndarray:
        """Zero rate(s) at ``du`` business days (decimal annual)."""
        return nss_ytm(du, self.beta, self.lam)

    def discount_factor(self, du: float | np.ndarray) -> float | np.ndarray:
        """ANBIMA BUS/252 discount factor at ``du`` business days."""
        du_arr = np.asarray(du, dtype=np.float64)
        r = nss_ytm(du_arr, self.beta, self.lam)
        return (1.0 + r) ** (-du_arr / 252.0)

    def curve(self, du_max: int = 3780, step: int = 21) -> pd.DataFrame:
        """
        Smooth NSS curve from 1 bday to ``du_max`` bdays.
        Returns DataFrame with columns ['du', 'ytm', 'discount_factor'].
        """
        dus = np.arange(1, du_max + 1, step, dtype=np.float64)
        return pd.DataFrame({
            'du':              dus,
            'ytm':             self.ytm(dus),
            'discount_factor': self.discount_factor(dus),
        })

    def __repr__(self) -> str:
        b, l = self.beta, self.lam
        return (
            f"NSSResult("
            f"β=[{b[0]:.6f}, {b[1]:.6f}, {b[2]:.6f}, {b[3]:.6f}], "
            f"λ=[{l[0]:.4f}, {l[1]:.4f}], "
            f"RMSE={self.rmse * 100:.4f}bps×100, "
            f"R²={self.r2:.6f}, n={self.n_obs})"
        )


# ── NSS basis functions ───────────────────────────────────────────────────────

def _nss_phi(t: np.ndarray, lam: float) -> np.ndarray:
    """φ(t, λ) = (1 − exp(−λt)) / (λt).  Safe at t → 0.

    λ is the Diebold-Li decay *rate* (per business-year): x = λ·t.
    """
    x = lam * t
    return np.where(x < 1e-12, 1.0, (1.0 - np.exp(-x)) / x)


def _nss_psi(t: np.ndarray, lam: float) -> np.ndarray:
    """ψ(t, λ) = φ(t, λ) − exp(−λt).  Safe at t → 0."""
    x = lam * t
    return np.where(x < 1e-12, 0.0, (1.0 - np.exp(-x)) / x - np.exp(-x))


def _nss_design(t: np.ndarray, lam1: float, lam2: float) -> np.ndarray:
    """Design matrix X (n × 4) for fixed λ₁, λ₂."""
    X = np.empty((len(t), 4), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1] = _nss_phi(t, lam1)
    X[:, 2] = _nss_psi(t, lam1)
    X[:, 3] = _nss_psi(t, lam2)
    return X


# ── public NSS evaluation ─────────────────────────────────────────────────────

def nss_ytm(
    du:   float | np.ndarray,
    beta: Sequence[float],
    lam:  Sequence[float],
) -> float | np.ndarray:
    """
    Nelson-Siegel-Svensson zero rate at ``du`` business days.

    Exact replication of ANBIMA's published curve when ANBIMA's β₁–β₄ and
    λ₁–λ₂ are passed directly (Diebold-Li convention, x = λ·t).

    Parameters
    ----------
    du   : business days to maturity (scalar or array)
    beta : [β₁, β₂, β₃, β₄] — as published by ANBIMA (decimal)
    lam  : [λ₁, λ₂]          — as published by ANBIMA (business years⁻¹)

    Returns
    -------
    Annual zero rate(s) as decimal (e.g. 0.1366 for 13.66%).
    Scalar in → scalar out; array in → array out.
    """
    scalar = np.ndim(du) == 0
    du_arr = np.atleast_1d(np.asarray(du, dtype=np.float64))
    b      = np.asarray(beta, dtype=np.float64)
    l      = np.asarray(lam,  dtype=np.float64)

    if len(b) != 4:
        raise ValueError("beta must have 4 elements [β₁, β₂, β₃, β₄]")
    if len(l) != 2:
        raise ValueError("lam must have 2 elements [λ₁, λ₂]")
    if np.any(l <= 0):
        raise ValueError("λ values must be strictly positive")

    t = du_arr / 252.0
    r = b[0] + b[1] * _nss_phi(t, l[0]) + b[2] * _nss_psi(t, l[0]) + b[3] * _nss_psi(t, l[1])
    return float(r[0]) if scalar else r


# ── fit NSS to zero rates ─────────────────────────────────────────────────────

def fit_nss(
    du:           np.ndarray,
    zero_rates:   np.ndarray,
    weights:      np.ndarray | None = None,
    lam1:         float | None = None,
    lam2:         float | None = None,
    lam1_grid:    np.ndarray | None = None,
    lam2_grid:    np.ndarray | None = None,
    n_starts:     int = 3,
    global_search: bool = False,
    min_lam_sep:  float = 0.10,
) -> NSSResult:
    """
    Fit Nelson-Siegel-Svensson to bootstrapped zero rates.

    Designed to receive the output of ``PrefixadoCurve.zero_curve()``:

        zc  = curve.zero_curve(date)
        res = fit_nss(zc['du'].values, zc['zero_rate'].values)
        res.ytm(1260)   # 5Y zero rate from fitted curve

    Estimation strategy
    -------------------
    Both λ₁ and λ₂ provided (fixed):
        Weighted least squares — closed-form, instant.
    One or both λ are None (free):
        Grid search over all (λ₁, λ₂) pairs, solving WLS for β at
        each node. Best n_starts grid points each refined with Nelder-Mead;
        global best returned.

        If global_search=True, skips grid+NM and uses differential_evolution
        for a true global search over (λ₁, λ₂) with β solved analytically.

    Parameters
    ----------
    du            : business days array
    zero_rates    : annual zero rates, decimal
    weights       : per-vertex weights; default = 1/du (short end weighted more)
    lam1          : fixed λ₁ in business years⁻¹; None to estimate
    lam2          : fixed λ₂ in business years⁻¹; None to estimate
    lam1_grid     : candidate λ₁ values; default arange(0.1, 8.1, 0.1)
    lam2_grid     : candidate λ₂ values; default arange(0.05, 3.05, 0.05)
    n_starts      : number of best grid candidates to refine with Nelder-Mead
                    (default 3); higher values reduce local-minimum trapping.
    global_search : if True, use differential_evolution (slow but globally
                    robust). Overrides grid+NM when both λ are free.
    """
    du_arr = np.asarray(du, dtype=np.float64)
    r_arr  = np.asarray(zero_rates, dtype=np.float64)

    mask   = np.isfinite(du_arr) & np.isfinite(r_arr) & (du_arr > 0)
    du_arr, r_arr = du_arr[mask], r_arr[mask]

    n = len(du_arr)
    if n < 4:
        raise ValueError(f"Need at least 4 valid observations to fit NSS (got {n}).")

    t = du_arr / 252.0

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)[mask]
    else:
        w = 1.0 / du_arr
    w = w / w.sum()

    def _wls(l1: float, l2: float):
        if l1 <= 0.0 or l2 <= 0.0:
            return None, np.inf
        X   = _nss_design(t, l1, l2)
        Xw  = X * w[:, None]
        XtX = Xw.T @ X
        Xtr = Xw.T @ r_arr
        try:
            beta = np.linalg.solve(XtX, Xtr)
        except np.linalg.LinAlgError:
            return None, np.inf
        resid = r_arr - X @ beta
        ssr   = float(resid @ (w * resid))
        return beta, ssr

    if lam1 is not None and lam2 is not None:
        best_beta, _ = _wls(lam1, lam2)
        best_lam     = np.array([lam1, lam2])
        if best_beta is None:
            raise RuntimeError("WLS failed with provided λ values.")

    else:
        if lam1_grid is None:
            lam1_grid = np.arange(0.1, 8.1, 0.1)   # wider: covers PREF λ₁~3.26
        if lam2_grid is None:
            lam2_grid = np.arange(0.05, 3.05, 0.05) # finer: covers PREF λ₂~0.18

        free_idx  = [i for i, fixed in enumerate([lam1, lam2]) if fixed is None]
        fixed_val = [lam1, lam2]

        def _obj(x_free):
            lam_trial = list(fixed_val)
            for fi, xi in zip(free_idx, x_free):
                lam_trial[fi] = float(xi)
            if lam_trial[0] - lam_trial[1] < min_lam_sep:
                return 1e12
            _, ssr = _wls(lam_trial[0], lam_trial[1])
            return ssr

        best_ssr  = np.inf
        best_beta = None
        best_lam  = None

        if global_search and lam1 is None and lam2 is None:
            # ── differential evolution: true global search over (λ₁, λ₂) ──────
            # Parametrize: x[0] = λ₁_base > 0, x[1] = δ > 0, λ₂ = λ₁_base - δ
            # (ensures λ₁ > λ₂ > 0 without constraints)
            def _obj_de(x):
                l1 = float(x[0])
                l2 = float(x[1])
                if l1 - l2 < min_lam_sep or l2 <= 0:
                    return 1e12
                _, ssr = _wls(l1, l2)
                return ssr

            de_res = differential_evolution(
                _obj_de,
                bounds=[(0.05, 10.0), (0.02, 6.0)],
                seed=42, maxiter=500, popsize=20, tol=1e-10,
                mutation=(0.5, 1.5), recombination=0.9,
            )
            if de_res.success or de_res.fun < 1e11:
                best_lam = np.array([float(de_res.x[0]), float(de_res.x[1])])
                best_beta, best_ssr = _wls(best_lam[0], best_lam[1])
                if best_beta is None:
                    best_ssr = np.inf

        if not global_search or best_lam is None:
            # ── grid search: collect all valid (λ₁, λ₂) results ─────────────
            l1_candidates = np.array([lam1]) if lam1 is not None else lam1_grid
            l2_candidates = np.array([lam2]) if lam2 is not None else lam2_grid

            grid_results = []   # list of (lam_array, ssr)
            for l1 in l1_candidates:
                for l2 in l2_candidates:
                    if lam1 is None and lam2 is None and l1 - l2 < min_lam_sep:
                        continue   # DL convention: λ₁ > λ₂ with minimum separation
                    beta, ssr = _wls(l1, l2)
                    if beta is not None:
                        grid_results.append((np.array([l1, l2]), ssr))

            if not grid_results:
                raise RuntimeError("Grid search failed: no valid (λ₁, λ₂) pair found.")

            # sort by SSR, take top n_starts
            grid_results.sort(key=lambda x: x[1])
            top_starts = grid_results[:max(1, n_starts)]

            # ── multi-start Nelder-Mead refinement ───────────────────────────
            for lam_init, ssr_init in top_starts:
                x0  = lam_init[free_idx]
                res = minimize(_obj, x0, method='Nelder-Mead',
                               options={'xatol': 1e-6, 'fatol': 1e-12, 'maxiter': 3000})
                if res.fun < best_ssr:
                    best_ssr = res.fun
                    refined = list(fixed_val)
                    for fi, xi in zip(free_idx, res.x):
                        refined[fi] = float(xi)
                    best_lam  = np.array(refined)
                    best_beta, _ = _wls(best_lam[0], best_lam[1])

            # keep best raw grid result if NM never improved it
            if best_lam is None:
                best_lam  = grid_results[0][0]
                best_beta, best_ssr = _wls(best_lam[0], best_lam[1])

    X      = _nss_design(t, best_lam[0], best_lam[1])
    fitted = X @ best_beta
    resid  = r_arr - fitted
    rmse   = float(np.sqrt(np.mean(resid ** 2)))
    ss_tot = float(np.sum((r_arr - r_arr.mean()) ** 2))
    r2     = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 1.0

    return NSSResult(beta=best_beta, lam=best_lam, rmse=rmse, r2=r2, n_obs=n)


# ── internals for price-based fitting ─────────────────────────────────────────

def _nss_bond_price(
    cf_du: np.ndarray,
    cashflows: np.ndarray,
    beta: np.ndarray,
    lam: np.ndarray,
) -> float:
    """Model price of a bond from NSS zero curve."""
    r  = nss_ytm(cf_du, beta, lam)
    df = (1.0 + r) ** (-cf_du / 252.0)
    return float(np.sum(cashflows * df))


def _modified_duration(
    ytm: float,
    cf_du: np.ndarray,
    cashflows: np.ndarray,
    price: float,
) -> float:
    """
    Modified duration in ANBIMA BUS/252 convention.

    MacaulayDur = Σ (du_j/252) × PV(CF_j) / P
    ModDur      = MacaulayDur / (1 + ytm)
    """
    t   = cf_du / 252.0
    pv  = cashflows / (1.0 + ytm) ** t
    mac = float(np.sum(t * pv) / price)
    return mac / (1.0 + ytm)


# ── fit NSS to bond prices ────────────────────────────────────────────────────

def fit_nss_bonds(
    bonds_df: pd.DataFrame,
    cal: Calendar,
    yield_col:    str = 'yield_base',
    coupon_col:   str = 'coupon',
    du_col:       str = 'du',
    maturity_col: str = 'maturity',
    date_col:     str = 'date',
    face:         float = 1000.0,
    freq:         int   = 2,
    coupon_dates=None,
    lam1_grid: np.ndarray | None = None,
    lam2_grid: np.ndarray | None = None,
    n_starts:  int = 3,
) -> NSSResult:
    """
    Fit NSS to Brazilian bonds following ANBIMA price-error methodology.

    Objective
    ---------
    Min  Σᵢ (1/Dᵢ) × (P_actual_i − P_model_i)²

    where:
      P_actual_i  = price from indicative yield (ANBIMA BUS/252)
      P_model_i   = Σⱼ CF_{i,j} × (1 + r_NSS(du_{i,j}))^(−du_{i,j}/252)
      Dᵢ          = modified duration at indicative yield (BUS/252)
      1/Dᵢ        = weight — penalises short-end errors more

    Parameters
    ----------
    bonds_df      : DataFrame for a **single date**, with columns
                    [du_col, yield_col, coupon_col, maturity_col, date_col].
                    Yields must be decimal. LTN/NTN-B-P: coupon=0.
    cal           : ANBIMA business-day calendar.
    coupon_dates  : ANBIMA-adjusted coupon dates (sorted datetime-like).
                    Not None → prefixado path (LTN/NTN-F, Jan 1/Jul 1).
                    None     → NTN-B path (Jan 15/Jul 15 from maturity).
    face          : face value (1000 for LTN/NTN-F, 100 for NTN-B cotação).
    lam1_grid     : search grid for λ₁; default arange(0.1, 3.1, 0.2)
    lam2_grid     : search grid for λ₂; default arange(0.1, 3.1, 0.2)
    """
    if lam1_grid is None:
        lam1_grid = np.arange(0.1, 3.1, 0.2)
    if lam2_grid is None:
        lam2_grid = np.arange(0.1, 3.1, 0.2)

    cpn_dates_np = (
        np.sort(pd.to_datetime(coupon_dates).values.astype('datetime64[D]'))
        if coupon_dates is not None else None
    )

    needed = [yield_col, coupon_col, maturity_col, date_col]
    df = bonds_df[needed].dropna().copy()

    if du_col in bonds_df.columns:
        df[du_col] = bonds_df.loc[df.index, du_col]
    else:
        def _mat_adj(mat):
            return cal.adjust_next(mat) if not cal.isbizday(mat) else mat
        df[du_col] = df.apply(
            lambda r: cal.bizdays(r[date_col], _mat_adj(r[maturity_col]))
            if r[maturity_col] > r[date_col] else 0,
            axis=1,
        )

    df = df[df[du_col] > 0]

    if len(df) < 4:
        raise ValueError(f"Need at least 4 bonds to fit NSS (got {len(df)}).")

    pricing_date = df[date_col].iloc[0]
    date_np = (np.datetime64(pricing_date.date(), 'D')
               if hasattr(pricing_date, 'date') else np.datetime64(pricing_date, 'D'))

    schedules  = []
    prices_act = []
    durations  = []

    for row in df.itertuples(index=False):
        ytm = float(getattr(row, yield_col))
        cpn = float(getattr(row, coupon_col))
        mat = getattr(row, maturity_col)

        if coupon_dates is not None:
            du_mat = float(getattr(row, du_col))
            mat_np = np.datetime64(mat, 'D') if mat is not None else None
            cf_du, cashflows = bond_cashflow_schedule(
                du_mat, cpn, face, freq,
                date_np=date_np, mat_np=mat_np,
                cal=cal, coupon_dates=cpn_dates_np,
            )
        else:
            cf_du, cashflows = ntnb_cashflow_schedule(
                pricing_date, pd.Timestamp(mat), cal, cpn, face, freq,
            )

        P = float(np.sum(cashflows / (1.0 + ytm) ** (cf_du / 252.0)))
        D = _modified_duration(ytm, cf_du, cashflows, P)

        schedules.append((cf_du, cashflows))
        prices_act.append(P)
        durations.append(D)

    prices_act = np.array(prices_act)
    weights    = 1.0 / np.array(durations)
    weights   /= weights.sum()

    n = len(prices_act)

    def _ssr(beta: np.ndarray, lam: np.ndarray) -> float:
        total = 0.0
        for i, (cf_du, cashflows) in enumerate(schedules):
            p_model = _nss_bond_price(cf_du, cashflows, beta, lam)
            diff    = prices_act[i] - p_model
            total  += weights[i] * diff * diff
        return total

    _du_mid  = np.array([cf[0][-1] for cf in schedules], dtype=np.float64)
    _ytm_mid = np.array(
        [float(getattr(row, yield_col)) for row in df.itertuples(index=False)],
        dtype=np.float64,
    )
    b0_fallback = (np.array([0.12, -0.02, 0.02, -0.01]) if coupon_dates is not None
                   else np.array([0.06, -0.01, 0.01, -0.005]))

    # ── grid search: collect (beta, λ, ssr) for every valid (l1, l2) pair ───
    grid_candidates = []  # list of (ssr, x6_array)

    for l1 in lam1_grid:
        for l2 in lam2_grid:
            if l1 <= l2:
                continue
            t_mid = _du_mid / 252.0
            X     = _nss_design(t_mid, l1, l2)
            w_mid = 1.0 / _du_mid
            Xw    = X * w_mid[:, None]
            try:
                b0 = np.linalg.solve(Xw.T @ X, Xw.T @ _ytm_mid)
            except np.linalg.LinAlgError:
                b0 = b0_fallback

            res_b = minimize(
                lambda b: _ssr(b, np.array([l1, l2])),
                b0, method='Nelder-Mead',
                options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 500},
            )
            grid_candidates.append((res_b.fun, np.concatenate([res_b.x, [l1, l2]])))

    if not grid_candidates:
        raise RuntimeError("Grid search failed: no valid (λ₁, λ₂) found.")

    grid_candidates.sort(key=lambda x: x[0])
    top_starts = grid_candidates[:max(1, n_starts)]

    def _obj6(x):
        b, l = x[:4], x[4:]
        if np.any(l <= 0):
            return 1e12
        return _ssr(b, l)

    best_ssr = np.inf
    best_x6  = top_starts[0][1]   # fallback to best grid result

    for _init_ssr, _x6_init in top_starts:
        res6 = minimize(_obj6, _x6_init, method='Nelder-Mead',
                        options={'xatol': 1e-8, 'fatol': 1e-12,
                                 'maxiter': 10_000, 'adaptive': True})
        if res6.fun < best_ssr:
            best_ssr = res6.fun
            best_x6  = res6.x

    best_beta = best_x6[:4]
    best_lam  = np.abs(best_x6[4:])

    p_model = np.array([_nss_bond_price(cf_du, cf, best_beta, best_lam)
                        for cf_du, cf in schedules])
    resid   = prices_act - p_model
    rmse    = float(np.sqrt(np.mean(resid ** 2)))
    ss_tot  = float(np.sum((prices_act - prices_act.mean()) ** 2))
    r2      = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 1.0

    return NSSResult(beta=best_beta, lam=best_lam, rmse=rmse, r2=r2, n_obs=n)


# ── ANBIMA ETTJ PREF: bootstrap zeros → fit_nss ───────────────────────────────

def fit_nss_prefixado(
    bonds_df:     pd.DataFrame,
    cal:          Calendar,
    yield_col:    str = 'yield_base',
    coupon_col:   str = 'coupon',
    date_col:     str = 'date',
    coupon_dates  = None,
    method:       str = 'flatfwd',
    weights:      np.ndarray | None = None,
    lam1_grid:    np.ndarray | None = None,
    lam2_grid:    np.ndarray | None = None,
    n_starts:     int = 3,
) -> NSSResult:
    """
    Fit NSS to prefixado bonds (LTN + NTN-F) — ANBIMA ETTJ PREF methodology.

    Two-step approach (mirrors ANBIMA ETTJ construction):
    1. Build a PrefixadoCurve (flat-forward by default) from the panel.
    2. Bootstrap the zero (spot) curve via PrefixadoCurve.zero_curve().
    3. Fit NSS to the bootstrapped zero rates (WLS + Nelder-Mead on λ).

    Reduces RMSE vs published ETTJ vertices by ~40% relative to direct
    price-based fitting. For price-based NSS, use fit_nss_bonds() directly.
    """
    # Import here to avoid circular import (prefixado imports _schedules)
    from .prefixado import PrefixadoCurve

    date = pd.Timestamp(bonds_df[date_col].iloc[0])

    curve = PrefixadoCurve(
        bonds_df, cal,
        method=method,
        yield_col=yield_col,
        coupon_col=coupon_col,
        coupon_dates=coupon_dates,
    )
    zero = curve.zero_curve(date)

    return fit_nss(
        zero['du'].values,
        zero['zero_rate'].values,
        weights=weights,
        lam1_grid=lam1_grid,
        lam2_grid=lam2_grid,
        n_starts=n_starts,
    )


# ── ANBIMA ETTJ IPCA: fit_nss_bonds on NTN-B ──────────────────────────────────

def fit_nss_ntnb(
    bonds_df: pd.DataFrame,
    cal: Calendar,
    yield_col:    str = 'yield_base',
    coupon_col:   str = 'coupon',
    du_col:       str = 'du',
    maturity_col: str = 'maturity',
    date_col:     str = 'date',
    face:         float = 100.0,
    freq:         int   = 2,
    lam1_grid: np.ndarray | None = None,
    lam2_grid: np.ndarray | None = None,
    n_starts:  int = 3,
) -> NSSResult:
    """
    Fit NSS to NTN-B bonds — ANBIMA ETTJ IPCA methodology.

    Backward-compatible wrapper around fit_nss_bonds with face=100 and
    no coupon_dates (NTN-B schedules are derived analytically from maturity).
    """
    return fit_nss_bonds(
        bonds_df, cal,
        yield_col=yield_col, coupon_col=coupon_col, du_col=du_col,
        maturity_col=maturity_col, date_col=date_col,
        face=face, freq=freq, coupon_dates=None,
        lam1_grid=lam1_grid, lam2_grid=lam2_grid,
        n_starts=n_starts,
    )


# backward-compatible alias
fit_nss_anbima = fit_nss_prefixado
