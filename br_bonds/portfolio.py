"""
br_bonds/portfolio.py
─────────────────────
Unified position, portfolio, and risk layer for Brazilian fixed income.

Supported instruments
─────────────────────
  LTN  : zero-coupon prefixado
  NTNF : semi-annual 10% prefixado (NTN-F)
  NTNB : IPCA-linked real bond     (NTN-B)
  LFT  : Selic-linked floating     (LFT)
  DI1  : B3 CDI rate futures
  DAP  : B3 IPCA-coupon futures

Position modes
──────────────
  'rolling'      : constant-maturity daily roll (du_target fixed).
                   Daily carry is captured via the (1+ytm)^(1/252) accrual.
  'buy_and_hold' : hold a specific bond from entry_date to horizon.
                   Coupon payments are reinvested at the MTM price on payment date.

Quick start
───────────
    from br_bonds.portfolio import Instrument, Position, Portfolio
    from br_bonds.analytics import risk_metrics
    from bizdays import Calendar

    cal = Calendar.load('ANBIMA')

    # ── per-instrument analytics ───────────────────────────────────────────────
    ltn = Instrument('LTN', face=1000.0)
    ltn.price(ytm=0.135, du=1260)            # 1 240.xx R$
    ltn.dv01(ytm=0.135, du=1260)             # R$ per 1 bp
    ltn.convexity(ytm=0.135, du=1260)        # years²
    ltn.carry(ytm=0.135, du=1260, n_days=1)  # 1-day carry R$

    # ── rolling constant-maturity position ────────────────────────────────────
    # price_s and ytm_s are pd.Series from curve.build_series(dates, du_target=1260)
    pos = Position(ltn, quantity=1_000_000.0, mode='rolling', du_target=1260)
    tri = pos.tri_series(price_s, ytm_s)

    # ── buy-and-hold with coupon reinvestment ─────────────────────────────────
    ntnb = Instrument('NTNB', maturity='2035-05-15')
    pos  = Position(ntnb, quantity=100.0, mode='buy_and_hold', entry_date='2022-01-03')
    tri  = pos.tri_series(pu_s, ytm_s, cal=cal, vna_series=vna_s, reinvest=True)

    # ── portfolio ─────────────────────────────────────────────────────────────
    pf  = Portfolio([pos_ltn, pos_ntnb], weights=[0.6, 0.4])
    pf_tri = pf.tri_series({'LTN 5Y': tri_ltn, 'NTNB 2035': tri_ntnb})
    stats  = risk_metrics(pf_tri)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from bizdays import Calendar

from .prefixado import price_ltn, price_ntnf
from .ntnb import price_ntnb
from .lft import price_lft
from .derivatives.di1 import di1_price, di1_dv01
from .derivatives.dap import dap_price, dap_dv01
from ._schedules import ntnb_cashflow_schedule, bond_cashflow_schedule
from .analytics import convexity_zerocoupon, convexity_coupon, cm_tri


# ── constants ─────────────────────────────────────────────────────────────────

INSTRUMENT_TYPES = ('LTN', 'NTNF', 'NTNB', 'LFT', 'DI1', 'DAP')

_COUPON_DEFAULT: dict[str, float] = {
    'LTN': 0.00, 'NTNF': 0.10, 'NTNB': 0.06,
    'LFT': 0.00, 'DI1':  0.00, 'DAP':  0.00,
}
_FACE_DEFAULT: dict[str, float] = {
    'LTN': 1_000.0,  'NTNF': 1_000.0,   'NTNB': 100.0,
    'LFT': 1_000.0,  'DI1': 100_000.0,  'DAP':  100_000.0,
}


# ── coupon-date helpers ───────────────────────────────────────────────────────

def _ntnf_coupon_dates(
    start: pd.Timestamp,
    end:   pd.Timestamp,
    cal:   Calendar,
) -> list[pd.Timestamp]:
    """ANBIMA-adjusted NTN-F coupon dates (Jan 1 / Jul 1) in [start, end]."""
    raw = pd.date_range('2000-01-01', end + pd.DateOffset(days=30), freq='6MS')
    raw = raw[(raw >= start) & (raw <= end)]
    return sorted(
        pd.Timestamp(cal.adjust_next(d) if not cal.isbizday(d) else d)
        for d in raw
    )


def _ntnb_coupon_dates_for_bond(
    maturity: pd.Timestamp,
    start:    pd.Timestamp,
    end:      pd.Timestamp,
    cal:      Calendar,
) -> list[pd.Timestamp]:
    """
    Coupon dates for a specific NTN-B bond (6-month grid aligned to
    maturity day-of-month), ANBIMA-adjusted, in [start, end].
    """
    dates: list[pd.Timestamp] = []
    d = maturity
    while d >= start - pd.DateOffset(months=6):
        adj = pd.Timestamp(cal.adjust_next(d) if not cal.isbizday(d) else d)
        if start <= adj <= end:
            dates.append(adj)
        d = d - pd.DateOffset(months=6)
    return sorted(set(dates))


# ── Instrument ────────────────────────────────────────────────────────────────

class Instrument:
    """
    Static descriptor for a Brazilian fixed income instrument.

    Parameters
    ----------
    itype    : 'LTN' | 'NTNF' | 'NTNB' | 'LFT' | 'DI1' | 'DAP'
    maturity : contractual maturity date (optional for generic constant-maturity use)
    coupon   : annual coupon rate (decimal); None → market default
    face     : face value; None → market default
    """

    def __init__(
        self,
        itype:    str,
        maturity: pd.Timestamp | str | None = None,
        coupon:   float | None = None,
        face:     float | None = None,
    ) -> None:
        self.itype    = itype.upper()
        self.maturity = pd.Timestamp(maturity) if maturity is not None else None
        self.coupon   = coupon if coupon is not None else _COUPON_DEFAULT[self.itype]
        self.face     = face   if face   is not None else _FACE_DEFAULT[self.itype]
        if self.itype not in INSTRUMENT_TYPES:
            raise ValueError(
                f"Unknown type '{self.itype}'. Must be one of {INSTRUMENT_TYPES}"
            )

    # ── maturity helpers ─────────────────────────────────────────────────────

    def mat_adj(self, cal: Calendar) -> pd.Timestamp:
        """Maturity date adjusted to next ANBIMA bizday."""
        if self.maturity is None:
            raise ValueError(f"{self.itype}: maturity required")
        m = self.maturity
        return pd.Timestamp(cal.adjust_next(m) if not cal.isbizday(m) else m)

    def du(self, date: pd.Timestamp, cal: Calendar) -> int:
        """Business days from date to (adjusted) maturity."""
        return cal.bizdays(pd.Timestamp(date), self.mat_adj(cal))

    # ── pricing ──────────────────────────────────────────────────────────────

    def price(
        self,
        ytm: float,
        du:  int,
        vna: float | None = None,
    ) -> float:
        """
        Mark-to-market price in R$.

        NTNB: cotação × vna / 100 (PU) when vna provided; else cotação.
        LFT : PU = price_lft(ytm, du, vna); requires vna.
        DI1 / DAP: contract price (notional 100 000).
        """
        t = self.itype
        if t == 'LTN':
            return price_ltn(ytm, du, face=self.face)
        if t == 'NTNF':
            return price_ntnf(ytm, du, coupon=self.coupon, face=self.face)
        if t == 'NTNB':
            cot = price_ntnb(ytm, du, coupon_real=self.coupon, face=self.face)
            return cot * vna / 100.0 if vna is not None else cot
        if t == 'LFT':
            if vna is None:
                raise ValueError("LFT.price() requires vna (VNA-Selic)")
            return price_lft(ytm, du, vna)
        if t == 'DI1':
            return di1_price(ytm, du)
        if t == 'DAP':
            return dap_price(ytm, du)
        raise ValueError(t)

    # ── risk analytics ───────────────────────────────────────────────────────

    def dv01(
        self,
        ytm: float,
        du:  int,
        vna: float | None = None,
    ) -> float:
        """
        DV01: R$ price change per +1 bp rise in yield (positive convention).

        Uses dedicated formulas for DI1/DAP; central finite difference ±0.5 bp
        for all others. NTNB result is in R$ when vna provided, else cotação.
        """
        if self.itype == 'DI1':
            return di1_dv01(ytm, du)
        if self.itype == 'DAP':
            return abs(dap_dv01(ytm, du))
        eps  = 5e-5          # 0.5 bp → symmetric range = 1 bp
        p_up = self.price(ytm + eps, du, vna=vna)
        p_dn = self.price(ytm - eps, du, vna=vna)
        return -(p_up - p_dn) / 2.0

    def convexity(
        self,
        ytm:  float,
        du:   int,
        vna:  float | None = None,
        date: pd.Timestamp | None = None,
        cal:  Calendar | None = None,
    ) -> float:
        """
        Convexity in years².

        Zero-coupon instruments (LTN, LFT, DI1, DAP): exact closed form.
        Coupon bonds (NTNF, NTNB): exact cashflow-based when date+cal provided;
        otherwise uniform-spacing approximation (126-bday coupon intervals).
        """
        t = self.itype

        if t in ('LTN', 'LFT', 'DI1', 'DAP'):
            return convexity_zerocoupon(ytm, du)

        if t == 'NTNF':
            if date is not None and cal is not None and self.maturity is not None:
                raw_cpn = pd.date_range('2000-01-01', '2060-01-01', freq='6MS')
                cpn_dt  = np.array([
                    cal.adjust_next(d) if not cal.isbizday(d) else d
                    for d in raw_cpn
                ], dtype='datetime64[D]')
                mat_np  = np.datetime64(self.maturity.date(), 'D')
                date_np = np.datetime64(pd.Timestamp(date).date(), 'D')
                cf_du, cfs = bond_cashflow_schedule(
                    float(du), self.coupon, self.face, 2,
                    date_np=date_np, mat_np=mat_np,
                    cal=cal, coupon_dates=cpn_dt,
                )
                return convexity_coupon(cf_du, cfs, ytm)
            # uniform-spacing fallback
            cf_du, cfs = _uniform_schedule(du, self.coupon, self.face)
            return convexity_coupon(cf_du, cfs, ytm)

        if t == 'NTNB':
            if date is not None and cal is not None and self.maturity is not None:
                cf_du, cfs = ntnb_cashflow_schedule(
                    pd.Timestamp(date), self.maturity, cal,
                    coupon=self.coupon, face=self.face,
                )
                return convexity_coupon(cf_du, cfs, ytm)
            cf_du, cfs = _uniform_schedule(du, self.coupon, self.face)
            return convexity_coupon(cf_du, cfs, ytm)

        raise ValueError(t)

    def carry(
        self,
        ytm:      float,
        du:       int,
        n_days:   int = 1,
        vna:      float | None = None,
        vna_next: float | None = None,
    ) -> float:
        """
        Carry over n_days business days: price appreciation assuming yield unchanged.

        For LFT: pass vna (today) and vna_next (after n_days) to capture Selic
                 accrual. Without vna_next, approximates with the same vna.
        For NTNB with vna: returns carry in R$.
        Does not include coupon cashflows that fall within the period.
        """
        if self.itype == 'LFT':
            vna_n = vna_next if vna_next is not None else vna
            if vna is None:
                raise ValueError("LFT carry requires vna")
            return price_lft(ytm, du - n_days, vna_n) - price_lft(ytm, du, vna)
        return self.price(ytm, du - n_days, vna=vna) - self.price(ytm, du, vna=vna)

    def duration(
        self,
        ytm: float,
        du:  int,
        vna: float | None = None,
    ) -> tuple[float, float, float]:
        """
        (Macaulay duration, Modified duration, DV01).
        mac_dur, mod_dur in business years; DV01 in R$ per 1 bp.
        """
        p   = self.price(ytm, du, vna=vna)
        dv  = self.dv01(ytm,  du, vna=vna)
        mod = dv / (p * 1e-4) if p != 0 else 0.0
        mac = mod * (1.0 + ytm)
        return mac, mod, dv

    # ── coupon-date / payment helpers ─────────────────────────────────────────

    def coupon_dates(
        self,
        start: pd.Timestamp | str,
        end:   pd.Timestamp | str,
        cal:   Calendar,
    ) -> list[pd.Timestamp]:
        """
        Coupon payment dates in [start, end].

        Returns [] for zero-coupon and futures instruments.
        NTNF: Jan 1 / Jul 1 ANBIMA-adjusted.
        NTNB: 6-month grid aligned to maturity day-of-month (requires maturity).
        """
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if self.itype in ('LTN', 'LFT', 'DI1', 'DAP'):
            return []
        if self.itype == 'NTNF':
            return _ntnf_coupon_dates(s, e, cal)
        if self.itype == 'NTNB':
            if self.maturity is None:
                raise ValueError("NTNB.coupon_dates() requires maturity")
            return _ntnb_coupon_dates_for_bond(self.maturity, s, e, cal)
        return []

    def coupon_payment(self, vna: float | None = None) -> float:
        """
        Coupon cash flow per 1 unit of face on a payment date.

        NTNF : face × (1.10^0.5 − 1) ≈ 48.81 R$
        NTNB : face × (1.06^0.5 − 1) × vna/100 in R$ when vna provided;
               else in cotação terms.
        All others: 0.0
        """
        if self.itype == 'NTNF':
            return self.face * ((1.0 + self.coupon) ** 0.5 - 1.0)
        if self.itype == 'NTNB':
            cot = self.face * ((1.0 + self.coupon) ** 0.5 - 1.0)
            return cot * vna / 100.0 if vna is not None else cot
        return 0.0

    def analytics_row(
        self,
        ytm:  float,
        du:   int,
        vna:  float | None = None,
        date: pd.Timestamp | None = None,
        cal:  Calendar | None = None,
    ) -> dict:
        """Single-date analytics dict: price, dv01, convexity, carry, durations."""
        p    = self.price(ytm, du, vna=vna)
        dv   = self.dv01(ytm,  du, vna=vna)
        conv = self.convexity(ytm, du, vna=vna, date=date, cal=cal)
        cry  = self.carry(ytm, du, n_days=1, vna=vna)
        mod  = dv / (p * 1e-4) if p != 0 else 0.0
        mac  = mod * (1.0 + ytm)
        return {
            'ytm': ytm, 'du': du,
            'price': p, 'dv01': dv, 'convexity': conv,
            'carry_1d': cry, 'mac_dur': mac, 'mod_dur': mod,
        }

    def __repr__(self) -> str:
        mat = f", mat={self.maturity.date()}" if self.maturity else ''
        return (
            f"Instrument({self.itype}{mat}, "
            f"cpn={self.coupon*100:.1f}%, face={self.face:,.0f})"
        )


# ── Position ──────────────────────────────────────────────────────────────────

class Position:
    """
    A single-instrument position with total return computation.

    Parameters
    ----------
    instrument : Instrument descriptor
    quantity   : face value in R$ (or contracts for DI1/DAP)
    mode       : 'rolling' — constant-maturity daily roll
                 'buy_and_hold' — hold from entry_date, reinvest coupons
    du_target  : target business days for rolling positions
    entry_date : start date for buy_and_hold (TRI clipped here)
    entry_ytm  : entry yield in decimal (informational)
    label      : display label; defaults to "{itype} {maturity or du_target}"
    """

    def __init__(
        self,
        instrument: Instrument,
        quantity:   float = 1.0,
        mode:       str   = 'rolling',
        du_target:  int | None = None,
        entry_date: pd.Timestamp | str | None = None,
        entry_ytm:  float | None = None,
        label:      str | None = None,
    ) -> None:
        if mode not in ('rolling', 'buy_and_hold'):
            raise ValueError("mode must be 'rolling' or 'buy_and_hold'")
        self.instrument = instrument
        self.quantity   = quantity
        self.mode       = mode
        self.du_target  = du_target
        self.entry_date = pd.Timestamp(entry_date) if entry_date is not None else None
        self.entry_ytm  = entry_ytm
        if label is None:
            mat = (f" {instrument.maturity.strftime('%Y-%m-%d')}"
                   if instrument.maturity is not None
                   else f" du={du_target}")
            label = f"{instrument.itype}{mat}"
        self.label = label

    def tri_series(
        self,
        prices:     pd.Series,
        ytms:       pd.Series,
        cal:        Calendar | None = None,
        vna_series: pd.Series | None = None,
        reinvest:   bool = True,
        rf_series:  pd.Series | None = None,
    ) -> pd.Series:
        """
        Total return index (TRI), rebased to 100 at first valid observation.

        Rolling mode
        ────────────
        Daily return = (P_t / P_{t-1}) × (1 + ytm_{t-1})^(1/252).
        The accrual term captures the daily carry (hold-to-maturity return if
        yields are stable). Coupon reinvestment is implicit.

        Buy-and-hold mode
        ─────────────────
        Daily return = (P_t + C_t) / P_{t-1}  where C_t = coupon paid on day t.
        On coupon dates the payment is added to the numerator, which is
        equivalent to immediately reinvesting the coupon at the ex-coupon price.
        Position is clipped to [entry_date, end].

        Parameters
        ----------
        prices     : daily MTM prices indexed by date
                     (PU for NTNB when vna_series provided; else cotação)
        ytms       : annual YTMs (decimal) aligned with prices
        cal        : Calendar — required for buy_and_hold coupon detection
        vna_series : daily VNA indexed by date (NTNB/LFT coupon amounts)
        reinvest   : if True (default), reinvest coupon payments
        rf_series  : optional daily risk-free rates for excess-return TRI
        """
        prices = prices.dropna()
        ytms   = ytms.reindex(prices.index)

        # ── rolling mode ──────────────────────────────────────────────────────
        if self.mode == 'rolling':
            rf_arr = rf_series.reindex(prices.index).values if rf_series is not None else None
            if rf_arr is not None:
                from .analytics import ret_index
                tri_arr = ret_index(prices.values, ytms.values, rf_arr)
            else:
                tri_arr = cm_tri(prices.values, ytms.values)
            return pd.Series(tri_arr, index=prices.index, name=self.label)

        # ── buy_and_hold mode ─────────────────────────────────────────────────
        if self.entry_date is not None:
            prices = prices.loc[prices.index >= self.entry_date]
            ytms   = ytms.loc[ytms.index   >= self.entry_date]

        if len(prices) == 0:
            return pd.Series(dtype=float, name=self.label)

        # collect coupon dates within the series range
        cpn_set: set[pd.Timestamp] = set()
        if reinvest and cal is not None and self.instrument.coupon > 0:
            cpn_set = set(
                self.instrument.coupon_dates(prices.index[0], prices.index[-1], cal)
            )

        p_arr   = prices.values.astype(float)
        idx     = prices.index
        n       = len(p_arr)
        tri_arr = np.full(n, np.nan)
        tri_arr[0] = 100.0
        prev   = 100.0
        prev_p = p_arr[0]

        for i in range(1, n):
            if np.isnan(p_arr[i]):
                tri_arr[i] = prev
                continue

            cpn = 0.0
            if reinvest and idx[i] in cpn_set:
                vna_i = (float(vna_series.loc[idx[i]])
                         if vna_series is not None and idx[i] in vna_series.index
                         else None)
                cpn = self.instrument.coupon_payment(vna=vna_i)

            tri_arr[i] = prev * (p_arr[i] + cpn) / prev_p
            prev   = tri_arr[i]
            prev_p = p_arr[i]

        return pd.Series(tri_arr, index=idx, name=self.label)

    def analytics(
        self,
        date: pd.Timestamp,
        ytm:  float,
        du:   int,
        vna:  float | None = None,
        cal:  Calendar | None = None,
    ) -> dict:
        """Per-date analytics scaled by position quantity."""
        row   = self.instrument.analytics_row(ytm, du, vna=vna, date=date, cal=cal)
        scale = self.quantity / self.instrument.face
        return {
            'label'    : self.label,
            'quantity' : self.quantity,
            **row,
            'dv01_pos' : row['dv01']     * scale,
            'conv_pos' : row['convexity'] * scale,
            'carry_pos': row['carry_1d'] * scale,
        }

    def __repr__(self) -> str:
        return (
            f"Position({self.label!r}, qty={self.quantity:,.0f}, "
            f"mode={self.mode})"
        )


# ── Portfolio ─────────────────────────────────────────────────────────────────

class Portfolio:
    """
    Weighted combination of positions.

    Parameters
    ----------
    positions : list of Position objects
    weights   : allocation weights (must sum to 1); None = equal weight

    Notes
    -----
    Portfolio TRI is constructed from the weighted sum of daily returns,
    which implies daily rebalancing to maintain target weights.
    For a static (buy-and-hold) allocation, supply each position's TRI
    computed independently and combine here.
    """

    def __init__(
        self,
        positions: list[Position],
        weights:   list[float] | None = None,
    ) -> None:
        self.positions = list(positions)
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            if not np.isclose(w.sum(), 1.0, atol=1e-6):
                raise ValueError(f"Weights must sum to 1; got {w.sum():.6f}")
            self.weights = list(w)
        else:
            n = len(positions)
            self.weights = [1.0 / n] * n

    def tri_series(
        self,
        tri_dict: dict[str, pd.Series],
    ) -> pd.Series:
        """
        Portfolio TRI from a dict of per-position TRIs.

        Parameters
        ----------
        tri_dict : {position.label: pd.Series} — one TRI per position.
                   Missing labels are silently skipped (weights renormalised).

        Returns
        -------
        pd.Series — portfolio TRI rebased to 100, on the intersection of dates.
        """
        present = [(p, w) for p, w in zip(self.positions, self.weights)
                   if p.label in tri_dict]
        if not present:
            return pd.Series(dtype=float, name='Portfolio')

        pos_list, wts = zip(*present)
        wts  = np.asarray(wts, dtype=float)
        wts /= wts.sum()

        combined = pd.concat(
            [tri_dict[p.label] for p in pos_list],
            axis=1,
            keys=[p.label for p in pos_list],
        ).ffill()

        rets   = combined.pct_change().fillna(0.0)
        pf_ret = (rets.values * wts).sum(axis=1)
        pf_tri = pd.Series(
            100.0 * (1.0 + pf_ret).cumprod(),
            index=combined.index,
            name='Portfolio',
        )
        return pf_tri

    def analytics_summary(
        self,
        analytics_list: list[dict],
    ) -> pd.DataFrame:
        """
        Aggregate per-position analytics into a summary table.

        Parameters
        ----------
        analytics_list : list of dicts from Position.analytics(), one per position.

        Returns
        -------
        pd.DataFrame with one row per position plus a 'Total' row for
        additive columns (dv01_pos, conv_pos, carry_pos, quantity).
        """
        df = pd.DataFrame(analytics_list)
        if df.empty:
            return df
        totals = {
            'label'    : 'Total',
            'quantity' : df['quantity'].sum(),
            'dv01_pos' : df['dv01_pos'].sum(),
            'conv_pos' : df['conv_pos'].sum(),
            'carry_pos': df['carry_pos'].sum(),
        }
        return pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    def __repr__(self) -> str:
        lines = [f"Portfolio({len(self.positions)} positions)"]
        for p, w in zip(self.positions, self.weights):
            lines.append(f"  {w*100:5.1f}%  {p}")
        return '\n'.join(lines)


# ── private helpers ───────────────────────────────────────────────────────────

def _uniform_schedule(
    du: int,
    coupon: float,
    face: float,
    step: int = 126,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform semi-annual cashflow schedule (126-bday spacing)."""
    cf_du = np.arange(step, du + step, step, dtype=float)
    cf_du = cf_du[cf_du <= du + 0.5]
    if len(cf_du) == 0 or cf_du[-1] < du:
        cf_du = np.append(cf_du, float(du))
    c   = face * ((1.0 + coupon) ** 0.5 - 1.0)
    cfs = np.full(len(cf_du), c)
    cfs[-1] += face
    return cf_du, cfs
