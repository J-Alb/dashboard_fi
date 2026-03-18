"""
br_bonds — Brazilian sovereign bond pricing library
====================================================

A personal package for pricing and analysing Brazilian government bonds
(Títulos Públicos Federais) following ANBIMA/STN official methodology.

Supported instruments
---------------------
- LTN         : zero-coupon prefixado (nominal)
- NTN-F       : semi-annual coupon prefixado (nominal)
- NTN-B       : semi-annual IPCA-linked (real), cotação convention
- NTN-B-P     : zero-coupon IPCA-linked (NTN-B Principal)
- DI1 futures : (stub) B3 interbank rate futures

Quick start
-----------
    from bizdays import Calendar
    import pandas as pd
    from br_bonds import (
        PrefixadoCurve, NTNBCurve,
        price_ltn, price_ntnf, price_ntnb,
        fit_nss_prefixado, fit_nss_ntnb, nss_ytm, NSSResult,
        bond_duration, cm_tri, ret_index, build_cd_series,
    )

    cal = Calendar.load('ANBIMA')

    # Build yield curve
    curve = PrefixadoCurve(panel, cal, coupon_col='coupon',
                           coupon_dates=cdates_adj)
    curve.zero_curve(date)          # bootstrapped zeros
    curve.build_series(dates, 1260) # 5Y constant-maturity YTM + price

    # Fit NSS (ANBIMA ETTJ methodology)
    nss = fit_nss_prefixado(day_bonds, cal, coupon_dates=cdates_adj)
    nss.ytm(1260)    # 5Y zero rate
    nss.curve()      # full smooth DataFrame

    # Analytics
    mac, mod, dv01 = bond_duration(ytm=0.13, du=1260)
    tri = cm_tri(prices, ytms)
    eri = ret_index(prices, ytms, rf_rates=rf_3m)
"""

# ── pricing ────────────────────────────────────────────────────────────────────
from .prefixado import price_ltn, price_ntnf, ytm_ltn, ytm_ntnf, build_tri, PrefixadoCurve, NTNFCurve
from .ntnb import price_ntnb, ytm_ntnb, ntnb_tri, NTNBCurve
from .lft import price_lft, ytm_lft, LFTCurve, fetch_vna_selic

# ── interpolation schedules (semi-public) ─────────────────────────────────────
from ._interpolation import flatfwd_df, interp_yield
from ._schedules import (
    price_from_schedule,
    bond_cashflow_schedule,
    ntnb_cashflow_schedule,
    ntnb_coupon_dates,
)

# ── NSS yield curve ────────────────────────────────────────────────────────────
from .nss import (
    NSSResult,
    nss_ytm,
    fit_nss,
    fit_nss_bonds,
    fit_nss_prefixado,
    fit_nss_ntnb,
    fit_nss_anbima,    # backward-compatible alias for fit_nss_prefixado
)

# ── analytics ──────────────────────────────────────────────────────────────────
from .analytics import (
    bond_duration,
    add_duration,
    cm_tri,
    ret_index,
    du_for_mod_dur,
    build_cd_series,
    convexity_zerocoupon,
    convexity_coupon,
    risk_metrics,
)

# ── portfolio layer ────────────────────────────────────────────────────────────
from .portfolio import Instrument, Position, Portfolio

# ── VNA (NTN-B IPCA accrual) ──────────────────────────────────────────────────
from .vna import fetch_ipca_index, build_vna

# ── secondary market / implied curves ─────────────────────────────────────────
from .secondary import (
    TESOURO_MAP,
    load_secondary_data,
    get_pre_curve,
    get_ntnb_curve,
    build_breakeven_panel,
    build_breakeven_futures,
)

# ── derivatives ───────────────────────────────────────────────────────────────
from .derivatives.di1   import DI1Contract, di1_price, di1_dv01, DI1Curve
from .derivatives.dap   import DAPContract, dap_price, dap_dv01, dap_rate, dap_from_df
from .derivatives.ddi   import DDIContract, ddi_price, ddi_dv01, ddi_rate, ddi_from_df
from .derivatives.copom import CopomCurve, COPOM_DATES, fetch_di1, fetch_overnight

__all__ = [
    # pricing
    'price_ltn', 'price_ntnf', 'price_ntnb', 'price_lft', 'build_tri', 'ntnb_tri',
    # yield inversion
    'ytm_ltn', 'ytm_ntnf', 'ytm_ntnb', 'ytm_lft',
    # curves
    'PrefixadoCurve', 'NTNFCurve', 'NTNBCurve', 'LFTCurve',
    # data helpers
    'fetch_vna_selic',
    # schedules
    'price_from_schedule', 'bond_cashflow_schedule',
    'ntnb_cashflow_schedule', 'ntnb_coupon_dates',
    # interpolation
    'flatfwd_df', 'interp_yield',
    # NSS
    'NSSResult', 'nss_ytm', 'fit_nss',
    'fit_nss_bonds', 'fit_nss_prefixado', 'fit_nss_ntnb', 'fit_nss_anbima',
    # analytics
    'bond_duration', 'add_duration',
    'cm_tri', 'ret_index', 'du_for_mod_dur', 'build_cd_series',
    'convexity_zerocoupon', 'convexity_coupon', 'risk_metrics',
    # portfolio
    'Instrument', 'Position', 'Portfolio',
    # derivatives — DI1 futures
    'DI1Contract', 'di1_price', 'di1_dv01', 'DI1Curve',
    # derivatives — DAP (Cupom de IPCA futures)
    'DAPContract', 'dap_price', 'dap_dv01', 'dap_rate', 'dap_from_df',
    # derivatives — DDI (Cupom Cambial futures)
    'DDIContract', 'ddi_price', 'ddi_dv01', 'ddi_rate', 'ddi_from_df',
    # derivatives — COPOM pricing
    'CopomCurve', 'COPOM_DATES', 'fetch_di1', 'fetch_overnight',
    # VNA
    'fetch_ipca_index', 'build_vna',
    # secondary market / implied curves
    'TESOURO_MAP', 'load_secondary_data',
    'get_pre_curve', 'get_ntnb_curve',
    'build_breakeven_panel', 'build_breakeven_futures',
]
