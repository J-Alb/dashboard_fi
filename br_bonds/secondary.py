"""
br_bonds/secondary.py
─────────────────────
Implied real yield curve and inflation breakeven panel from BCB secondary
market data, using flat-forward interpolation (ANBIMA methodology).

Based on implicita_v2.py — flat-forward only (no NSS fitting).

Public API
----------
TESOURO_MAP
    Dict mapping raw sigla strings to canonical bond-type labels.

load_secondary_data(file_path, codes)
    Load and clean BCB secondary market parquet file.

get_pu_series(raw, sigla, mat_str, isin)
    Volume-weighted average PU per trading date for a specific bond (ISIN).

get_pre_curve(dta, date)
    Filter prefixado (LTN + NTN-F) bonds for a given date.

get_ntnb_curve(dta, date)
    Filter NTN-B bonds for a given date.

parse_ativo(code, itype, cal)
    Parse a B3 ATIVO code (e.g. 'DAPK25', 'DI1K25') to its expiry Timestamp.

build_breakeven_panel(dta, vna_series, dates, du_brkv, cdates, n_jobs, verbose)
    Build daily breakeven panel for a list of dates (parallel-safe).
"""

from __future__ import annotations

import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
from bizdays import Calendar

from ._interpolation import flatfwd_batch

_CAL = Calendar.load('ANBIMA')


# ── canonical bond-type mapping ───────────────────────────────────────────────

TESOURO_MAP: dict[str, str] = {
    # Selic
    'TESOURO SELIC': 'LFT', 'LFT': 'LFT',
    # Prefixado sem cupom
    'TESOURO PREFIXADO': 'LTN', 'LTN': 'LTN',
    # Prefixado com cupom
    'TESOURO PREFIXADO COM JUROS SEMESTRAIS': 'NTN-F',
    'NTN-F': 'NTN-F', 'NTNF': 'NTN-F',
    # IPCA sem cupom
    'TESOURO IPCA+': 'NTN-B Principal', 'TESOURO IPCA +': 'NTN-B Principal',
    'NTN-B PRINCIPAL': 'NTN-B Principal', 'NTNBP': 'NTN-B Principal',
    'NTN-BP': 'NTN-B Principal',
    # IPCA com cupom
    'TESOURO IPCA+ COM JUROS SEMESTRAIS': 'NTN-B',
    'TESOURO IPCA + COM JUROS SEMESTRAIS': 'NTN-B',
    'NTN-B': 'NTN-B', 'NTNB': 'NTN-B',
    # RendA+ / Educa+
    'TESOURO RENDA+': 'NTN-B1', 'TESOURO RENDA+ APOSENTADORIA EXTRA': 'NTN-B1',
    'TESOURO EDUCA+': 'NTN-B1', 'NTN-B1': 'NTN-B1',
    # Histórico
    'TESOURO IGPM+ COM JUROS SEMESTRAIS': 'NTN-C',
    'TESOURO IGP-M+ COM JUROS SEMESTRAIS': 'NTN-C',
    'NTN-C': 'NTN-C', 'NTNC': 'NTN-C',
}

_MAP_NAMES = {
    'data_mov':   'date',
    'vencimento': 'maturity',
    'sigla':      'bond_type',
    'taxa_med':   'yield_base',
    'pu_med':     'price_base',
    'pu_lastro':  'pu_lastro',
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_secondary_data(
    file_path: str | Path,
    codes: list[int] | None = None,
) -> pd.DataFrame:
    """
    Load and clean BCB secondary market parquet file.

    Parameters
    ----------
    file_path : path to the raw BCB parquet file.
    codes : BCB instrument codes to keep (default: [100000, 950199, 760199]).
        100000 = LTN, 950199 = NTN-F, 760199 = NTN-B.

    Returns
    -------
    pd.DataFrame with columns: date, maturity, bond_type, yield_base,
        price_base, pu_lastro, codigo, valor_par, num_de_oper, du, coupon.
    """
    if codes is None:
        codes = [100000, 950199, 760199]

    raw = pd.read_parquet(file_path, filters=[('codigo', 'in', codes)])
    raw['sigla'] = raw['sigla'].str.strip().str.upper().map(TESOURO_MAP)

    keep = list(_MAP_NAMES.keys()) + ['codigo', 'valor_par', 'num_de_oper']
    raw  = raw[keep].rename(columns=_MAP_NAMES)
    raw  = raw[~raw['bond_type'].isna()].reset_index(drop=True)
    raw.sort_values('date', inplace=True)

    raw['yield_base'] = raw['yield_base'] / 100.0
    raw['coupon']     = raw['bond_type'].map({'LTN': 0.0, 'NTN-F': 0.10, 'NTN-B': 0.06})

    cal = _CAL

    def _mat_adj(mat):
        return cal.adjust_next(mat) if not cal.isbizday(mat) else mat

    raw['du'] = raw.apply(
        lambda r: cal.bizdays(r['date'], _mat_adj(r['maturity'])), axis=1
    )
    return raw


def get_pu_series(
    raw: pd.DataFrame,
    sigla: str,
    mat_str: str,
    isin: str,
) -> pd.Series:
    """
    Volume-weighted average PU per trading date for a specific bond (ISIN).

    Parameters
    ----------
    raw     : raw BCB secondary market DataFrame with original column names
              (data_mov, sigla, vencimento, codigo_isin, quant_negociada, pu_med).
    sigla   : bond type string, e.g. 'NTN-B', 'NTN-F', 'LTN'.
    mat_str : maturity date string, e.g. '2035-05-15'.
    isin    : ISIN code, e.g. 'BRSTNCNTB0O7'.

    Returns
    -------
    pd.Series of volume-weighted PU indexed by date, sorted and NaN-dropped.
    """
    sub = raw[
        (raw['sigla'] == sigla) &
        (raw['vencimento'] == mat_str) &
        (raw['codigo_isin'] == isin)
    ].copy()
    sub['wt'] = sub['quant_negociada'] * sub['pu_med']
    g = sub.groupby('data_mov').agg(
        wt_sum=('wt', 'sum'),
        vol=('quant_negociada', 'sum'),
    )
    return (g['wt_sum'] / g['vol']).rename('pu').sort_index().dropna()


# ── curve filter functions ─────────────────────────────────────────────────────

def get_pre_curve(dta: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """
    Filter prefixado (LTN + NTN-F) bonds for *date* with basic quality filters.
    """
    day_ltn = dta[
        (dta['bond_type'] == 'LTN') &
        (dta['date'] == date) &
        (dta['num_de_oper'] >= 2)
    ].copy()
    day_ltn = day_ltn[day_ltn['yield_base'].between(0, 0.60)]

    # Spike filter: drop isolated valley/peak/tail nodes
    if len(day_ltn) >= 3:
        _srt  = day_ltn.sort_values('du').reset_index()
        _drop = []
        n     = len(_srt)
        for i in range(1, n):
            curr = _srt['yield_base'].iloc[i]
            if i == n - 1:
                if abs(curr - _srt['yield_base'].iloc[i - 1]) > 0.025:
                    _drop.append(_srt['index'].iloc[i])
            else:
                prev = _srt['yield_base'].iloc[i - 1]
                nxt  = _srt['yield_base'].iloc[i + 1]
                if (prev - curr > 0.020) and (nxt - curr > 0.020):
                    _drop.append(_srt['index'].iloc[i])
                elif (curr - prev > 0.020) and (curr - nxt > 0.020):
                    _drop.append(_srt['index'].iloc[i])
        if _drop:
            day_ltn = day_ltn.drop(index=_drop)

    _ntnf_all = dta[
        (dta['bond_type'] == 'NTN-F') &
        (dta['date'] == date) &
        (dta['codigo'] == 950199) &
        (dta['num_de_oper'] >= 2)
    ].copy()
    _cot_ntnf = _ntnf_all['price_base'] / _ntnf_all['valor_par']
    day_ntnf  = (
        _ntnf_all[_cot_ntnf.between(0.50, 2.00)]
        .sort_values('price_base', ascending=False)
        .drop_duplicates(subset=['maturity'])
        .reset_index(drop=True)
    )
    return pd.concat([day_ntnf, day_ltn], axis=0)


def get_ntnb_curve(dta: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Filter NTN-B bonds for *date* with basic quality filters."""
    _ntnb_all = dta[
        (dta['bond_type'] == 'NTN-B') &
        (dta['date'] == date) &
        (dta['codigo'] == 760199) &
        (dta['num_de_oper'] >= 2)
    ].copy()
    _cotacao = _ntnb_all['price_base'] / _ntnb_all['valor_par']
    return _ntnb_all[_cotacao.between(0.50, 2.00)].copy()


# ── yield inversion helpers ────────────────────────────────────────────────────

def _compute_yields_ntnb(
    raw: pd.DataFrame,
    date: pd.Timestamp,
    vna_series: pd.Series,
    cal: Calendar,
) -> pd.DataFrame:
    from br_bonds import ytm_ntnb as _ytm_ntnb

    vna_t = vna_series.get(date)
    if vna_t is None:
        return pd.DataFrame()

    def _ytm(row):
        if pd.isna(row['price_base']):
            return np.nan
        try:
            return _ytm_ntnb(row['price_base'], vna_t, date,
                             pd.Timestamp(row['maturity']), cal)
        except Exception:
            return np.nan

    out = raw.copy()
    out['yield_base'] = out.apply(_ytm, axis=1)
    return (
        out.sort_values('price_base', ascending=False)
        .drop_duplicates(subset=['maturity'])
        .reset_index(drop=True)
        .dropna(subset=['yield_base'])
    )


def _compute_yields_pre(
    raw: pd.DataFrame,
    date: pd.Timestamp,
    cal: Calendar,
    cdates: np.ndarray,
) -> pd.DataFrame:
    from br_bonds import ytm_ntnf as _ytm_ntnf, ytm_ltn as _ytm_ltn

    out      = raw.copy().reset_index(drop=True)
    ntnf_idx = out.index[out['bond_type'] == 'NTN-F']

    def _ntnf_ytm(row):
        if pd.isna(row['price_base']):
            return np.nan
        try:
            return _ytm_ntnf(row['price_base'], float(row['du']),
                             date, pd.Timestamp(row['maturity']),
                             cal, cdates)
        except Exception:
            return np.nan

    out.loc[ntnf_idx, 'yield_base'] = out.loc[ntnf_idx].apply(_ntnf_ytm, axis=1)
    ltn_mask = (out['bond_type'] == 'LTN') & out['yield_base'].isna()
    out.loc[ltn_mask, 'yield_base'] = out.loc[ltn_mask].apply(
        lambda r: _ytm_ltn(r['price_base'], r['du']), axis=1
    )
    return out.dropna(subset=['yield_base'])


# ── flat-forward interpolation on a zero-curve DataFrame ──────────────────────

def _flatfwd_ytm(zc_df: pd.DataFrame, du_arr: np.ndarray) -> np.ndarray:
    """
    Flat-forward interpolation on a zero-curve DataFrame (columns: du, zero_rate).

    ANBIMA convention: flat extrapolation at both ends.
    - du < dus[0]  → first zero rate (flat at short end)
    - du > dus[-1] → last zero rate  (flat at long end)
    """
    dus    = zc_df['du'].values.astype(float)
    zrs    = zc_df['zero_rate'].values.astype(float)
    dfs    = (1.0 + zrs) ** (-dus / 252.0)
    du_arr = np.asarray(du_arr, dtype=float)

    lo  = du_arr <= dus[0]
    hi  = du_arr >= dus[-1]
    mid = ~lo & ~hi

    out      = np.empty(len(du_arr))
    out[lo]  = zrs[0]
    out[hi]  = zrs[-1]
    if mid.any():
        df_mid    = flatfwd_batch(dus, dfs, du_arr[mid])
        out[mid]  = df_mid ** (-252.0 / du_arr[mid]) - 1.0
    return out


# ── single-date worker (parallel-safe) ────────────────────────────────────────

def _process_date(
    date: pd.Timestamp,
    grp: pd.DataFrame,
    vna_series: pd.Series,
    du_brkv: np.ndarray,
    cdates: np.ndarray,
    dap_grp: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    One-date worker: invert yields → bootstrapped zero curves → flat-forward
    breakeven. No NSS fitting — pure flat-forward interpolation.

    Parameters
    ----------
    dap_grp : optional DAP settlement group for this date (DATA, VENCIMENTO,
        AJUSTE_ATUAL). When supplied, DAP zero rates are merged into the
        NTN-B bootstrapped real zero curve as additional anchors — NTN-B
        nodes take priority at overlapping maturities.

    Returns a DataFrame or None if the date should be skipped.
    """
    from bizdays import Calendar
    from br_bonds import NTNBCurve, PrefixadoCurve

    cal = Calendar.load('ANBIMA')

    try:
        # ── NTN-B ──────────────────────────────────────────────────────────────
        raw_ntnb   = get_ntnb_curve(grp, date)
        day_ntnb_d = _compute_yields_ntnb(raw_ntnb, date, vna_series, cal)
        day_ntnb_d = day_ntnb_d[day_ntnb_d['du'] >= 126].reset_index(drop=True)
        day_ntnb_d = day_ntnb_d[
            day_ntnb_d['yield_base'].between(-0.01, 0.14)
        ].reset_index(drop=True)

        # Valley / peak / tail filter on input yields
        if len(day_ntnb_d) >= 2:
            _srt     = day_ntnb_d.sort_values('du').reset_index()
            _drop_nb = []
            _n       = len(_srt)
            if _srt['yield_base'].iloc[0] - _srt['yield_base'].iloc[1] > 0.020:
                _drop_nb.append(_srt['index'].iloc[0])
            for _i in range(1, _n):
                _curr = _srt['yield_base'].iloc[_i]
                if _i == _n - 1:
                    if abs(_curr - _srt['yield_base'].iloc[_i - 1]) > 0.020:
                        _drop_nb.append(_srt['index'].iloc[_i])
                else:
                    _prev = _srt['yield_base'].iloc[_i - 1]
                    _nxt  = _srt['yield_base'].iloc[_i + 1]
                    if (_prev - _curr > 0.013) and (_nxt - _curr > 0.013):
                        _drop_nb.append(_srt['index'].iloc[_i])
                    elif (_curr - _prev > 0.013) and (_curr - _nxt > 0.013):
                        _drop_nb.append(_srt['index'].iloc[_i])
            if _drop_nb:
                day_ntnb_d = day_ntnb_d.drop(index=_drop_nb).reset_index(drop=True)

        if len(day_ntnb_d) < 4:
            return None

        ntnb_crv = NTNBCurve(day_ntnb_d, cal, method='flatfwd',
                             yield_col='yield_base', coupon_col='coupon',
                             vna_series=vna_series)
        zc_ntnb  = ntnb_crv.zero_curve(date)
        if zc_ntnb is None or len(zc_ntnb) < 4:
            return None
        zc_ntnb = zc_ntnb[
            zc_ntnb['zero_rate'].between(-0.01, 0.14)
        ].reset_index(drop=True)
        if len(zc_ntnb) < 4:
            return None

        # Post-bootstrap zero-curve outlier filter (valley/peak, 1.5pp threshold)
        _zs      = zc_ntnb.sort_values('du').reset_index(drop=True)
        _drop_zc = []
        _nz      = len(_zs)
        for _i in range(1, _nz - 1):
            _zc_v = _zs['zero_rate'].iloc[_i]
            _zp_v = _zs['zero_rate'].iloc[_i - 1]
            _zn_v = _zs['zero_rate'].iloc[_i + 1]
            if (_zp_v - _zc_v > 0.015) and (_zn_v - _zc_v > 0.015):
                _drop_zc.append(_i)
            elif (_zc_v - _zp_v > 0.015) and (_zc_v - _zn_v > 0.015):
                _drop_zc.append(_i)
        if _drop_zc:
            zc_ntnb = _zs.drop(index=_drop_zc).reset_index(drop=True)
        if len(zc_ntnb) < 4:
            return None

        # ── DAP augmentation (optional) ────────────────────────────────────────
        # DAP expires on the 15th of the month (same as NTN-B), so with
        # expiry_day=15 the du values align exactly.  concat + drop_duplicates
        # (NTN-B first → keep='first') correctly prefers bootstrapped bond nodes
        # and lets DAP fill genuine gaps (short end, inter-maturity slots).
        if dap_grp is not None and len(dap_grp) > 0:
            zc_dap = _futures_to_zero_curve(dap_grp, date, cal, expiry_day=15)
            if len(zc_dap) > 0:
                zc_ntnb = (
                    pd.concat([zc_ntnb, zc_dap], ignore_index=True)
                    .drop_duplicates(subset=['du'], keep='first')  # NTN-B wins
                    .sort_values('du')
                    .reset_index(drop=True)
                )

        # ── Nominal zero curve (LTN/NTN-F bootstrapped) ───────────────────────
        raw_pre   = get_pre_curve(grp, date)
        day_pre_d = _compute_yields_pre(raw_pre, date, cal, cdates)
        if len(day_pre_d) < 2:
            return None

        _pre_crv = PrefixadoCurve(day_pre_d, cal, method='flatfwd',
                                  yield_col='yield_base', coupon_col='coupon',
                                  coupon_dates=cdates)
        _zc_pre  = _pre_crv.zero_curve(date)
        if _zc_pre is None or len(_zc_pre) == 0:
            return None
        _zc_pre = _zc_pre[
            _zc_pre['zero_rate'].between(0.00, 0.40)
        ].reset_index(drop=True)
        if len(_zc_pre) < 2:
            return None

        # Forward-rate consistency check: iteratively drop the node at the
        # lower-du end of the most negative forward interval until all forwards
        # are non-negative. This removes individual anomalous NTN-F bootstrap
        # nodes rather than discarding the entire NTN-F set.
        # Falls back to LTN-only only when fewer than 2 nodes remain.
        for _ in range(len(_zc_pre)):
            _zp_s   = _zc_pre.sort_values('du').reset_index(drop=True)
            _dp_dus = _zp_s['du'].values.astype(float)
            _dp_dfs = (1.0 + _zp_s['zero_rate'].values) ** (-_dp_dus / 252.0)
            _dp_ddu = np.diff(_dp_dus)
            _dp_fwd = np.where(
                _dp_ddu > 0,
                (_dp_dfs[:-1] / _dp_dfs[1:]) ** (252.0 / _dp_ddu) - 1.0,
                _zp_s['zero_rate'].values[1:],
            )
            if not np.any(_dp_fwd < 0.0):
                _zc_pre = _zp_s
                break
            _worst  = int(np.argmin(_dp_fwd))
            _zc_pre = _zp_s.drop(index=_worst + 1).reset_index(drop=True)
            if len(_zc_pre) < 2:
                break

        if len(_zc_pre) < 2:
            _ltn_rows = (
                day_pre_d[day_pre_d['bond_type'] == 'LTN'][['du', 'yield_base']]
                .rename(columns={'yield_base': 'zero_rate'})
                .sort_values('du')
                .reset_index(drop=True)
            )
            if len(_ltn_rows) >= 2:
                _zc_pre = _ltn_rows
            else:
                return None

        # Post-bootstrap zero-curve outlier filter for prefixado (1.5pp threshold)
        if len(_zc_pre) >= 3:
            _zsp     = _zc_pre.sort_values('du').reset_index(drop=True)
            _drop_zp = []
            for _i in range(1, len(_zsp) - 1):
                _zc_v = _zsp['zero_rate'].iloc[_i]
                _zp_v = _zsp['zero_rate'].iloc[_i - 1]
                _zn_v = _zsp['zero_rate'].iloc[_i + 1]
                if (_zp_v - _zc_v > 0.015) and (_zn_v - _zc_v > 0.015):
                    _drop_zp.append(_i)
                elif (_zc_v - _zp_v > 0.015) and (_zc_v - _zn_v > 0.015):
                    _drop_zp.append(_i)
            if _drop_zp:
                _zc_pre = _zsp.drop(index=_drop_zp).reset_index(drop=True)
        if len(_zc_pre) < 2:
                return None

        # ── Flat-forward interpolation ─────────────────────────────────────────
        _ytm_n    = _flatfwd_ytm(zc_ntnb, du_brkv)
        _ytm_p    = _flatfwd_ytm(_zc_pre, du_brkv)
        breakeven = (1.0 + _ytm_p) / (1.0 + _ytm_n) - 1.0
        breakeven = np.where((breakeven >= -0.02) & (breakeven <= 0.20),
                             breakeven, np.nan)

        return pd.DataFrame({
            'date'     : date,
            'du'       : du_brkv,
            'years'    : du_brkv / 252,
            'r_nominal': _ytm_p   * 100,
            'r_real'   : _ytm_n   * 100,
            'breakeven': breakeven * 100,
        })

    except Exception:
        print(f"  [SKIP] {date.date()}:\n{tb.format_exc()}")
        return None


# ── panel builder ──────────────────────────────────────────────────────────────

def build_breakeven_panel(
    dta: pd.DataFrame,
    vna_series: pd.Series,
    dates: list[pd.Timestamp] | None = None,
    du_brkv: np.ndarray | None = None,
    cdates: np.ndarray | None = None,
    df_dap: pd.DataFrame | None = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build a daily breakeven (implied inflation) panel via flat-forward
    interpolation from bootstrapped zero curves.

    Parameters
    ----------
    dta : cleaned secondary market DataFrame (from ``load_secondary_data()``).
    vna_series : daily VNA Series indexed by Timestamp (from ``build_vna()``).
    dates : list of dates to process (default: all dates in dta).
    du_brkv : tenor grid in business days (default: 252 to 2520 in steps of 126).
    cdates : adjusted NTN-F / NTN-B coupon dates array (default: auto-generated
        from 2000-01-01 to 2040-01-01 at 6-month intervals).
    df_dap : optional DAP settlement DataFrame (DATA, VENCIMENTO, AJUSTE_ATUAL).
        When supplied, DAP zero rates are merged into the NTN-B bootstrapped
        real zero curve as additional anchors — NTN-B nodes take priority at
        overlapping maturities, DAP fills in gaps. Useful from 2018 onwards
        when DAP liquidity is reliable.
        Typically fetched from ``ajuste_pregao WHERE ATIVO LIKE 'DAP%'``.
    n_jobs : number of parallel jobs (joblib). -1 = all cores.
    verbose : print joblib progress.

    Returns
    -------
    pd.DataFrame with columns: date, du, years, r_nominal, r_real, breakeven.
    """
    from joblib import Parallel, delayed

    if du_brkv is None:
        du_brkv = np.arange(126, 2646, 126, dtype=float)

    if cdates is None:
        raw_cpn = pd.date_range('2000-01-01', '2040-01-01', freq='6MS')
        cal     = _CAL
        cdates  = np.array(
            [cal.adjust_next(d) if not cal.isbizday(d) else d for d in raw_cpn],
            dtype='datetime64[D]',
        )

    if dates is None:
        dates = sorted(pd.to_datetime(dta['date'].unique()))

    dta_by_date = {d: grp for d, grp in dta.groupby('date')}
    dates       = [d for d in dates if d in dta_by_date]

    dap_by_date: dict = {}
    if df_dap is not None:
        _dap = df_dap.copy()
        _dap['DATA'] = pd.to_datetime(_dap['DATA'])
        dap_by_date  = {d: grp for d, grp in _dap.groupby('DATA')}

    joblib_verbose = 5 if verbose else 0
    results = Parallel(n_jobs=n_jobs, verbose=joblib_verbose)(
        delayed(_process_date)(
            date, dta_by_date[date], vna_series, du_brkv, cdates,
            dap_by_date.get(date),
        )
        for date in dates
    )

    out_list = [r for r in results if r is not None]
    if not out_list:
        return pd.DataFrame(columns=['date', 'du', 'years', 'r_nominal', 'r_real', 'breakeven'])

    panel = pd.concat(out_list, ignore_index=True)
    if verbose:
        print(f"\nPanel (flat-forward): {len(out_list)} dates, {len(panel)} rows")
    return panel


# ── futures-based breakeven (DI1 + DAP) ───────────────────────────────────────

_B3_MONTH = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
             'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}

# Portuguese month abbreviations used in pre-2006 B3 VENCIMENTO codes
# e.g. 'ABR0' = April 2000, 'NOV1' = November 2001
_PT_MONTH = {
    'JAN': 1, 'FEV': 2, 'MAR': 3, 'ABR': 4, 'MAI': 5, 'JUN': 6,
    'JUL': 7, 'AGO': 8, 'SET': 9, 'OUT': 10, 'NOV': 11, 'DEZ': 12,
}


def _b3_code_to_date(code: str, expiry_day: int = 1) -> pd.Timestamp:
    """
    Convert B3 contract code (e.g. 'K26') to the expiry day of that month.

    Parameters
    ----------
    expiry_day : calendar day of expiration.
        1  → DI1: expires on the 1st business day of the month.
        15 → DAP: expires on the 15th calendar day (same as NTN-B).
    """
    m = _B3_MONTH[code[0].upper()]
    y = 2000 + int(code[1:])
    return pd.Timestamp(y, m, expiry_day)


def parse_ativo(
    code: str,
    itype: str,
    cal: Calendar | None = None,
) -> pd.Timestamp:
    """
    Parse a B3 ATIVO code to its expiry Timestamp.

    Strips the instrument prefix before parsing, so both 6-char codes
    ('DAPK25') and 5-char codes with single-digit year ('DAPK5') work
    correctly. Applies a business-day adjustment to the raw expiry date.

    Parameters
    ----------
    code  : ATIVO ticker, e.g. 'DAPK25', 'DI1N27', or bare suffix 'K25'.
    itype : Instrument prefix, e.g. 'DAP', 'DI1'.
            DAP → expiry on the 15th calendar day of the month.
            All others → expiry on the 1st business day of the month.
    cal   : ANBIMA calendar for business-day adjustment (default: module _CAL).

    Returns
    -------
    pd.Timestamp of the expiry date, adjusted to the next business day if needed.
    """
    if cal is None:
        cal = _CAL
    code       = code.strip()
    suffix     = code[len(itype):].strip() if code.upper().startswith(itype.upper()) else code
    expiry_day = 15 if itype.upper() == 'DAP' else 1
    d = _b3_code_to_date(suffix, expiry_day)
    return pd.Timestamp(cal.adjust_next(d) if not cal.isbizday(d) else d)


def _parse_vencimento(s: pd.Series, expiry_day: int = 1) -> pd.Series:
    """
    Parse VENCIMENTO column to Timestamps.

    Handles three formats:
    - B3 single-letter code  : 'K26', 'F07'  (len ≤ 3)
    - Portuguese abbreviation: 'ABR0', 'NOV1' (3-letter month + digit(s), pre-2006)
    - ISO date string        : '2026-05-15'
    """
    first = str(s.dropna().iloc[0]) if len(s.dropna()) > 0 else ''

    if len(first) <= 3:   # modern B3 code like 'K26'
        return s.apply(
            lambda v: _b3_code_to_date(str(v), expiry_day) if pd.notna(v) else pd.NaT
        )

    if first[:3].upper() in _PT_MONTH:   # pre-2006 Portuguese code like 'ABR0'
        def _pt(v: str) -> pd.Timestamp:
            m = _PT_MONTH.get(v[:3].upper())
            if m is None:
                return pd.NaT
            y = 2000 + int(v[3:])
            return pd.Timestamp(y, m, expiry_day)
        return s.apply(lambda v: _pt(str(v)) if pd.notna(v) else pd.NaT)

    return pd.to_datetime(s, errors='coerce')


def _futures_to_zero_curve(
    grp: pd.DataFrame,
    date: pd.Timestamp,
    cal: Calendar,
    expiry_day: int = 1,
) -> pd.DataFrame:
    """
    Convert a futures settlement DataFrame (VENCIMENTO, AJUSTE_ATUAL) to a
    zero-rate curve DataFrame (du, zero_rate).

    AJUSTE_ATUAL stores settlement PU (~100 000 / (1+r)^(du/252)), not a rate.

    Parameters
    ----------
    expiry_day : calendar day of expiration used when VENCIMENTO is a B3 code.
        1  → DI1 (1st business day of month).
        15 → DAP (15th calendar day = same as NTN-B).
    """
    g = grp[['VENCIMENTO', 'AJUSTE_ATUAL']].copy()
    g['VENCIMENTO'] = _parse_vencimento(g['VENCIMENTO'], expiry_day)
    g = g.dropna(subset=['AJUSTE_ATUAL'])
    g = g[g['AJUSTE_ATUAL'] > 0]

    def _adj(m: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(cal.adjust_next(m)) if not cal.isbizday(m) else m

    g['du'] = g['VENCIMENTO'].apply(lambda m: cal.bizdays(date, _adj(m)))
    g = g[g['du'] > 0]
    g['zero_rate'] = (100_000.0 / g['AJUSTE_ATUAL']) ** (252.0 / g['du']) - 1.0
    g = g[g['zero_rate'].between(-0.05, 0.50)]
    return (
        g[['du', 'zero_rate']]
        .sort_values('du')
        .drop_duplicates(subset=['du'])
        .reset_index(drop=True)
    )


def _process_date_futures(
    date: pd.Timestamp,
    di1_grp: pd.DataFrame,
    dap_grp: pd.DataFrame,
    du_brkv: np.ndarray,
    cal: Calendar,
) -> pd.DataFrame | None:
    """
    Single-date worker: DI1 + DAP settlement rates → breakeven via Fisher.

    Both contracts are zero-coupon — the settlement rate (ajuste) is directly
    the zero rate at that maturity. No bootstrapping or outlier filtering needed.
    """
    zc_nom  = _futures_to_zero_curve(di1_grp, date, cal, expiry_day=1)
    zc_real = _futures_to_zero_curve(dap_grp, date, cal, expiry_day=15)

    if len(zc_nom) < 2 or len(zc_real) < 2:
        return None

    ytm_nom  = _flatfwd_ytm(zc_nom,  du_brkv)
    ytm_real = _flatfwd_ytm(zc_real, du_brkv)

    breakeven = (1.0 + ytm_nom) / (1.0 + ytm_real) - 1.0
    breakeven = np.where((breakeven >= -0.02) & (breakeven <= 0.20), breakeven, np.nan)

    return pd.DataFrame({
        'date'     : date,
        'du'       : du_brkv,
        'years'    : du_brkv / 252,
        'r_nominal': ytm_nom  * 100,
        'r_real'   : ytm_real * 100,
        'breakeven': breakeven * 100,
    })


def build_breakeven_futures(
    df_di1: pd.DataFrame,
    df_dap: pd.DataFrame,
    dates: list[pd.Timestamp] | None = None,
    du_brkv: np.ndarray | None = None,
    cal: Calendar | None = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build a daily breakeven panel from DI1 and DAP B3 settlement rates.

    DI1 and DAP are both zero-coupon contracts — their settlement rate is
    directly the zero rate at that maturity (no bootstrapping needed).
    The breakeven is computed via the Fisher equation:
        breakeven = (1 + r_nominal) / (1 + r_real) - 1

    Parameters
    ----------
    df_di1  : DI1 settlement DataFrame with columns DATA, VENCIMENTO, AJUSTE_ATUAL.
              Typically: ``SELECT * FROM ajuste_pregao WHERE ATIVO LIKE 'DI1%'``
    df_dap  : DAP settlement DataFrame, same column layout.
              Typically: ``SELECT * FROM ajuste_pregao WHERE ATIVO LIKE 'DAP%'``
    dates   : dates to process (default: intersection of dates in both DataFrames).
    du_brkv : tenor grid in business days (default: 126 to 2520 in steps of 126).
    cal     : ANBIMA Calendar (loaded automatically if None).
    n_jobs  : parallel jobs (joblib). -1 = all cores.
    verbose : print joblib progress.

    Returns
    -------
    pd.DataFrame with columns: date, du, years, r_nominal (%), r_real (%), breakeven (%).
    """
    from joblib import Parallel, delayed

    if cal is None:
        cal = _CAL
    if du_brkv is None:
        du_brkv = np.arange(126, 2646, 126, dtype=float)

    df_di1 = df_di1.copy()
    df_dap = df_dap.copy()
    df_di1['DATA'] = pd.to_datetime(df_di1['DATA'])
    df_dap['DATA'] = pd.to_datetime(df_dap['DATA'])

    di1_dates = set(df_di1['DATA'].unique())
    dap_dates = set(df_dap['DATA'].unique())

    if dates is None:
        dates = sorted(di1_dates & dap_dates)
    else:
        dates = [d for d in pd.to_datetime(dates) if d in di1_dates and d in dap_dates]

    di1_by_date = {d: grp for d, grp in df_di1.groupby('DATA')}
    dap_by_date = {d: grp for d, grp in df_dap.groupby('DATA')}

    joblib_verbose = 5 if verbose else 0
    results = Parallel(n_jobs=n_jobs, verbose=joblib_verbose)(
        delayed(_process_date_futures)(
            date, di1_by_date[date], dap_by_date[date], du_brkv, cal
        )
        for date in dates
    )

    out_list = [r for r in results if r is not None]
    if not out_list:
        return pd.DataFrame(columns=['date', 'du', 'years', 'r_nominal', 'r_real', 'breakeven'])

    panel = pd.concat(out_list, ignore_index=True)
    if verbose:
        print(f"\nPanel (futures): {len(out_list)} dates, {len(panel)} rows")
    return panel


# ── NSS panel ─────────────────────────────────────────────────────────────────

_NSS_MIN_R2      = 0.85    # below this, treat fit as failed and reset warm start
_NSS_RATE_LO_PCT = -10.0   # fitted rate floor (real rates hit ~-2% during COVID)
_NSS_RATE_HI_PCT =  60.0   # fitted rate ceiling

_NSS_DU_GRID = np.array([
    126, 252, 378, 504, 630, 756, 882, 1008,
    1134, 1260, 1386, 1512, 1638, 1764, 1890,
    2016, 2142, 2268, 2394, 2520,
], dtype=float)


def _nss_fit_one(
    du: np.ndarray,
    rates_pct: np.ndarray,
    prev_lam: tuple[float, float] | None,
    n_starts_cold: int = 5,
):
    """
    Fit NSS to zero rates (in %). Warm-starts from prev_lam when provided;
    falls back to a full cold-start grid if the warm result is degenerate.

    Returns NSSResult, or None if fitting fails / result is degenerate.
    """
    from br_bonds.nss import fit_nss, NSSResult

    rates = rates_pct / 100.0
    mask  = np.isfinite(rates) & (du > 0)
    du_ok, rates_ok = du[mask], rates[mask]
    if len(du_ok) < 4:
        return None

    def _degenerate(nss: 'NSSResult') -> bool:
        if nss.r2 < _NSS_MIN_R2:
            return True
        fitted_pct = nss.ytm(du_ok) * 100
        return bool(np.any(fitted_pct < _NSS_RATE_LO_PCT) or
                    np.any(fitted_pct > _NSS_RATE_HI_PCT))

    try:
        if prev_lam is not None:
            nss = fit_nss(
                du_ok, rates_ok,
                lam1_grid=np.array([prev_lam[0]]),
                lam2_grid=np.array([prev_lam[1]]),
                n_starts=1,
            )
        else:
            nss = fit_nss(du_ok, rates_ok, n_starts=n_starts_cold)
    except Exception:
        return None

    if _degenerate(nss):
        if prev_lam is None:
            return None
        try:
            nss = fit_nss(du_ok, rates_ok, n_starts=n_starts_cold)
        except Exception:
            return None
        if _degenerate(nss):
            return None

    return nss


def build_nss_panel(
    panel: pd.DataFrame,
    du_grid: np.ndarray | None = None,
    date_from: str | pd.Timestamp | None = None,
    date_to:   str | pd.Timestamp | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit Nelson-Siegel-Svensson curves to a flat-forward breakeven panel,
    using a warm-start chain for speed and parameter smoothness.

    Parameters
    ----------
    panel     : long-format DataFrame with columns date, du, r_nominal, r_real
                (output of build_breakeven_panel or build_breakeven_futures).
                Values in % p.a.
    du_grid   : tenors (business days) at which to record results.
                Default: 126 to 2520 in steps of 126.
    date_from : first date to process (inclusive). Default: all.
    date_to   : last  date to process (inclusive). Default: all.
    verbose   : print progress every 20 dates.

    Returns
    -------
    nss_panel  : DataFrame — date, du, r_nominal, r_real, breakeven (% p.a.)
    nss_params : DataFrame — date, curve, b1..b4, l1, l2, rmse_bps, r2
    """
    if du_grid is None:
        du_grid = _NSS_DU_GRID

    pb = panel.copy()
    pb['date'] = pd.to_datetime(pb['date'])
    if date_from is not None:
        pb = pb[pb['date'] >= pd.Timestamp(date_from)]
    if date_to is not None:
        pb = pb[pb['date'] <= pd.Timestamp(date_to)]

    pivot_nom  = pb.pivot_table(values='r_nominal', columns='du', index='date')
    pivot_real = pb.pivot_table(values='r_real',    columns='du', index='date')
    dates      = pivot_nom.index.intersection(pivot_real.index).sort_values()
    du_cols    = np.array(pivot_nom.columns, dtype=float)

    prev_lam_nom  = None
    prev_lam_real = None
    panel_rows: list[dict] = []
    param_rows: list[dict] = []
    n_ok = n_skip = 0

    for i, ts in enumerate(dates):
        row_nom  = pivot_nom.loc[ts].values.astype(float)
        row_real = pivot_real.loc[ts].values.astype(float)

        nss_nom  = _nss_fit_one(du_cols, row_nom,  prev_lam_nom)
        nss_real = _nss_fit_one(du_cols, row_real, prev_lam_real)

        if nss_nom is None or nss_real is None:
            n_skip += 1
        else:
            prev_lam_nom  = (float(nss_nom.lam[0]),  float(nss_nom.lam[1]))
            prev_lam_real = (float(nss_real.lam[0]), float(nss_real.lam[1]))
            n_ok += 1

            r_nom_g  = nss_nom.ytm(du_grid)  * 100
            r_real_g = nss_real.ytm(du_grid) * 100
            brkv_g   = ((1 + r_nom_g / 100) / (1 + r_real_g / 100) - 1) * 100

            for j, du in enumerate(du_grid):
                panel_rows.append({
                    'date':      ts,
                    'du':        int(du),
                    'r_nominal': round(r_nom_g[j],  6),
                    'r_real':    round(r_real_g[j], 6),
                    'breakeven': round(brkv_g[j],   6),
                })

            for label, nss in [('nominal', nss_nom), ('real', nss_real)]:
                param_rows.append({
                    'date':     ts,
                    'curve':    label,
                    'b1':       round(nss.beta[0] * 100, 6),
                    'b2':       round(nss.beta[1] * 100, 6),
                    'b3':       round(nss.beta[2] * 100, 6),
                    'b4':       round(nss.beta[3] * 100, 6),
                    'l1':       round(nss.lam[0],  6),
                    'l2':       round(nss.lam[1],  6),
                    'rmse_bps': round(nss.rmse * 10_000, 4),
                    'r2':       round(nss.r2, 6),
                })

        if verbose and (i + 1) % 20 == 0:
            print(f'  {i+1:>5}/{len(dates)}  ok={n_ok}  skip={n_skip}', flush=True)

    if verbose:
        print(f'  Finished: {n_ok} dates fitted, {n_skip} skipped.')

    return pd.DataFrame(panel_rows), pd.DataFrame(param_rows)
