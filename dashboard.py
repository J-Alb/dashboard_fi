"""
dashboard.py  —  Brazilian Rates Dashboard (Prefixado angle)
─────────────────────────────────────────────────────────────
Run:
    streamlit run dashboard.py
"""

import sys
import warnings
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from bizdays import Calendar

warnings.filterwarnings('ignore')

_HERE = Path(__file__).parent
_DATA = _HERE / 'dados_bcb_secundario'
sys.path.insert(0, str(_HERE))  # finds get_focus.py and br_bonds/ in same dir

from br_bonds.secondary import (
    load_secondary_data,
    get_pre_curve,
    _compute_yields_pre,
    _flatfwd_ytm,
)
from br_bonds import PrefixadoCurve
from br_bonds.derivatives.di1 import DI1Curve
from br_bonds.derivatives.copom import CopomCurve, COPOM_DATES

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='Brazilian Rates',
    page_icon='📈',
    layout='wide',
)

st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# ── constants ─────────────────────────────────────────────────────────────────

CAL = Calendar.load('ANBIMA')

C_PRE_T   = '#1a3a5c'   # dark navy   — prefixado T
C_PRE_1M  = '#5b9bd5'   # steel blue  — prefixado T-1m
C_DI1_T   = '#7b1c00'   # dark red    — DI1 T
C_DI1_1M  = '#e07040'   # terracotta  — DI1 T-1m
C_NTNB_T  = '#1a5c2a'   # dark green  — NTN-B T
C_NTNB_1M = '#5cb87a'   # light green — NTN-B T-1m
C_DAP_T   = '#5c3a7b'   # dark purple — DAP T
C_DAP_1M  = '#a07abe'   # light purple— DAP T-1m

_FONT = dict(family='Montserrat, sans-serif', size=12)

# ── cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(show_spinner='Loading secondary market data…')
def _load_secondary():
    return load_secondary_data(_DATA / 'bcb_mercado_secundario_raw.parquet')


@st.cache_data(show_spinner='Loading DI1…')
def _load_di1_sql():
    path = _DATA / 'di1_futs.parquet'
    df   = pd.read_parquet(path)
    df['DATA'] = pd.to_datetime(df['DATA'])
    return df


@st.cache_data(show_spinner='Loading panels…')
def _load_panels():
    pb = pd.read_parquet(_DATA / 'breakeven_panel_bfuts.parquet')
    #pb = pd.read_parquet(_DATA / 'breakeven_panel_ff.parquet')
    pf = pd.read_parquet(_DATA / 'breakeven_panel_futures.parquet')
    pb['date'] = pd.to_datetime(pb['date'])
    pf['date'] = pd.to_datetime(pf['date'])

    pivot_pre_nom  = pb.pivot_table(values='r_nominal',  columns='du', index='date')
    pivot_pre_real = pb.pivot_table(values='r_real',     columns='du', index='date')
    pivot_pre_brkv = pb.pivot_table(values='breakeven',  columns='du', index='date')

    pivot_fut_nom  = pf.pivot_table(values='r_nominal',  columns='du', index='date')
    pivot_fut_real = pf.pivot_table(values='r_real',     columns='du', index='date')
    pivot_fut_brkv = pf.pivot_table(values='breakeven',  columns='du', index='date')

    # DAP liquidity filter
    valid = pivot_fut_nom.index >= '2016-01-01'
    dense = pivot_fut_nom.notna().sum(axis=1) >= 10
    for p in (pivot_fut_nom, pivot_fut_real, pivot_fut_brkv):
        p.drop(p.index[~(valid & dense)], inplace=True)

    return (pivot_pre_nom, pivot_pre_real, pivot_pre_brkv,
            pivot_fut_nom, pivot_fut_real, pivot_fut_brkv)


def _fwd_brkv(pivot_nom: pd.DataFrame, pivot_real: pd.DataFrame,
              du1: float, du2: float) -> pd.Series:
    """
    Forward breakeven inflation from du1 to du2 (business days).

    Uses spot zero rates from the panel pivots and the Fisher equation:
        fwd_brkv = (1 + r_nom_fwd) / (1 + r_real_fwd) − 1

    where each forward rate is implied by discount-factor arithmetic:
        df(r, du) = (1 + r/100)^(−du/252)
        r_fwd     = (df(du1)/df(du2))^(252/(du2−du1)) − 1

    Standard tenors:  1y1y → (252,504)  |  2y1y → (504,756)  |  5y1y → (1260,1512)
    """
    span = du2 - du1

    def _df(pivot, du):
        r = pivot[du] / 100
        return (1 + r) ** (-du / 252)

    df_nom_1  = _df(pivot_nom,  du1);  df_nom_2  = _df(pivot_nom,  du2)
    df_real_1 = _df(pivot_real, du1);  df_real_2 = _df(pivot_real, du2)

    r_nom_fwd  = (df_nom_1  / df_nom_2)  ** (252 / span) - 1
    r_real_fwd = (df_real_1 / df_real_2) ** (252 / span) - 1

    return ((1 + r_nom_fwd) / (1 + r_real_fwd) - 1) * 100


@st.cache_data(show_spinner='Fetching CDI overnight…')
def _get_cdi(ref_date: date) -> float:
    try:
        from bcb import sgs
        start  = pd.Timestamp(ref_date) - pd.Timedelta(days=10)
        df_cdi = sgs.get({'cdi': 12}, start=start.date(), end=ref_date).dropna()
        if not df_cdi.empty and pd.Timestamp(df_cdi.index[-1]) >= pd.Timestamp(ref_date) - pd.Timedelta(days=1):
            daily = float(df_cdi['cdi'].iloc[-1])
            return ((1 + daily / 100) ** 252 - 1) * 100
        df_sel = sgs.get({'selic': 432}, start=start.date(), end=ref_date).dropna()
        return float(df_sel['selic'].iloc[-1]) - 0.10
    except Exception:
        return 13.25


@st.cache_data(show_spinner='Building curves…')
def _build_curves(ref_date: date, cdi_pct: float):
    """Returns (zc_pre DataFrame | None, DI1Curve)."""
    _B3M = {'F':1,'G':2,'H':3,'J':4,'K':5,'M':6,
            'N':7,'Q':8,'U':9,'V':10,'X':11,'Z':12}

    def code2date(v):
        m   = _B3M[str(v)[0].upper()]
        y   = 2000 + int(str(v)[1:])
        raw = pd.Timestamp(y, m, 1)
        return pd.Timestamp(CAL.adjust_next(raw)) if not CAL.isbizday(raw) else raw

    # ── prefixado ────────────────────────────────────────────────────────────
    raw_cpn = pd.date_range('2000-01-01', '2040-01-01', freq='6MS')
    cdates  = np.array(
        [CAL.adjust_next(d) if not CAL.isbizday(d) else d for d in raw_cpn],
        dtype='datetime64[D]',
    )
    dta = _load_secondary()
    ts  = pd.Timestamp(ref_date)
    raw = get_pre_curve(dta, ts)
    day = _compute_yields_pre(raw, ts, CAL, cdates)
    zc_pre = None
    if len(day) >= 2:
        crv    = PrefixadoCurve(day, CAL, method='flatfwd',
                                yield_col='yield_base', coupon_col='coupon',
                                coupon_dates=cdates)
        zc_pre = crv.zero_curve(ts)

    # ── DI1 ──────────────────────────────────────────────────────────────────
    df_all = _load_di1_sql()
    grp    = df_all[df_all['DATA'] == ts].copy()
    if grp.empty:
        available = df_all['DATA'].sort_values().unique()
        fallback  = available[available <= ts]
        grp = df_all[df_all['DATA'] == pd.Timestamp(fallback[-1])].copy()

    grp = grp.dropna(subset=['AJUSTE_ATUAL'])
    grp = grp[grp['AJUSTE_ATUAL'] > 0]

    first = str(grp['VENCIMENTO'].dropna().iloc[0])
    if len(first) <= 3:
        grp['maturity_date'] = grp['VENCIMENTO'].apply(code2date)
    else:
        grp['maturity_date'] = pd.to_datetime(grp['VENCIMENTO'])

    grp['du']   = grp['maturity_date'].apply(lambda m: CAL.bizdays(ts, m))
    grp         = grp[grp['du'] > 1].copy()
    grp['rate'] = ((100_000.0 / grp['AJUSTE_ATUAL']) ** (252.0 / grp['du']) - 1.0) * 100.0
    grp         = grp[grp['rate'].between(0, 50)].sort_values('du').reset_index(drop=True)

    di1_crv = DI1Curve.from_data(grp[['maturity_date', 'rate']], cdi_pct, ts, CAL)
    return zc_pre, di1_crv


@st.cache_data(show_spinner='Fitting COPOM…')
def _fit_copom(ref_date: date, cdi_pct: float):
    _, di1_crv = _build_curves(ref_date, cdi_pct)
    copom      = CopomCurve(di1_crv, COPOM_DATES, CAL).fit()
    return copom.decisions


@st.cache_data(show_spinner='Fetching Focus Selic…', ttl=3600)
def _load_focus_selic():
    """Load Focus meeting-by-meeting Selic expectations + COPOM calendar."""
    try:
        from get_focus import fetch_selic_meetings, load_copom_calendar
        df      = fetch_selic_meetings(start='2022-01-01')
        # prefer bundled copy; fall back to default (local Windows path)
        _copom_path = str(_DATA / 'copom_meetings.xlsx')
        cal_map = load_copom_calendar(path=_copom_path)
        return df, cal_map
    except Exception:
        return None, None


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title('🇧🇷 Brazilian Rates')
    st.markdown('---')

    # available dates
    _di1_all    = _load_di1_sql()
    _avail_raw  = sorted(_di1_all['DATA'].unique(), reverse=True)
    _avail_dates = [pd.Timestamp(d).date() for d in _avail_raw]

    sel_date = st.selectbox(
        'Reference date',
        options=_avail_dates,
        index=0,
        format_func=lambda d: d.strftime('%d/%m/%Y'),
    )

    show_comp = st.toggle('Show T − 1 month', value=True)

    comp_date = None
    if show_comp:
        _t1m_raw  = pd.Timestamp(sel_date) - pd.DateOffset(months=1)
        _t1m_snap = pd.Timestamp(
            CAL.adjust_previous(_t1m_raw) if not CAL.isbizday(_t1m_raw) else _t1m_raw
        ).date()
        comp_date = st.selectbox(
            'Comparison date',
            options=_avail_dates,
            index=_avail_dates.index(_t1m_snap) if _t1m_snap in _avail_dates else 0,
            format_func=lambda d: d.strftime('%d/%m/%Y'),
        )

    st.markdown('---')

    all_du     = [126, 252, 378, 504, 630, 756, 1008, 1260, 1512, 2016]
    tenor_lbls = {126:'6M', 252:'1Y', 378:'18M', 504:'2Y', 630:'30M',
                  756:'3Y', 1008:'4Y', 1260:'5Y', 1512:'6Y', 2016:'8Y'}
    sel_tenors = st.multiselect(
        'Tenors (time series)',
        options=all_du,
        default=[252, 504, 1008],
        format_func=lambda x: tenor_lbls.get(x, f'du={x}'),
    )

    st.markdown('---')
    ts_start = st.date_input(
        'Time series start',
        value=date(2016, 1, 1),
        min_value=date(2003, 1, 1),
    )


# ── fetch data for selected dates ────────────────────────────────────────────

cdi_t  = _get_cdi(sel_date)
zc_t, di1_t = _build_curves(sel_date, cdi_t)

zc_1m = di1_1m = None
if show_comp and comp_date:
    cdi_1m      = _get_cdi(comp_date)
    zc_1m, di1_1m = _build_curves(comp_date, cdi_1m)

dec_t  = _fit_copom(sel_date, cdi_t)
dec_1m = _fit_copom(comp_date, cdi_1m) if show_comp and comp_date else None


# ── helpers ───────────────────────────────────────────────────────────────────

du_grid = np.arange(21, 2521, 21, dtype=float)

def _pre_xy(zc):
    lo, hi = float(zc['du'].min()), float(zc['du'].max())
    d = du_grid[(du_grid >= lo) & (du_grid <= hi)]
    r = _flatfwd_ytm(zc, d) * 100
    return d / 252, r

def _di1_xy(crv):
    d = du_grid[(du_grid >= crv.du_min) & (du_grid <= crv.du_max)]
    r = np.array([crv.ytm(int(x)) * 100 for x in d])
    return d / 252, r


# ── tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(['Curvas', 'COPOM', 'Série Histórica', 'Breakeven'])


# ════════════════════════════════════════════════════════════════════════════
# Tab 1 — Spot Curves
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader('Estrutura a Termo: Prefixada (LTN/NTN-F) vs DI1')

    fig1 = go.Figure()

    lbl_t  = pd.Timestamp(sel_date).strftime('%d/%m/%Y')
    lbl_1m = pd.Timestamp(comp_date).strftime('%d/%m/%Y') if comp_date else ''

    if zc_t is not None:
        x, y = _pre_xy(zc_t)
        fig1.add_trace(go.Scatter(
            x=x, y=y, name=f'Prefixado {lbl_t}',
            line=dict(color=C_PRE_T, width=2.5),
            hovertemplate='%{y:.2f}%<extra>Prefixado ' + lbl_t + '</extra>',
        ))

    x, y = _di1_xy(di1_t)
    fig1.add_trace(go.Scatter(
        x=x, y=y, name=f'DI1 {lbl_t}',
        line=dict(color=C_DI1_T, width=2.5),
        hovertemplate='%{y:.2f}%<extra>DI1 ' + lbl_t + '</extra>',
    ))

    if show_comp and zc_1m is not None:
        x, y = _pre_xy(zc_1m)
        fig1.add_trace(go.Scatter(
            x=x, y=y, name=f'Prefixado {lbl_1m}',
            line=dict(color=C_PRE_1M, width=1.8, dash='dash'),
            hovertemplate='%{y:.2f}%<extra>Prefixado ' + lbl_1m + '</extra>',
        ))

    if show_comp and di1_1m is not None:
        x, y = _di1_xy(di1_1m)
        fig1.add_trace(go.Scatter(
            x=x, y=y, name=f'DI1 {lbl_1m}',
            line=dict(color=C_DI1_1M, width=1.8, dash='dash'),
            hovertemplate='%{y:.2f}%<extra>DI1 ' + lbl_1m + '</extra>',
        ))

    fig1.update_layout(
        xaxis_title='Prazo (anos)',
        yaxis_title='Taxa (% a.a.)',
        yaxis_tickformat='.1f',
        yaxis_ticksuffix='%',
        hovermode='x unified',
        height=480,
        font=_FONT,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(t=60),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    for du_show, lbl in [(252, '1Y'), (504, '2Y'), (1008, '4Y'), (1260, '5Y')]:
        val_pre = _flatfwd_ytm(zc_t, np.array([float(du_show)]))[0] * 100 if zc_t is not None else None
        val_di1 = di1_t.ytm(du_show) * 100 if di1_t.ytm(du_show) else None
        spread  = (val_pre - val_di1) if val_pre and val_di1 else None

    for col, (du_show, lbl) in zip([col1, col2, col3, col4],
                                    [(252,'1Y'),(504,'2Y'),(1008,'4Y'),(1260,'5Y')]):
        val_pre = _flatfwd_ytm(zc_t, np.array([float(du_show)]))[0] * 100 if zc_t is not None else None
        ytm_di1 = di1_t.ytm(du_show)
        val_di1 = ytm_di1 * 100 if ytm_di1 is not None else None
        spread  = round(val_pre - val_di1, 2) if val_pre is not None and val_di1 is not None else None
        with col:
            st.metric(
                label=f'Prefixado {lbl} vs DI1',
                value=f'{val_pre:.2f}%' if val_pre else 'n/a',
                delta=f'{spread:+.2f}pp spread' if spread is not None else None,
            )


# ════════════════════════════════════════════════════════════════════════════
# Tab 2 — COPOM
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader('Decisões COPOM Implícitas na Curva DI1')

    fig2 = go.Figure()

    for dec, crv, color, dash, lbl, rdate in [
        (dec_t,  di1_t,  C_DI1_T,  'solid', lbl_t,  sel_date),
        (dec_1m, di1_1m, C_DI1_1M, 'dash',  lbl_1m, comp_date),
    ]:
        if dec is None:
            continue
        overnight_pct = crv.overnight * 100
        dates_step    = [pd.Timestamp(rdate)] + list(dec['meeting_date'])
        rates_step    = [overnight_pct]       + list(dec['cdi_pct'])

        # step line
        fig2.add_trace(go.Scatter(
            x=dates_step, y=rates_step,
            name=f'CDI implícita — {lbl}',
            mode='lines',
            line=dict(color=color, width=2.5, dash=dash, shape='hv'),
            hovertemplate='%{y:.2f}%<br>%{x|%d/%m/%Y}<extra>' + lbl + '</extra>',
        ))
        # meeting dots
        fig2.add_trace(go.Scatter(
            x=dec['meeting_date'], y=dec['cdi_pct'],
            mode='markers',
            marker=dict(color=color, size=7),
            showlegend=False,
            hovertemplate='%{y:.2f}%<br>%{x|%d/%m/%Y}<extra></extra>',
        ))

    # ── Focus Selic overlay ───────────────────────────────────────────────────
    _focus_df, _copom_cal = _load_focus_selic()
    if _focus_df is not None and _copom_cal:
        _ts_ref = pd.Timestamp(sel_date)
        _latest = _focus_df[_focus_df['Data'] <= _ts_ref]['Data'].max()
        if pd.notna(_latest):
            _snap = (
                _focus_df[_focus_df['Data'] == _latest]
                .groupby('Reuniao')['Mediana']
                .last()
                .reset_index()
            )
            _snap['meeting_date'] = _snap['Reuniao'].map(_copom_cal)
            _snap = _snap.dropna(subset=['meeting_date', 'Mediana']).sort_values('meeting_date')
            fig2.add_trace(go.Scatter(
                x=_snap['meeting_date'], y=_snap['Mediana'],
                mode='markers',
                name=f'Focus Selic ({_latest.strftime("%d/%m/%Y")})',
                marker=dict(color='#f5a623', size=9, symbol='diamond',
                            line=dict(color='#c47a00', width=1)),
                hovertemplate='%{y:.2f}%<br>%{x|%d/%m/%Y}<extra>Focus</extra>',
            ))

    fig2.update_layout(
        xaxis_title='Reunião COPOM',
        yaxis_title='CDI implícita (% a.a.)',
        yaxis_tickformat='.2f',
        yaxis_ticksuffix='%',
        hovermode='x unified',
        height=460,
        font=_FONT,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(t=60),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # COPOM table
    with st.expander('Tabela de decisões implícitas'):
        tbl = dec_t[['meeting_date', 'bps', 'bps_cum', 'cdi_pct', 'selic_pct']].copy()
        tbl.columns = ['Reunião', 'Decisão (bps)', 'Acumulado (bps)', 'CDI impl. (%)', 'Selic impl. (%)']
        tbl['Reunião'] = tbl['Reunião'].dt.strftime('%d/%m/%Y')
        for c in ['Decisão (bps)', 'Acumulado (bps)']:
            tbl[c] = tbl[c].round(1)
        for c in ['CDI impl. (%)', 'Selic impl. (%)']:
            tbl[c] = tbl[c].round(2)
        st.dataframe(tbl, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# Tab 3 — Time Series
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader('Prefixado vs DI1 — Taxas a Termo Constante')

    if not sel_tenors:
        st.info('Select at least one tenor in the sidebar.')
    elif not (_DATA / 'breakeven_panel_ff.parquet').exists():
        st.warning('Panel parquets not found — run implicita_v3.py first.')
    else:
        (pivot_pre, _, _, pivot_di1, _, _) = _load_panels()

        pivot_pre = pivot_pre.loc[pivot_pre.index >= pd.Timestamp(ts_start)]
        pivot_di1 = pivot_di1.loc[pivot_di1.index >= pd.Timestamp(ts_start)]

        def _nearest(pivot, du_target):
            cols = np.array(pivot.columns, dtype=float)
            return float(cols[np.argmin(np.abs(cols - du_target))])

        fig3 = make_subplots(
            rows=len(sel_tenors), cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=[f'{tenor_lbls.get(d, d)}  (du={d})' for d in sel_tenors],
        )

        for i, du_req in enumerate(sel_tenors, start=1):
            col_pre = _nearest(pivot_pre, du_req)
            col_di1 = _nearest(pivot_di1, du_req) if du_req in pivot_di1.columns or True else None

            s_pre = pivot_pre[col_pre].dropna()
            fig3.add_trace(go.Scatter(
                x=s_pre.index, y=s_pre.values,
                name=f'Prefixado {tenor_lbls.get(du_req, du_req)}',
                line=dict(color=C_PRE_T, width=1.5),
                hovertemplate='%{y:.2f}%<extra>Prefixado</extra>',
                legendgroup=f'pre_{du_req}',
                showlegend=(i == 1),
            ), row=i, col=1)

            if col_di1 is not None and col_di1 in pivot_di1.columns:
                s_di1 = pivot_di1[col_di1].dropna()
                fig3.add_trace(go.Scatter(
                    x=s_di1.index, y=s_di1.values,
                    name=f'DI1 {tenor_lbls.get(du_req, du_req)}',
                    line=dict(color=C_DI1_T, width=1.5, dash='dash'),
                    hovertemplate='%{y:.2f}%<extra>DI1</extra>',
                    legendgroup=f'di1_{du_req}',
                    showlegend=(i == 1),
                ), row=i, col=1)

            # vertical line at reference date
            fig3.add_vline(
                x=pd.Timestamp(sel_date).timestamp() * 1000,
                line=dict(color='gray', width=1, dash='dot'),
                row=i, col=1,
            )

            fig3.update_yaxes(ticksuffix='%', tickformat='.1f', row=i, col=1)

        fig3.update_layout(
            height=320 * len(sel_tenors),
            hovermode='x unified',
            font=_FONT,
            legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
            margin=dict(t=80),
        )
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# Tab 4 — Breakeven
# ════════════════════════════════════════════════════════════════════════════

with tab4:
    if not (_DATA / 'breakeven_panel_ff.parquet').exists():
        st.warning('Panel parquets not found — run implicita_v3.py first.')
    else:
        (pivot_pre_nom, pivot_pre_real, pivot_pre_brkv,
         pivot_fut_nom, pivot_fut_real, pivot_fut_brkv) = _load_panels()

        pivot_pre_nom  = pivot_pre_nom.loc[pivot_pre_nom.index   >= pd.Timestamp(ts_start)]
        pivot_pre_real = pivot_pre_real.loc[pivot_pre_real.index >= pd.Timestamp(ts_start)]
        pivot_pre_brkv = pivot_pre_brkv.loc[pivot_pre_brkv.index >= pd.Timestamp(ts_start)]
        pivot_fut_nom  = pivot_fut_nom.loc[pivot_fut_nom.index   >= pd.Timestamp(ts_start)]
        pivot_fut_real = pivot_fut_real.loc[pivot_fut_real.index >= pd.Timestamp(ts_start)]
        pivot_fut_brkv = pivot_fut_brkv.loc[pivot_fut_brkv.index >= pd.Timestamp(ts_start)]

        # ── Section 1: spot breakeven curves (snapshot) ──────────────────────
        st.subheader('Curva de Breakeven: Bonds (NTN-B/Pre) vs Futuros (DAP/DI1)')

        fig_brkv = go.Figure()

        # get spot breakeven from panel for the selected date
        def _spot_brkv_curve(pivot_brkv, date):
            ts = pd.Timestamp(date)
            if ts not in pivot_brkv.index:
                idx = pivot_brkv.index.get_indexer([ts], method='nearest')
                ts  = pivot_brkv.index[idx[0]]
            row = pivot_brkv.loc[ts].dropna()
            return row.index.values / 252, row.values  # (years, breakeven%)

        for pivot_brkv, color, dash, lbl in [
            (pivot_pre_brkv, C_PRE_T,  'solid', f'Bonds (NTN-B/Pre)  {lbl_t}'),
            (pivot_fut_brkv, C_DI1_T,  'solid', f'Futuros (DAP/DI1)  {lbl_t}'),
        ]:
            x, y = _spot_brkv_curve(pivot_brkv, sel_date)
            fig_brkv.add_trace(go.Scatter(
                x=x, y=y, name=lbl,
                line=dict(color=color, width=2.5, dash=dash),
                hovertemplate='%{y:.2f}%<extra>' + lbl + '</extra>',
            ))

        if show_comp and comp_date:
            for pivot_brkv, color, lbl in [
                (pivot_pre_brkv, C_PRE_1M, f'Bonds  {lbl_1m}'),
                (pivot_fut_brkv, C_DI1_1M, f'Futuros  {lbl_1m}'),
            ]:
                x, y = _spot_brkv_curve(pivot_brkv, comp_date)
                fig_brkv.add_trace(go.Scatter(
                    x=x, y=y, name=lbl,
                    line=dict(color=color, width=1.8, dash='dash'),
                    hovertemplate='%{y:.2f}%<extra>' + lbl + '</extra>',
                ))

        fig_brkv.update_layout(
            xaxis_title='Prazo (anos)',
            yaxis_title='Breakeven (% a.a.)',
            yaxis_ticksuffix='%', yaxis_tickformat='.1f',
            hovermode='x unified', height=420, font=_FONT,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            margin=dict(t=60),
        )
        st.plotly_chart(fig_brkv, use_container_width=True)

        st.markdown('---')

        # ── Section 2: forward breakeven time series ──────────────────────────
        st.subheader('Breakeven Forward: 1y1y · 2y1y · 5y1y')

        FWD_TENORS = [(252, 504, '1y1y'), (504, 756, '2y1y'), (1260, 1512, '5y1y')]
        FWD_COLORS = ['#2e86ab', '#e84855', '#3bb273']

        # check all required du columns exist
        def _has_cols(pivot, *dus):
            return all(float(d) in pivot.columns for d in dus)

        fig_fwd = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=[f'{lbl}' for _, _, lbl in FWD_TENORS],
        )

        for row, ((du1, du2, lbl), color) in enumerate(zip(FWD_TENORS, FWD_COLORS), start=1):
            for pivot_nom, pivot_real, src_lbl, dash in [
                (pivot_pre_nom, pivot_pre_real, 'Bonds',   'solid'),
                (pivot_fut_nom, pivot_fut_real, 'Futuros', 'dash'),
            ]:
                if not _has_cols(pivot_nom, du1, du2) or not _has_cols(pivot_real, du1, du2):
                    continue

                fwd = _fwd_brkv(pivot_nom, pivot_real, float(du1), float(du2)).dropna()

                fig_fwd.add_trace(go.Scatter(
                    x=fwd.index, y=fwd.values,
                    name=f'{src_lbl} {lbl}',
                    line=dict(color=color, width=1.5, dash=dash),
                    hovertemplate='%{y:.2f}%<extra>' + f'{src_lbl} {lbl}' + '</extra>',
                    legendgroup=src_lbl,
                    showlegend=(row == 1),
                ), row=row, col=1)

            # vertical marker at reference date
            fig_fwd.add_vline(
                x=pd.Timestamp(sel_date).timestamp() * 1000,
                line=dict(color='gray', width=1, dash='dot'),
                row=row, col=1,
            )
            fig_fwd.update_yaxes(ticksuffix='%', tickformat='.1f', row=row, col=1)

        fig_fwd.update_layout(
            height=900,
            hovermode='x unified',
            font=_FONT,
            legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
            margin=dict(t=80),
        )
        st.plotly_chart(fig_fwd, use_container_width=True)

        # ── Metric cards: current forward breakeven ───────────────────────────
        st.markdown('---')
        st.markdown(f'**Breakeven forward implícito em {lbl_t}**')
        cols = st.columns(3)
        for col, (du1, du2, lbl) in zip(cols, FWD_TENORS):
            val_bonds = val_fut = None
            if _has_cols(pivot_pre_nom, du1, du2) and _has_cols(pivot_pre_real, du1, du2):
                ts = pd.Timestamp(sel_date)
                if ts in pivot_pre_nom.index:
                    s = _fwd_brkv(pivot_pre_nom, pivot_pre_real, float(du1), float(du2))
                    val_bonds = float(s.loc[ts]) if ts in s.index else None
            if _has_cols(pivot_fut_nom, du1, du2) and _has_cols(pivot_fut_real, du1, du2):
                ts = pd.Timestamp(sel_date)
                if ts in pivot_fut_nom.index:
                    s = _fwd_brkv(pivot_fut_nom, pivot_fut_real, float(du1), float(du2))
                    val_fut = float(s.loc[ts]) if ts in s.index else None
            with col:
                st.metric(f'Bonds {lbl}',   f'{val_bonds:.2f}%' if val_bonds else 'n/a')
                st.metric(f'Futuros {lbl}', f'{val_fut:.2f}%'   if val_fut   else 'n/a')
