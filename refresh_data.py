"""
refresh_data.py  —  Pull fresh data from local sources and save to parquet.

Run this locally whenever you want to update the dashboard data, then
commit and push the updated parquet files to the private GitHub repo.
Streamlit Cloud will redeploy automatically on each push.

Usage:
    python refresh_data.py              # refresh everything
    python refresh_data.py --di1-only  # refresh DI1 only
"""

import argparse
import shutil
from pathlib import Path

import mysql.connector
import pandas as pd

# ── config ────────────────────────────────────────────────────────────────────

DB_CONFIG = dict(host='localhost', user='root', password='foxpro77', database='db_01')
DATA_DIR  = Path(__file__).parent / 'dados_bcb_secundario'

# Paths that are built by other pipelines — just copy them if they live elsewhere
_SECONDARY_SRC = DATA_DIR / 'bcb_mercado_secundario_raw.parquet'   # already in-place
_PANEL_FF_SRC  = DATA_DIR / 'breakeven_panel_bfuts.parquet'        # already in-place
_PANEL_FUT_SRC = DATA_DIR / 'breakeven_panel_futures.parquet'      # already in-place

# COPOM calendar (used by get_focus.py — copy into data dir for cloud)
_COPOM_SRC = Path(r'C:\Users\Ivanildo\py_finance\copom_meetings.xlsx')
_COPOM_DST = DATA_DIR / 'copom_meetings.xlsx'


# ── helpers ───────────────────────────────────────────────────────────────────

def _size(path: Path) -> str:
    mb = path.stat().st_size / 1_048_576
    return f'{mb:.1f} MB'


# ── refresh tasks ─────────────────────────────────────────────────────────────

def refresh_di1() -> None:
    """Export full DI1 futures table from MySQL to parquet."""
    print('Pulling DI1 from MySQL...', flush=True)
    conn = mysql.connector.connect(**DB_CONFIG)
    df   = pd.read_sql(
        "SELECT DATA, ATIVO, VENCIMENTO, AJUSTE_ATUAL "
        "FROM db_01.ajuste_pregao WHERE ATIVO LIKE 'DI1%'",
        conn,
    )
    conn.close()

    df['DATA'] = pd.to_datetime(df['DATA'])
    df = df.sort_values('DATA').reset_index(drop=True)

    out = DATA_DIR / 'di1_futs.parquet'
    df.to_parquet(out, index=False)
    print(f'  Saved {len(df):,} rows → {out}  ({_size(out)})')
    print(f'  Date range: {df["DATA"].min().date()} → {df["DATA"].max().date()}')


def refresh_cdi() -> None:
    """Pull CDI overnight series from BCB SGS (direct HTTP, chunked) and save to parquet."""
    import requests

    print('Pulling CDI from BCB API (chunked)...', flush=True)

    def _fetch_chunk(start: str, end: str) -> list:
        url = (f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados'
               f'?formato=json&dataInicial={start}&dataFinal={end}')
        r = requests.get(url, timeout=30).json()
        if not isinstance(r, list):
            raise RuntimeError(f'BCB CDI fetch failed for {start}-{end}: {r}')
        return r

    try:
        today_str = pd.Timestamp.today().strftime('%d/%m/%Y')
        chunks = [
            _fetch_chunk('01/01/2000', '31/12/2009'),
            _fetch_chunk('01/01/2010', '31/12/2019'),
            _fetch_chunk('01/01/2020', today_str),
        ]
        rows = [row for chunk in chunks for row in chunk]
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['data'], dayfirst=True)
        df['cdi_daily'] = pd.to_numeric(df['valor'].str.replace(',', '.'), errors='coerce')
        df['cdi_annual'] = ((1 + df['cdi_daily'] / 100) ** 252 - 1) * 100
        df = df[['date', 'cdi_annual']].dropna().drop_duplicates('date').sort_values('date')
        out = DATA_DIR / 'cdi_series.parquet'
        df.to_parquet(out, index=False)
        print(f'  Saved {len(df):,} rows  ({_size(out)})')
        print(f'  Date range: {df["date"].min().date()} to {df["date"].max().date()}')
        print(f'  Latest: {df["date"].iloc[-1].date()} = {df["cdi_annual"].iloc[-1]:.4f}% p.a.')
    except Exception as e:
        print(f'  ERROR: {e}')


def copy_copom_calendar() -> None:
    """Copy copom_meetings.xlsx into the data folder."""
    if not _COPOM_SRC.exists():
        print(f'  SKIP: copom_meetings.xlsx not found at {_COPOM_SRC}')
        return
    shutil.copy2(_COPOM_SRC, _COPOM_DST)
    print(f'  Copied copom_meetings.xlsx → {_COPOM_DST}  ({_size(_COPOM_DST)})')


def check_panels() -> None:
    """Report sizes of the breakeven panel parquets."""
    for p in (_PANEL_FF_SRC, _PANEL_FUT_SRC, _SECONDARY_SRC):
        if p.exists():
            df = pd.read_parquet(p)
            dates = pd.to_datetime(df['date']) if 'date' in df.columns else None
            rng   = f'{dates.min().date()} → {dates.max().date()}' if dates is not None else ''
            print(f'  {p.name:<45} {len(df):>8,} rows  {_size(p)}  {rng}')
        else:
            print(f'  MISSING: {p.name}')


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--di1-only', action='store_true')
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    print('\n=== refresh_data.py ===\n')

    print('[1/4] DI1 futures')
    refresh_di1()

    if not args.di1_only:
        print('\n[2/4] CDI overnight series')
        refresh_cdi()

        print('\n[3/4] COPOM calendar')
        copy_copom_calendar()

        print('\n[4/4] Panel parquets (status check)')
        check_panels()

    print('\nDone.')
    print('Next step: git add dados_bcb_secundario/ && git commit -m "data: refresh" && git push')


if __name__ == '__main__':
    main()
