"""
update_all.py — Daily pipeline runner

Steps:
  1. update_pregao.py        B3 BDI PDF → MySQL ajuste_pregao    (Schonfeld)
  2. BCB secondary market    current month refresh → parquet      (inline)
  3. VNA rebuild             SIDRA IPCA → vna_ntnb.parquet        (build_vna.py)
  4. compute_breakeven.py    MySQL + secondary + VNA → parquets   (fixed income)
  5. refresh_data.py         MySQL DI1 → di1_futs.parquet         (fixed income)
  6. git add + commit + push

Usage:
    python update_all.py               # full daily run
    python update_all.py --no-push     # skip git push (steps 1-5 only)
    python update_all.py --from-step 2 # start from step 2
    python update_all.py --date 2026-03-14  # process a specific date
"""

import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────

PYTHON    = sys.executable
HERE      = Path(__file__).parent
SCHONFELD = Path(r'C:\Users\Ivanildo\py_finance\Schonfeld')
DATA      = HERE / 'dados_bcb_secundario'

STEP1_SCRIPT = SCHONFELD / 'update_pregao.py'
STEP3_SCRIPT = HERE      / 'build_vna.py'
STEP4_SCRIPT = HERE      / 'compute_breakeven.py'
STEP5_SCRIPT = HERE      / 'refresh_data.py'
STEP6_SCRIPT = HERE      / 'build_nss_panel.py'

# ── helpers ───────────────────────────────────────────────────────────────────

def _header(n: int, label: str) -> None:
    print(f'\n{"="*60}')
    print(f'  Step {n}: {label}')
    print(f'{"="*60}')


def _run(script: Path) -> int:
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'   # suppress plt.show() popups
    result = subprocess.run([PYTHON, str(script)], cwd=str(script.parent), env=env)
    return result.returncode


def _run_with_args(script: Path, args: list[str]) -> int:
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    result = subprocess.run([PYTHON, str(script)] + args,
                            cwd=str(script.parent), env=env)
    return result.returncode


def _git(args: list[str]) -> int:
    return subprocess.run(['git'] + args, cwd=str(HERE)).returncode


# ── steps ─────────────────────────────────────────────────────────────────────

def step1_update_pregao(target_date: str | None = None) -> bool:
    _header(1, 'B3 BDI PDF → MySQL  (update_pregao.py)')
    flag = ['--date', target_date] if target_date else ['--today']
    rc = _run_with_args(STEP1_SCRIPT, flag)
    if rc != 0:
        print(f'[ERROR] update_pregao.py exited with code {rc}')
        return False
    return True


def step2_bcb_secondary() -> bool:
    """
    Incremental BCB secondary market refresh.
    Downloads only the current month (and previous month as safety buffer),
    replaces those months in the existing parquet, and saves.
    BCB publishes monthly ZIPs — re-downloading the full history daily would be ~25 years.
    """
    _header(2, 'BCB secondary market — current month refresh')

    sys.path.insert(0, str(HERE))
    from bdbcb_secondary_v2 import download_bcb_secondary_market_month

    raw_path = DATA / 'bcb_mercado_secundario_raw.parquet'
    if not raw_path.exists():
        print('[WARN] bcb_mercado_secundario_raw.parquet not found — run full historical download first.')
        print('       python bdbcb_secondary_v2.py')
        return False

    today    = pd.Timestamp.today()
    # Download current month + previous month (safety: BCB sometimes corrects prior month)
    months   = [
        (today - pd.DateOffset(months=1)).strftime('%Y%m'),
        today.strftime('%Y%m'),
    ]

    new_frames = []
    import requests
    with requests.Session() as sess:
        for yyyymm in months:
            try:
                print(f'  Downloading BCB secondary {yyyymm}...')
                from bdbcb_secondary_v2 import download_bcb_secondary_market_month
                df_m = download_bcb_secondary_market_month(yyyymm, tipo='T', session=sess)
                new_frames.append(df_m)
                print(f'  {yyyymm}: {len(df_m):,} rows')
            except Exception as e:
                print(f'  [WARN] {yyyymm} failed: {e}')

    if not new_frames:
        print('[ERROR] No new data downloaded.')
        return False

    df_new = pd.concat(new_frames, ignore_index=True)

    # Load existing parquet and drop the refreshed months
    df_old = pd.read_parquet(raw_path)
    df_old['download_month'] = df_old['download_month'].astype(str)
    df_old = df_old[~df_old['download_month'].isin(months)]

    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined.to_parquet(raw_path, index=False)
    print(f'[OK] Saved {len(df_combined):,} rows → {raw_path}')
    return True


def step3_vna() -> bool:
    _header(3, 'VNA rebuild — SIDRA IPCA → vna_ntnb.parquet  (build_vna.py)')
    rc = _run(STEP3_SCRIPT)
    if rc != 0:
        print(f'[ERROR] build_vna.py exited with code {rc}')
        return False
    return True


def step4_compute_breakeven() -> bool:
    _header(4, 'MySQL + secondary + VNA → breakeven parquets  (compute_breakeven.py)')
    rc = _run(STEP4_SCRIPT)
    if rc != 0:
        print(f'[ERROR] compute_breakeven.py exited with code {rc}')
        return False
    return True


def step5_refresh_data() -> bool:
    _header(5, 'MySQL DI1 → di1_futs.parquet  (refresh_data.py)')
    rc = _run(STEP5_SCRIPT)
    if rc != 0:
        print(f'[ERROR] refresh_data.py exited with code {rc}')
        return False
    return True


def step6_nss_panel() -> bool:
    _header(6, 'NSS breakeven panel  (build_nss_panel.py)')
    rc = _run(STEP6_SCRIPT)
    if rc != 0:
        print(f'[ERROR] build_nss_panel.py exited with code {rc}')
        return False
    return True


def step7_git_push() -> bool:
    _header(7, 'git add → commit → push')
    today = date.today().strftime('%Y-%m-%d')

    _git(['add', 'dados_bcb_secundario/'])
    rc_commit = _git(['commit', '-m', f'data: refresh {today}'])

    if rc_commit == 1:
        print('[INFO] Nothing to commit — data unchanged.')
        return True
    if rc_commit != 0:
        print(f'[ERROR] git commit failed (code {rc_commit})')
        return False

    rc_push = _git(['push'])
    if rc_push != 0:
        print(f'[ERROR] git push failed (code {rc_push})')
        return False

    print('[OK] Pushed — Streamlit Cloud will redeploy automatically.')
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Daily pipeline runner.')
    parser.add_argument('--no-push',   action='store_true',
                        help='Skip git push (steps 1-5 only)')
    parser.add_argument('--from-step', type=int, default=1, metavar='N',
                        help='Start from step N (1-6, default 1)')
    parser.add_argument('--date',      metavar='YYYY-MM-DD', default=None,
                        help='Process a specific date for step 1 (default: today)')
    args = parser.parse_args()

    steps = {
        1: lambda: step1_update_pregao(args.date),
        2: step2_bcb_secondary,
        3: step3_vna,
        4: step4_compute_breakeven,
        5: step5_refresh_data,
        6: step6_nss_panel,
        7: step7_git_push,
    }

    if args.no_push:
        steps.pop(7)

    print(f'\n  Brazilian Rates — Daily Pipeline')
    print(f'  {args.date or date.today()}')

    for n, fn in steps.items():
        if n < args.from_step:
            continue
        ok = fn()
        if not ok:
            print(f'\n[ABORT] Pipeline stopped at step {n}.')
            sys.exit(1)

    print(f'\n{"="*60}')
    print('  All steps completed successfully.')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
