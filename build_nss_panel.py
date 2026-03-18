"""
build_nss_panel.py — CLI wrapper for br_bonds.secondary.build_nss_panel.

Reads breakeven_panel_bfuts.parquet (flat-forward panel from compute_breakeven.py),
fits NSS with warm-start chain, and saves:
  dados_bcb_secundario/breakeven_panel_nss.parquet
  dados_bcb_secundario/nss_params.parquet

Usage
-----
    python build_nss_panel.py
    python build_nss_panel.py --from 2010-01-01
    python build_nss_panel.py --from 2010-01-01 --to 2026-03-16
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent
_DATA = _HERE / 'dados_bcb_secundario'
sys.path.insert(0, str(_HERE))

from br_bonds.secondary import build_nss_panel


def main() -> None:
    parser = argparse.ArgumentParser(description='Build NSS breakeven panel.')
    parser.add_argument('--from', dest='date_from', default=None,
                        metavar='YYYY-MM-DD')
    parser.add_argument('--to',   dest='date_to',   default=None,
                        metavar='YYYY-MM-DD')
    args = parser.parse_args()

    print('\n=== build_nss_panel.py ===\n')

    src = _DATA / 'breakeven_panel_bfuts.parquet'
    if not src.exists():
        print(f'ERROR: {src} not found — run compute_breakeven.py first.')
        sys.exit(1)

    print(f'Loading {src.name}...', flush=True)
    panel = pd.read_parquet(src)

    nss_panel, nss_params = build_nss_panel(
        panel,
        date_from=args.date_from,
        date_to=args.date_to,
        verbose=True,
    )

    out_panel  = _DATA / 'breakeven_panel_nss.parquet'
    out_params = _DATA / 'nss_params.parquet'
    nss_panel.to_parquet(out_panel,  index=False)
    nss_params.to_parquet(out_params, index=False)

    print(f'\nSaved:')
    print(f'  {out_panel.name}   {len(nss_panel):,} rows  '
          f'({nss_panel["date"].nunique()} dates)')
    print(f'  {out_params.name}  {len(nss_params):,} rows')


if __name__ == '__main__':
    main()
