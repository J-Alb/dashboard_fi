"""
build_nss_panel.py — NSS breakeven panel with warm-start chain.

Fits Nelson-Siegel-Svensson curves to:
  - r_nominal  (Prefixado flat-forward zero rates from breakeven_panel_bfuts)
  - r_real     (NTN-B flat-forward real zero rates from breakeven_panel_bfuts)

Each day's (λ₁, λ₂) is seeded from the previous day's fit, making the
optimizer fast and the parameter series smooth.  Full grid search is used
only on the first date and after a failed/degenerate fit.

Outputs
-------
  dados_bcb_secundario/breakeven_panel_nss.parquet
      columns: date, du, r_nominal, r_real, breakeven  (all in % p.a.)
  dados_bcb_secundario/nss_params.parquet
      columns: date, curve, b1..b4, l1, l2, rmse_bps, r2

Usage
-----
    python build_nss_panel.py
    python build_nss_panel.py --from 2010-01-01
    python build_nss_panel.py --from 2010-01-01 --to 2026-03-16
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
_DATA = _HERE / 'dados_bcb_secundario'
sys.path.insert(0, str(_HERE))

from br_bonds.nss import fit_nss, NSSResult


# ── constants ─────────────────────────────────────────────────────────────────

DU_GRID = np.array([126, 252, 378, 504, 630, 756, 882, 1008,
                    1134, 1260, 1386, 1512, 1638, 1764, 1890,
                    2016, 2142, 2268, 2394, 2520], dtype=float)

_MIN_R2      = 0.85   # below this, treat fit as failed and reset warm start
_RATE_LO_PCT = -10.0  # fitted NSS rate floor (% — real rates hit -2% during COVID)
_RATE_HI_PCT = 60.0   # fitted NSS rate ceiling (%)


# ── core fitting function ─────────────────────────────────────────────────────

def _fit_one(du: np.ndarray, rates_pct: np.ndarray,
             prev_lam: tuple[float, float] | None,
             n_starts_cold: int = 5) -> NSSResult | None:
    """
    Fit NSS to zero rates (in %).  Warm-starts from prev_lam if provided.

    Returns NSSResult, or None if fitting fails / result is degenerate.
    """
    rates = rates_pct / 100.0
    mask  = np.isfinite(rates) & (du > 0)
    du_ok, rates_ok = du[mask], rates[mask]
    if len(du_ok) < 4:
        return None

    try:
        if prev_lam is not None:
            # Warm start: single-point grid at previous λ, one NM refinement
            nss = fit_nss(
                du_ok, rates_ok,
                lam1_grid=np.array([prev_lam[0]]),
                lam2_grid=np.array([prev_lam[1]]),
                n_starts=1,
            )
        else:
            # Cold start: full grid
            nss = fit_nss(du_ok, rates_ok, n_starts=n_starts_cold)
    except Exception:
        return None

    def _is_degenerate(n: NSSResult) -> bool:
        if n.r2 < _MIN_R2:
            return True
        # Evaluate curve at data points — reject if rates blow up
        fitted_pct = n.ytm(du_ok) * 100
        if np.any(fitted_pct < _RATE_LO_PCT) or np.any(fitted_pct > _RATE_HI_PCT):
            return True
        return False

    if _is_degenerate(nss):
        if prev_lam is not None:
            # Retry cold
            try:
                nss = fit_nss(du_ok, rates_ok, n_starts=n_starts_cold)
            except Exception:
                return None
            if _is_degenerate(nss):
                return None
        else:
            return None

    return nss


# ── main panel builder ────────────────────────────────────────────────────────

def compute_breakeven_nss(
    pivot_nom:  pd.DataFrame,
    pivot_real: pd.DataFrame,
    du_grid:    np.ndarray = DU_GRID,
    verbose:    bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit NSS with warm-start to each date in the panels.

    Parameters
    ----------
    pivot_nom  : DataFrame, shape (n_dates, n_du), values = r_nominal in % p.a.
    pivot_real : DataFrame, shape (n_dates, n_du), values = r_real    in % p.a.
    du_grid    : tenors (business days) at which to record results.
    verbose    : print progress every 20 dates.

    Returns
    -------
    panel  : DataFrame  — date, du, r_nominal, r_real, breakeven (% p.a.)
    params : DataFrame  — date, curve, b1..b4, l1, l2, rmse_bps, r2
    """
    dates = pivot_nom.index.intersection(pivot_real.index).sort_values()
    du_cols = np.array(pivot_nom.columns, dtype=float)

    prev_lam_nom  = None
    prev_lam_real = None

    panel_rows  = []
    param_rows  = []
    n_ok = n_skip = 0

    for i, ts in enumerate(dates):
        row_nom  = pivot_nom.loc[ts].values.astype(float)
        row_real = pivot_real.loc[ts].values.astype(float)

        nss_nom  = _fit_one(du_cols, row_nom,  prev_lam_nom)
        nss_real = _fit_one(du_cols, row_real, prev_lam_real)

        if nss_nom is None or nss_real is None:
            n_skip += 1
        else:
            # Update warm-start state
            prev_lam_nom  = (float(nss_nom.lam[0]),  float(nss_nom.lam[1]))
            prev_lam_real = (float(nss_real.lam[0]), float(nss_real.lam[1]))
            n_ok += 1

            # Breakeven panel at du_grid
            r_nom_grid  = nss_nom.ytm(du_grid)   * 100
            r_real_grid = nss_real.ytm(du_grid)  * 100
            brkv_grid   = ((1 + r_nom_grid / 100) / (1 + r_real_grid / 100) - 1) * 100

            for j, du in enumerate(du_grid):
                panel_rows.append({
                    'date':      ts,
                    'du':        int(du),
                    'r_nominal': round(r_nom_grid[j],  6),
                    'r_real':    round(r_real_grid[j], 6),
                    'breakeven': round(brkv_grid[j],   6),
                })

            # NSS parameter rows
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
            print(f'  {i+1:>5}/{len(dates)}  ok={n_ok}  skip={n_skip}',
                  flush=True)

    if verbose:
        print(f'  Finished: {n_ok} dates fitted, {n_skip} skipped.')

    panel  = pd.DataFrame(panel_rows)
    params = pd.DataFrame(param_rows)
    return panel, params


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Build NSS breakeven panel.')
    parser.add_argument('--from', dest='date_from', default=None,
                        metavar='YYYY-MM-DD', help='Start date (default: all)')
    parser.add_argument('--to',   dest='date_to',   default=None,
                        metavar='YYYY-MM-DD', help='End date   (default: all)')
    args = parser.parse_args()

    print('\n=== build_nss_panel.py ===\n')

    # Load source panel
    src = _DATA / 'breakeven_panel_bfuts.parquet'
    if not src.exists():
        print(f'ERROR: {src} not found — run compute_breakeven.py first.')
        sys.exit(1)

    print(f'Loading {src.name}...', flush=True)
    pb = pd.read_parquet(src)
    pb['date'] = pd.to_datetime(pb['date'])

    if args.date_from:
        pb = pb[pb['date'] >= pd.Timestamp(args.date_from)]
    if args.date_to:
        pb = pb[pb['date'] <= pd.Timestamp(args.date_to)]

    pivot_nom  = pb.pivot_table(values='r_nominal', columns='du', index='date')
    pivot_real = pb.pivot_table(values='r_real',    columns='du', index='date')

    print(f'  {pivot_nom.shape[0]} dates  x  {pivot_nom.shape[1]} tenors')
    print(f'  Date range: {pivot_nom.index.min().date()} to '
          f'{pivot_nom.index.max().date()}')
    print()

    print('Fitting NSS (warm-start chain)...')
    panel, params = compute_breakeven_nss(pivot_nom, pivot_real, verbose=True)

    # Save
    out_panel  = _DATA / 'breakeven_panel_nss.parquet'
    out_params = _DATA / 'nss_params.parquet'
    panel.to_parquet(out_panel,  index=False)
    params.to_parquet(out_params, index=False)

    print(f'\nSaved:')
    print(f'  {out_panel.name}   {len(panel):,} rows  '
          f'({panel["date"].nunique()} dates)')
    print(f'  {out_params.name}  {len(params):,} rows')
    print()
    print('Next: git add dados_bcb_secundario/ && git commit -m "data: NSS panel" '
          '&& git push')


if __name__ == '__main__':
    main()
