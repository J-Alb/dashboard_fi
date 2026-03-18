# Data Flow Schema — `fixed income/`

_Last updated: 2026-03-18_

---

## Pipeline Overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                           EXTERNAL SOURCES                                   ║
╠══════════════╦═════════════╦══════════════════════════╦══════════════════════╣
║ B3 BDI PDFs  ║ BCB HTTP    ║  SIDRA / IBGE            ║  BCB Focus API       ║
║ (b3.com.br)  ║ (secondary  ║  (IPCA index)            ║  (olinda.bcb.gov.br) ║
║              ║  market ZIP)║                          ║                      ║
╚══════╤═══════╩══════╤══════╩══════════╤═══════════════╩════════════╤═════════╝
       │              │                 │                             │
       ▼              ▼                 ▼                             ▼
┌─────────────┐ ┌───────────────┐ ┌──────────────┐         ┌──────────────────┐
│pregao_extrc │ │bdbcb_secondary│ │  build_vna   │         │   get_focus.py   │
│   .py       │ │   _v2.py      │ │    .py       │         │  (Schonfeld/     │
│(Schonfeld/  │ │               │ │              │         │   fixed income)  │
│ functions/) │ └───────┬───────┘ └──────┬───────┘         └────────┬─────────┘
└──────┬──────┘         │                │                          │
       │ called by      │                │                          │
       ▼                ▼                ▼                          ▼
┌──────────────┐  ┌─────────────────────────────┐         ┌──────────────────┐
│update_pregao │  │  bcb_mercado_secundario_raw  │         │ focus_features   │
│   .py        │  │       .parquet  ★ HUB ★      │         │ .csv             │
│(Schonfeld/)  │  └─────────────┬───────────────┘         │(Schonfeld/data/) │
└──────┬───────┘                │                          └──────────────────┘
       │                        │
       ▼                        │        ┌─────────────┐
  ┌─────────────────────────┐   │        │ vna_ntnb    │
  │  MySQL db_01            │   │        │ .parquet    │
  │  ajuste_pregao          │   │        └──────┬──────┘
  │  (ALL futures:          │   │               │
  │   DI1, DAP, DDI,        │   ├───────────────┘
  │   DOL, WIN, etc.)       │   │
  └──────────┬──────────────┘   │
             │                  │
     ┌───────┘                  ▼
     │               ┌──────────────────────────────────────┐
     │               │  implicita_v3.py / compute_breakeven  │
     │               │     (also reads MySQL DI1 + DAP)      │
     ├───────────────►         uses br_bonds.secondary        │
     │               └──────────────┬───────────────────────┘
     │                              │
     │               ┌──────────────┼──────────────┐
     │               ▼              ▼               ▼
     │      ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
     │      │breakeven     │ │breakeven     │ │breakeven         │
     │      │panel_ff      │ │panel_bfuts   │ │panel_futures     │
     │      │.parquet      │ │.parquet      │ │.parquet          │
     │      │(bonds only)  │ │(bonds+DAP)   │ │(DI1+DAP)         │
     │      └──────┬───────┘ └──────┬───────┘ └────────┬─────────┘
     │             │                │                   │
     │     ┌───────┘                └────────┬──────────┘
     │     │                                 │
     ▼     ▼                                 ▼
┌─────────────────┐                ┌─────────────────────────────────────┐
│ refresh_data.py │                │          dashboard.py               │
│ (DI1 only       │                │  reads: di1_futs, bfuts, futures,   │
│  export to      │                │  secondary_raw, BCB SGS API,        │
│  parquet)       │                │  Focus API (get_focus.py)           │
└────────┬────────┘                └─────────────────────────────────────┘
         │
         ▼                         ┌──────────────────────────────────────┐
  ┌──────────────┐                 │        report_prefixado.py           │
  │ di1_futs     ├────────────────►│  reads: secondary_raw, di1_futs,     │
  │ .parquet     │                 │  panel_ff, panel_futures             │
  └──────────────┘                 └──────────────────────────────────────┘
```

---

## Parquet Inventory

| File | Produced by | Consumed by | Notes |
|------|-------------|-------------|-------|
| `bcb_mercado_secundario_raw.parquet` | `bdbcb_secondary_v2.py` | `implicita_v3`, `compute_breakeven`, `panel_builder`, `dashboard`, `report_prefixado` | Central hub — all bond curve work starts here |
| `bcb_mercado_secundario_titulos_publicos_filtrado.parquet` | `bdbcb_secondary_v2.py` | `portfolio_examples.ipynb`, `get_pu_series()` | Sigla-whitelist filtered slice (NTN-B, NTN-F, LTN, LFT); one row per transaction — use volume-weighted avg per date via `get_pu_series()` |
| `vna_ntnb.parquet` | `build_vna.py` | `implicita_v3`, `compute_breakeven`, `panel_builder`, `portfolio_examples.ipynb` | Daily IPCA-accreted face value for NTN-B |
| `cdi_series.parquet` | `refresh_data.py` (BCB SGS series 12) | `dashboard`, `portfolio_examples.ipynb` | Columns: `date`, `cdi_annual` (% p.a. annualised); CDI daily to annual conversion applied |
| `di1_futs.parquet` | `refresh_data.py` (export from MySQL) | `dashboard`, `report_prefixado`, `portfolio_examples.ipynb` | DI1-only slice; ~206k rows 2000-01-03 to present; VENCIMENTO uses B3 codes post-2006 (`K26`) and Portuguese abbreviations pre-2006 (`ABR0`, `NOV1`) — both parsed by `_futures_to_zero_curve` |
| `breakeven_panel_ff.parquet` | `implicita_v3.py` | `report_prefixado` | Bonds-only breakeven (NTN-B vs Prefixado) |
| `breakeven_panel_bfuts.parquet` | `compute_breakeven.py` | `dashboard`, `build_nss_panel`, `portfolio_examples.ipynb` | Bonds + DAP combined breakeven; flat-forward real zero rates at fixed du grid (126–2520); best source for NTN-B rolling constant-maturity ytm |
| `breakeven_panel_futures.parquet` | `implicita_v3.py` / `compute_breakeven.py` | `dashboard`, `report_prefixado`, `portfolio_examples.ipynb` | Futures breakeven (DI1 vs DAP); columns include `r_nominal` (DI1) and `r_real` (DAP) |
| `breakeven_panel_nss.parquet` | `build_nss_panel.py` | `dashboard` (planned) | NSS-smoothed nominal + real + breakeven at standard tenors |
| `nss_params.parquet` | `build_nss_panel.py` | monitoring | Daily NSS β₁–β₄, λ₁–λ₂, RMSE, R² for both nominal and real curves |
| `bcb_curve_panel.parquet` | `bdbcb_secondary_v2.py` | *(unused — orphan?)* | Curve-ready panel, not consumed by any current script |
| `breakeven_panel.parquet` | `panel_builder.py` | *(unused — orphan?)* | Optional output from panel_builder, not wired to dashboard |

All parquets live in `dados_bcb_secundario/`.

---

## Script Roles

| Script | Location | Role |
|--------|----------|------|
| `pregao_extrc.py` | `Schonfeld/functions/` | PDF parser — extracts derivatives settlement prices from B3 BDI PDFs (handles old format pre-2025-12-12 and new format post-2025-12-12, and v2 layout post-2026-02-19) |
| `update_pregao.py` | `Schonfeld/` | Daily ingestion — downloads BDI PDFs, calls pregao_extrc, inserts to MySQL `ajuste_pregao` |
| `bdbcb_secondary_v2.py` | `fixed income/` | Downloads BCB secondary market ZIP files, builds `bcb_mercado_secundario_raw.parquet` |
| `build_vna.py` | `fixed income/` | Fetches IPCA from SIDRA, builds `vna_ntnb.parquet` |
| `get_focus.py` | `Schonfeld/functions/` + `fixed income/` (copy) | Fetches BCB Focus survey expectations; produces `focus_features.csv` |
| `implicita_v3.py` | `fixed income/` | Builds breakeven panels from bonds + futures |
| `compute_breakeven.py` | `fixed income/` | Alternate/updated breakeven builder (produces `breakeven_panel_bfuts`) |
| `panel_builder.py` | `fixed income/` | Historical NSS-based breakeven panel (parallel) — superseded by `build_nss_panel.py` |
| `build_nss_panel.py` | `fixed income/` | Thin CLI wrapper → calls `br_bonds.secondary.build_nss_panel()` (core logic lives there); produces `breakeven_panel_nss.parquet` + `nss_params.parquet`; NSS warm-start chain, cold-start on R²<0.85 |
| `refresh_data.py` | `fixed income/` | Exports `di1_futs.parquet` + `cdi_series.parquet` from MySQL / BCB SGS for cloud deployment |
| `dashboard.py` | `fixed income/` | Streamlit app — reads parquets only (no live MySQL) |
| `report_prefixado.py` | `fixed income/` | Matplotlib report — prefixado curves, COPOM path, time series |

---

## MySQL Table

**`db_01.ajuste_pregao`** — settlement prices for ALL B3 futures contracts

| Column | Description |
|--------|-------------|
| `DATA` | Trade date |
| `ATIVO` | Full instrument label (e.g. `DI1 - DI de 1 Dia`, `DAP - Cupom de IPCA`) — **not** a ticker; use `VENCIMENTO` to identify contracts |
| `VENCIMENTO` | B3 month code identifying the contract (e.g. `F26`, `N07`) — always 3-char post-2006 |
| `AJUSTE_ANTERIOR` | Previous day settlement |
| `AJUSTE_ATUAL` | Current settlement PU |
| `VARIACAO` | Daily change in points |
| `AJUSTE_CONTRATO` | Contract adjustment value (NaN — not in PDF) |

Covers: DI1, DAP, DDI, DOL, WIN, WDO, BGI, CCM, and all other B3 derivatives.

**DAP-specific notes:**
- `ATIVO` values: `'DAP - Cupom de DI x IPCA'` (older) or `'DAP - Cupom de IPCA'` (current)
- Filter: `WHERE ATIVO LIKE 'DAP%'`
- Date range: 2006-08-16 to present (~51k rows); VENCIMENTO always 3-char B3 code
- Parse maturity: `_parse_vencimento(df['VENCIMENTO'], expiry_day=15)` (DAP expires on 15th)
- Do **not** use `parse_ativo()` on the ATIVO column — it contains full labels, not ticker codes

---

## Daily Update Sequence

```
1. python Schonfeld/update_pregao.py      # B3 PDF → MySQL
2. python implicita_v3.py                 # MySQL → breakeven parquets
3. python refresh_data.py                 # MySQL → di1_futs.parquet
4. git push                               # triggers Streamlit Cloud redeploy
```

---

## br_bonds Package

Core pricing and analytics library for the pipeline. Versioning:

| Directory | Status | Notes |
|-----------|--------|-------|
| `br_bonds/` | **production** | Active version imported by all scripts and notebooks |
| `br_bonds_old/` | archived | Original pre-refactor version (kept as backup) |
| `br_bonds_v2/` | *(to be removed)* | Refactored copy — superseded once `br_bonds` is replaced |

Key modules consumed by pipeline scripts:

| Module | Used by |
|--------|---------|
| `secondary.build_breakeven_panel` | `compute_breakeven.py` |
| `secondary.build_breakeven_futures` | `compute_breakeven.py` |
| `secondary.build_nss_panel` | `build_nss_panel.py` |
| `secondary.get_pu_series` | `portfolio_examples.ipynb` |
| `secondary._futures_to_zero_curve` | `portfolio_examples.ipynb` (DI1 rolling loop) |
| `secondary._parse_vencimento` | `portfolio_examples.ipynb` (DAP B&H) |
| `portfolio.Instrument/Position/Portfolio` | `portfolio_examples.ipynb` |
| `derivatives.DI1Curve` | `portfolio_examples.ipynb`, `compute_breakeven.py` |

---

## Tutorial Notebooks

| Notebook | Purpose |
|----------|---------|
| `portfolio_examples.ipynb` | Full walkthrough of `Instrument`, `Position`, `Portfolio`: NTN-B B&H, NTN-B 5Y rolling, NTN-F B&H, DI1 rolling, DAP rolling, DAP B&H (MySQL), comparison plot + drawdown |

---

## Known Orphans / Cleanup Candidates

- `bcb_curve_panel.parquet` — written by `bdbcb_secondary_v2.py`, not read anywhere
- `breakeven_panel.parquet` — written by `panel_builder.py`, not read by dashboard or report
- `implicita_v2.py`, `implicita.py` — older versions, likely superseded by `implicita_v3.py`
- `diag_ff_oscillation.py`, `diag_spikes2.py` — diagnostic scripts, not part of pipeline
