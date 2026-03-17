# br_bonds

A Python library for pricing and analysing Brazilian government bonds (Títulos Públicos Federais), following ANBIMA/STN official methodology.

## Supported instruments

| Instrument | Type | Convention |
|---|---|---|
| LTN | Zero-coupon prefixado | BUS/252, face = R$ 1000 |
| NTN-F | Semi-annual coupon prefixado (10% p.a.) | BUS/252, face = R$ 1000 |
| NTN-B | Semi-annual IPCA-linked real bond | BUS/252, cotação (% of VNA) |
| LFT | Zero-coupon Selic-linked floating rate | BUS/252, PU = VNA / (1 + deságio)^(du/252) |
| DI1 futures | B3 interbank rate futures | BUS/252, notional R$ 100k |
| DAP futures | B3 IPCA coupon futures (implied real rate) | BUS/252, notional R$ 100k |
| DDI futures | B3 dollar coupon futures (implied USD coupon) | Simple act/360, notional R$ 100k |

## Installation

```bash
pip install numpy scipy pandas bizdays
```

Clone or copy the `br_bonds/` folder into your project, then import directly:

```python
import br_bonds as bb
```

## Quick start

```python
from bizdays import Calendar
import numpy as np
import pandas as pd
import br_bonds as bb

cal = Calendar.load('ANBIMA')
```

### Standalone pricing

```python
# LTN: zero-coupon prefixado
bb.price_ltn(ytm=0.13, du=252)          # 1Y at 13% → R$ 884.96

# NTN-F: semi-annual coupon prefixado
bb.price_ntnf(ytm=0.13, du=1260)        # 5Y at 13% → R$ 896.92

# NTN-B: IPCA-linked (cotação = % of VNA)
bb.price_ntnb(ytm_real=0.07, du=1260)   # 5Y at 7% real → 95.96
# PU (R$) = cotação × VNA / 100
```

### Yield curve interpolation

```python
# Prepare a panel: date, maturity, yield_base (decimal), coupon
panel['coupon'] = panel['bond_type'].map({'LTN': 0.0, 'NTN-F': 0.10})

# NTN-F coupon dates: Jan 1 / Jul 1, adjusted to next business day
raw = pd.date_range('2000-01-01', '2040-01-01', freq='6MS')
cdates_adj = np.array(
    [cal.adjust_next(d) if not cal.isbizday(d) else d for d in raw],
    dtype='datetime64[D]',
)

curve = bb.PrefixadoCurve(
    panel, cal,
    coupon_col='coupon',
    coupon_dates=cdates_adj,   # enables exact NTN-F stub handling
)

# Flat-forward interpolated YTM at any maturity
y5y = curve.ytm(date, du_target=1260)

# Bootstrapped zero (spot) curve
zc = curve.zero_curve(date)
# → DataFrame: du, bond_type, zero_rate, discount_factor

# Constant-maturity time series
cm = curve.build_series(dates, du_target=1260)
# → DataFrame indexed by date: ytm, price
```

### NTN-B real yield curve

```python
curve_real = bb.NTNBCurve(
    panel, cal,
    coupon_col='coupon',
    vna_series=vna,   # optional pd.Series of daily VNA values
)

ytm_real  = curve_real.ytm(date, du_target=1260)
cotacao   = curve_real.cotacao(date, du_target=1260, mat=mat)  # exact schedule
pu        = curve_real.pu(date, du_target=1260, mat=mat)       # R$, needs VNA
zc_real   = curve_real.zero_curve(date)
```

### Nelson-Siegel-Svensson (ANBIMA ETTJ)

```python
# ANBIMA ETTJ PREF methodology: bootstrap zeros → fit NSS
nss = bb.fit_nss_prefixado(day_bonds, cal, coupon_dates=cdates_adj)
nss.ytm(1260)           # 5Y zero rate (decimal)
nss.discount_factor(252)
nss.curve()             # smooth DataFrame: du, ytm, discount_factor

# ANBIMA ETTJ IPCA methodology: fit NSS to NTN-B prices
nss_real = bb.fit_nss_ntnb(day_bonds, cal)

# Evaluate NSS with known parameters (e.g. ANBIMA published values)
beta = [0.1114, 0.0343, 0.0295, 0.0922]
lam  = [3.2647, 0.1840]
bb.nss_ytm(du=1260, beta=beta, lam=lam)
```

#### fit_nss options

`fit_nss` (called internally by all `fit_nss_*` functions) exposes two extra parameters
to improve robustness against local minima:

| Parameter | Default | Description |
|---|---|---|
| `n_starts` | `3` | Run Nelder-Mead from the top-N grid candidates, keep the best result |
| `global_search` | `False` | Use `differential_evolution` over (λ₁, λ₂) with β solved analytically at each step — slower but avoids local minima |

Default λ grids (widened vs earlier versions): `λ₁ ∈ [0.1, 8.0]` step 0.1, `λ₂ ∈ [0.05, 3.0]` step 0.05.

```python
# Pass through fit_nss_prefixado / fit_nss_ntnb via **kwargs
nss = bb.fit_nss_prefixado(day_bonds, cal, coupon_dates=cdates_adj, n_starts=5)
nss = bb.fit_nss_ntnb(day_bonds, cal, global_search=True)
```

#### Dense flat-forward augmentation (small bond sets)

When fitting NSS to a small set of bonds, augment the bootstrapped zeros with a dense
flat-forward grid before calling `fit_nss`. This improves curve smoothness and reduces
local-minimum trapping:

```python
from br_bonds._interpolation import flatfwd_df

du_dense = np.arange(int(du_v.min()), int(du_v.max()) + 1, 21, dtype=float)
df_dense = np.array([flatfwd_df(du_v, df_v, d) for d in du_dense])
zr_dense = df_dense ** (-252.0 / du_dense) - 1.0

# Up-weight NTN-F nodes to anchor the long end when LTNs dominate
ntnf_dus = set(zero_ntnf.loc[zero_ntnf['bond_type'] == 'NTN-F', 'du'].values.astype(int))
w = np.array([5.0 if int(d) in ntnf_dus else 1.0 for d in du_dense])

nss = bb.fit_nss(du_dense, zr_dense, weights=w, n_starts=5)
```

### Duration and DV01

```python
mac, mod, dv01 = bb.bond_duration(ytm=0.13, du=1260)
# mac  : Macaulay duration (business years)
# mod  : Modified duration (years)
# dv01 : R$ per bp per R$1000 face

# Annotate a constant-maturity DataFrame
cm = bb.add_duration(cm, du_target=1260)
# adds columns: mac_dur, mod_dur, dv01
```

### Total return and excess return indices

```python
# Constant-maturity TRI (includes daily carry accrual)
tri = bb.cm_tri(cm['price'].values, cm['ytm'].values)

# Total return index (absolute) or excess return (over risk-free)
rf_3m = np.array([curve.ytm(d, 63) for d in dates])
tri = bb.ret_index(prices, ytms)              # absolute TRI
eri = bb.ret_index(prices, ytms, rf_3m)       # excess return index
```

### NTN-B buy-and-hold TRI

`ntnb_tri` builds a buy-and-hold NTN-B total return index from daily PU series,
with explicit coupon reinvestment. Does not require a yield assumption — IPCA accrual
and roll-down are already embedded in the PU ratio.

```python
tri = bb.ntnb_tri(
    pus,          # array of daily PU values (R$)
    vna,          # array of daily VNA values (R$)
    is_coupon,    # bool array, True on coupon payment dates
    coupon_real=0.06,   # contractual real coupon rate (6% p.a.)
    freq=2,             # payments per year (semi-annual)
    face=100.0,         # face in cotação units
)
```

### Constant-duration strategy

```python
# Solve the maturity that achieves a fixed modified duration each day
du_star = bb.du_for_mod_dur(date, curve, dur_target=4.0)

# Build a full daily constant-duration price series
cd = bb.build_cd_series(dates, curve, dur_target=4.0)
# → DataFrame indexed by date: du, ytm, price, mod_dur

cd['TRI'] = bb.ret_index(cd['price'].values, cd['ytm'].values)
cd['ERI'] = bb.ret_index(cd['price'].values, cd['ytm'].values, rf_3m)
```

### LFT (Selic-linked floating rate bond)

```python
# VNA Selic: download from BCB API and cache
vna_selic = bb.fetch_vna_selic()   # pd.Series indexed by Timestamp, values in R$
vna_selic.to_frame('vna').to_parquet('vna_selic.parquet')

# Standalone pricing: PU = VNA / (1 + ytm)^(du/252)
pu  = bb.price_lft(ytm=0.002, du=252, vna=13_500.0)  # 0.20% deságio, 1Y
ytm = bb.ytm_lft(pu=13_450.0, du=252, vna=13_500.0)  # analytical inversion

# Yield curve (flat-forward interpolation of deságio yields)
curve_lft = bb.LFTCurve(panel, cal, vna_series=vna_selic)

y1y = curve_lft.ytm(date, 252)        # 1Y deságio yield
pu  = curve_lft.pu(date, 252)         # PU in R$ (requires vna_series)
zc  = curve_lft.zero_curve(date)      # zero_rate = ytm (no bootstrap needed)
cm  = curve_lft.build_series(dates, 252)  # DataFrame: ytm, vna, pu
```

**Key conventions:**
- VNA Selic base: R$1,000 on 2000-07-01 (Tesouro Nacional base date)
- Yield = *taxa de deságio* — typically in [−0.01, +0.02] (small spread over Selic)
- No coupons: zero rate = par yield = YTM (no bootstrap required)

### DI1 futures

```python
contract = bb.DI1Contract(rate=0.135, du=252)
contract.price   # R$ 88,105.73
contract.dv01    # R$ per bp per contract

bb.di1_price(0.135, 252)
bb.di1_dv01(0.135, 252)
```

### DAP futures (Futuro de Cupom de IPCA)

DAP implies a real (IPCA coupon) rate. Same BUS/252 compound convention as DI1.
Maturities fall on the 15th of each month, adjusted to next ANBIMA business day
(same anniversary dates as NTN-B coupons).

```python
# Standalone pricing
bb.dap_price(rate=0.065, du=504)   # 2Y at 6.5% real → R$ 87,848
bb.dap_dv01(rate=0.065, du=504)    # DV01 in R$ per bp per contract
bb.dap_rate(price=87_848.0, du=504)  # invert price → rate

# Dataclass
contract = bb.DAPContract(rate=0.065, du=504)
contract.price
contract.dv01

# Enrich a settlement DataFrame (from MySQL ajuste_pregao)
# Note: AJUSTE_ATUAL stores settlement PU (price), not rate.
# Use secondary.build_breakeven_futures for curve work (handles PU inversion).
df_enriched = bb.dap_from_df(df_dap)
# adds columns: du, rate (decimal), price, dv01
```

**Convention:** `PU = 100 000 / (1 + rate)^(du/252)`

### DDI futures (Futuro de Cupom Cambial)

DDI implies a dollar coupon rate using **simple interest act/360** — unlike
DI1/DAP which use compound BUS/252.

```python
# Standalone pricing
bb.ddi_price(rate=0.045, dc=360)   # 1Y at 4.5% → R$ 95,694
bb.ddi_dv01(rate=0.045, dc=360)    # DV01 in R$ per bp per contract
bb.ddi_rate(price=95_694.0, dc=360)  # invert price → rate

# Dataclass
contract = bb.DDIContract(rate=0.045, dc=360)
contract.price
contract.dv01

# Enrich a settlement DataFrame
df_enriched = bb.ddi_from_df(df_ddi)
# adds columns: dc (calendar days), rate (decimal), price, dv01
```

**Convention:** `PU = 100 000 / (1 + rate × dc/360)` — linear act/360, where `dc` is calendar days.

### Futures breakeven panel (DI1 vs DAP)

Builds a daily implied inflation breakeven panel from B3 settlement data.
Both DI1 and DAP are zero-coupon contracts — their settlement PU directly
implies a zero rate, so no bootstrapping is needed. Fisher equation applied
after flat-forward interpolation.

```python
import mysql.connector
import pandas as pd

conn   = mysql.connector.connect(host='localhost', user='root',
                                 password='...', database='db_01')
df_di1 = pd.read_sql("SELECT * FROM ajuste_pregao WHERE ATIVO LIKE 'DI1%'", conn)
df_dap = pd.read_sql("SELECT * FROM ajuste_pregao WHERE ATIVO LIKE 'DAP%'", conn)
conn.close()

panel = bb.build_breakeven_futures(df_di1, df_dap)
# → DataFrame: date, du, years, r_nominal (%), r_real (%), breakeven (%)

panel.to_parquet('dados_bcb_secundario/breakeven_panel_futures.parquet')

pivot = panel.pivot_table(values='breakeven', columns='du', index='date')
```

**Notes:**
- `AJUSTE_ATUAL` in the source table stores settlement **PU** (~93,000), not a rate.
  The function inverts via `(100 000 / PU)^(252/du) − 1` internally.
- `VENCIMENTO` may be stored as B3 contract codes (`K26`, `F07`, etc.) or as
  date strings — both are handled automatically.
- DAP liquidity is thin before 2018; filter `date >= '2018-01-01'` for reliable series.

### VNA — IPCA-accreted face value (NTN-B)

```python
# Fetch IPCA Número-índice from SIDRA (table 1737, variable 2266)
ipca_idx = bb.fetch_ipca_index()
# → pd.Series indexed by month-start Timestamp, 13 decimal places

# Build daily VNA series from 2000-07-17 (ANBIMA base) to end_date
vna = bb.build_vna(
    end_date       = pd.Timestamp('2026-03-14'),
    ipca_index     = ipca_idx,         # fetched automatically if None
    projected_ipca = None,             # pd.Series of % rates for unpublished months
    verbose        = True,
)
# → pd.Series indexed by ANBIMA business day Timestamps, values in R$
# VNA base: R$ 1,000.00 on 2000-07-17 (July 15 was Saturday)

vna.to_frame('vna').to_parquet('dados_bcb_secundario/vna_ntnb.parquet')
```

**Projection fallback chain** when SIDRA hasn't yet published a month:
1. `projected_ipca[month]` — caller-supplied % rate (e.g., `0.52` for 0.52%)
2. Previous month's realized ratio (ANBIMA default)

### Breakeven inflation panel (secondary market)

Builds a daily panel of implied real yields, nominal yields, and breakeven inflation
from BCB secondary market data using flat-forward interpolation on bootstrapped zero curves.
No NSS fitting — pure flat-forward.

```python
from br_bonds import load_secondary_data, build_vna, build_breakeven_panel

# 1. Load and clean BCB secondary market data
dta = bb.load_secondary_data(
    'dados_bcb_secundario/bcb_mercado_secundario_raw.parquet',
    codes=[100000, 950199, 760199],  # LTN, NTN-F, NTN-B
)
# → DataFrame with columns: date, maturity, bond_type, yield_base,
#   price_base, pu_lastro, codigo, valor_par, num_de_oper, du, coupon

# 2. Build VNA series
vna_series = bb.build_vna(verbose=True)

# 3. Build breakeven panel (parallel, joblib)
import numpy as np
panel = bb.build_breakeven_panel(
    dta,
    vna_series,
    dates   = None,                               # all dates in dta
    du_brkv = np.arange(126, 2646, 126, dtype=float),   # 6M to ~10.5Y
    n_jobs  = -1,
    verbose = True,
)
# → DataFrame: date, du, years, r_nominal (%), r_real (%), breakeven (%)

panel.to_parquet('dados_bcb_secundario/breakeven_panel_ff.parquet')

# Optional: augment the NTN-B real zero curve with DAP anchors.
# DAP maturities fall on the 15th (same as NTN-B), so nodes align exactly.
# NTN-B bootstrapped nodes win at matching maturities; DAP fills genuine gaps
# (short end below the first NTN-B, and slots between NTN-B maturities).
# Recommended: supply df_dap only from 2018+ when DAP liquidity is reliable.
conn   = mysql.connector.connect(host='localhost', user='root',
                                 password='...', database='db_01')
df_dap = pd.read_sql("SELECT * FROM ajuste_pregao WHERE ATIVO LIKE 'DAP%'", conn)
conn.close()

panel_aug = bb.build_breakeven_panel(dta, vna_series, df_dap=df_dap)

# Pivot for time-series analysis
pivot = panel.pivot_table(values='breakeven', columns='du', index='date')
```

**Pipeline detail:**
- NTN-B real yields: invert BCB `pu_med` via `ytm_ntnb(pu, vna_t, date, mat, cal)` — exact ANBIMA cashflow schedule
- Prefixado nominal yields: invert NTN-F from prices via `ytm_ntnf`; LTN uses BCB `taxa_med` directly
- Zero curves: bootstrapped from bond prices via `NTNBCurve.zero_curve()` and `PrefixadoCurve.zero_curve()`
- Breakeven: Fisher equation `(1 + r_nominal) / (1 + r_real) − 1`
- Filters applied: minimum 2 operations, cotação sanity check, valley/peak outlier removal on input yields and bootstrapped zeros, negative-forward fallback to LTN-only curve

Low-level access:

```python
from br_bonds import get_pre_curve, get_ntnb_curve, TESOURO_MAP

# Filter bonds for a single date (with quality filters applied)
pre  = bb.get_pre_curve(dta, date)    # LTN + NTN-F
ntnb = bb.get_ntnb_curve(dta, date)  # NTN-B

# Bond-type mapping from raw BCB sigla strings
bb.TESOURO_MAP['TESOURO IPCA+ COM JUROS SEMESTRAIS']  # → 'NTN-B'
```

## Package structure

```
br_bonds/
├── __init__.py          # public API
├── _interpolation.py    # flat-forward and linear interpolation
├── _schedules.py        # cashflow schedule builders (LTN/NTN-F, NTN-B)
├── prefixado.py         # price_ltn, ytm_ltn, price_ntnf, ytm_ntnf, build_tri, PrefixadoCurve
├── ntnb.py              # price_ntnb, ytm_ntnb, NTNBCurve
├── lft.py               # price_lft, ytm_lft, fetch_vna_selic, LFTCurve
├── nss.py               # NSSResult, nss_ytm, fit_nss, fit_nss_*
├── analytics.py         # duration, TRI/ERI, constant-duration solver
├── vna.py               # fetch_ipca_index, build_vna (NTN-B IPCA accrual)
├── secondary.py         # load_secondary_data, build_breakeven_panel, build_breakeven_futures
└── derivatives/
    ├── di1.py           # DI1Contract, di1_price, di1_dv01, DI1Curve
    ├── dap.py           # DAPContract, dap_price, dap_dv01, dap_rate, dap_from_df
    ├── ddi.py           # DDIContract, ddi_price, ddi_dv01, ddi_rate, ddi_from_df
    └── copom.py         # CopomCurve, fetch_di1, fetch_overnight
```

## Methodology notes

### ANBIMA BUS/252 day count
All pricing uses business days counted on the ANBIMA calendar (`bizdays`), with maturity dates adjusted to the next business day when they fall on holidays or weekends. The discount factor is `(1 + r)^(du/252)`.

### NSS parameterisation
ANBIMA publishes λ as a Diebold-Li decay **rate** — the basis function is `φ(λ, t) = (1 − exp(−λt)) / (λt)` with `x = λ·t`. Using the Svensson convention (`x = t/λ`) causes errors of ~177 bp at the short end.

### Flat-forward interpolation
The ANBIMA standard: discount factors are interpolated log-linearly between vertices. The zero rate at `du` is recovered as `df(du)^(−252/du) − 1`.

### NTN-F coupon convention
ANBIMA uses a **fixed** semi-annual coupon for all periods including stubs:
`C = face × [(1 + coupon)^(1/freq) − 1]`
Pro-rating stub coupons is incorrect and increases pricing errors substantially.

## Validation

Validated against ANBIMA indicative rate files (2026-03-12):

| Instrument | Max error vs ANBIMA |
|---|---|
| LTN PU | < 0.000002 R$ (13 bonds) |
| NTN-F PU | < 0.000025 R$ (6 bonds) |
| NTN-B cotação | < 0.01 R$ (14 bonds) |
| NSS PREF vs ETTJ vertices | < 0.01 bp |
| NSS IPCA vs ETTJ (du ≤ 756) | < 0.67 bp |

## Possible next steps

### VNA
- **`fetch_ipca_projection()` auto from Focus/IPCA-15** — auto-fills `projected_ipca` without manual input. Priority chain: realized SIDRA → IPCA-15 (SIDRA table 3065) → Focus median (BCB Expectativas API). Convenience wrapper `build_vna_auto()`.
- **NTN-C support** — IGP-M-linked VNA using SIDRA table 1629 (IGP-M Número-índice).

### Secondary market / breakeven
- **Short-end anchor** — pass `df_dap` to `build_breakeven_panel` to extend the real zero curve with DAP monthly contracts below the shortest NTN-B maturity (typically 2Y+). DAP expiry on the 15th aligns exactly with NTN-B; NTN-B wins at overlapping maturities. Reliable from 2018+.
- **NTN-B Principal inclusion** — zero-coupon IPCA bonds (NTN-B P) simplify yield inversion (no cashflow schedule needed) and improve coverage at short and very long maturities. Currently excluded (`codigo=760199` only captures NTN-B coupon bonds).
- **Liquidity weighting** — weight bootstrapped zero-curve nodes by `num_de_oper` or notional volume before flat-forward interpolation, reducing noise from thinly-traded bonds.
- **Incremental update** — `build_breakeven_panel` currently reprocesses all dates. Add an incremental mode that loads an existing panel and appends only new dates.
- **Outlier post-processing** — apply a cross-sectional z-score or rolling median filter to the final breakeven panel to flag dates where all tenors shift anomalously (e.g., data errors vs genuine moves).

### Yield curve analytics
- **Historical NSS panel** — roll `fit_nss_prefixado` and `fit_nss_ntnb` across the full date range of secondary market data, saving daily β and λ parameters for factor analysis.
- **Forward rate surface** — build `r_nominal(date, du)` and `r_real(date, du)` grids and expose a `fra(date, short_du, long_du)` helper.
- **LFT pricing** — Selic-linked floating rate bonds (different convention: discount from `du` to 0 at overnight CDI).
- **NTN-C pricing** — IGP-M-linked (similar to NTN-B but uses IGP-M VNA).

## Dependencies

- `numpy >= 1.20`
- `scipy >= 1.7`
- `pandas >= 1.3`
- `bizdays`
- `requests` (for `fetch_ipca_index` and `fetch_vna_selic`)
- `joblib` (optional — required only for `build_breakeven_panel` with `n_jobs != 1`)
