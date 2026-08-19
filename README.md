# Pakistan Macroeconomic Relationships (1947–2025)

Three linked models on annual Pakistani macro data: what drives GDP growth, what
drives the PKR/USD exchange rate, and whether Okun's Law holds for unemployment.

**Headline result: two of the three models fail, and that is the finding.** GDP
growth and short-run exchange-rate deviations are essentially unpredictable from
slow-moving structural ratios — both LOOCV R² values sit at or below zero, i.e.
no better than predicting the mean. Only unemployment is predictable, and the
variable that carries it is not GDP growth but conflict.

## Data and its problems — read this first

| File | Role | Coverage |
|---|---|---|
| `data/pakistan_economic_data_1947_2025.csv` | primary compilation | 1947–2025, 79 rows |
| `data/rupee_vs_dollar.csv` | PKR/USD series | 1947–2025 |
| `data/Pakistan_Yearly_UnemploymentGDP_And_CPI.csv` | World Bank cross-check | 1991–2020 |
| `data/GDP_of_Pakistan_1960-2021.xlsx` | independent GDP levels | 1960–2021 |

The primary file is a third-party compilation and **its pre-1991 portion is not
historical record.** Inflation, unemployment and GDP growth are all carried to
16 decimal places for 1947 — a year in which Pakistan had no statistical
apparatus capable of producing such figures. Those values are interpolated or
generated.

Run `src/verify_provenance.py` to reproduce the checks:

- **1991–2020 vs World Bank:** inflation is *identical* (max absolute difference
  0.000000), so the post-1991 portion is derived from World Bank data. GDP growth
  correlates 0.957, unemployment 0.933.
- **GDP levels vs independent series:** correlation 1.000 from 2016 onward;
  the primary file runs about 13% high over 2007–2015. GDP level is not a model
  input, so this affects context only.

**Consequence:** every model is reported twice — on the full 79-year range and
on the 1991+ verified window (n=35). Where the two disagree, trust the second.

## Method

79 annual observations, 35 in the verified window. That rules out tree ensembles,
which would overfit badly at this sample size. This uses **Ridge regression with
leave-one-out cross-validation** — the right-sized tool for small-N annual macro
data.

Scaling happens *inside* each CV fold via a `Pipeline`. Fitting a scaler on the
full matrix before cross-validating leaks the held-out row's own mean and
variance into its prediction. The effect here is under 0.002 R², but Ridge is
scale-sensitive and the correct construction is free.

## Results

### Model 1 — GDP growth from structural indicators

Features: FDI, inflation, trade balance, military expenditure, sector shares, conflict indicator.

| Window | n | LOOCV R² | MAE |
|---|---|---|---|
| Full range | 79 | **−0.001** | 1.74 |
| 1991+ verified | 35 | **−0.084** | 1.65 |

Both are effectively zero or negative: these slow-moving structural ratios carry
no linear predictive information about a given year's growth beyond the
unconditional mean. Annual growth is dominated by idiosyncratic shocks that
annually-averaged ratios cannot encode.

### Model 2 — Exchange rate, trend plus residual

The trend is real and strong: **6.03% average annual depreciation**, consistent
with the PKR's documented fall from a fixed 3.31/USD to roughly 278/USD by 2024.

The deviations from that trend are not explainable: residual model **LOOCV R² =
−0.007** (n=78) from GDP growth, inflation, trade balance and FDI.

This mirrors the petrol-price finding in the companion study — the smooth
component is forecastable, while short-run deviations driven by discrete events
(IMF negotiations, political crises, speculative pressure) are not captured by
continuous macroeconomic ratios.

### Model 3 — Unemployment and Okun's Law

| Window | n | LOOCV R² | MAE |
|---|---|---|---|
| Full range | 79 | 0.299 | 1.25 |
| 1991+ verified | 35 | **0.624** | 0.98 |

Direct GDP-growth-to-unemployment correlation is negative but weak: **−0.104**
full range, **−0.218** in the verified window. Okun's Law is present but faint,
consistent with mixed findings in the Pakistan literature.

The model outperforms GDP growth alone because of the conflict variable:

| | Mean unemployment | Years |
|---|---|---|
| Conflict years | **1.40%** | 29 |
| Non-conflict years | **3.32%** | 50 |

The gap is stable across both the full range and the verified window. **This is
correlational.** The data cannot separate conflict-era state and military
employment absorption from differences in how unemployment was measured under
military-era governments. It is reported as a replicated pattern, not a causal
claim.

## Reproducing

```bash
pip install -r requirements.txt
python src/verify_provenance.py   # writes results/provenance_report.json
python src/train_macro_model.py   # writes results/macro_model_results.json
```

Both scripts resolve paths relative to the repository root. Committed results
were produced by exactly these commands.

## Limitations

- Pre-1991 data is unverified and probably interpolated; findings resting on it
  carry reduced confidence.
- The conflict–unemployment relationship is correlational, as above.
- Annual frequency is coarse. A quarterly series would likely explain materially
  more variance in the GDP growth model, since within-year dynamics are averaged
  away.
- No leading indicators (global commodity prices, FX reserves, IMF programme
  status) are included. Adding them is the most direct route to improving on the
  null results in Models 1 and 2.
