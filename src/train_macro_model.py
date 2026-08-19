"""
Pakistan macroeconomic relationship model: economic indicators -> GDP ->
exchange rate & unemployment.

DATA QUALITY NOTE (important, read before trusting results):
pakistan_economic_data_1947_2025.csv's 1991-2025 portion matches an
independent World Bank source almost exactly (inflation identical to many
decimals; GDP growth/unemployment correlate 0.93-0.96). Its GDP LEVEL values
match an independent source almost perfectly from 2016 onward (corr=0.996),
diverge somewhat 2007-2015 (runs ~10-20% higher), and have NO independent
verification before 1991 - combined with implausible decimal precision for
1947 (a year with no real recorded unemployment/inflation statistics), the
pre-1991 portion is treated here as likely synthetic/interpolated, not
verified history. Models are run on the FULL range but also validated on
the 1991+ "verified-reliable" window as a robustness check.

Small sample size (79 annual rows, 34 in the reliable window) means
tree-based ML is inappropriate here (would overfit badly). This uses
Ridge regression (regularized linear) with leave-one-out cross-validation,
the right-sized tool for small-N annual macro data.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

main = pd.read_csv(DATA_DIR / 'pakistan_economic_data_1947_2025.csv')
fx = pd.read_csv(DATA_DIR / 'rupee_vs_dollar.csv').rename(columns={'year': 'Year', 'pk(Rs)': 'PKR_per_USD'})

df = main.merge(fx[['Year', 'PKR_per_USD']], on='Year', how='left')
df = df.sort_values('Year').reset_index(drop=True)
df['War_Conflict_Indicator'] = df['War_Conflict_Indicator'].fillna(0)
df['log_fx'] = np.log(df['PKR_per_USD'])
df['time_idx'] = df['Year'] - df['Year'].min()

df.to_csv(RESULTS_DIR / 'consolidated_macro_data.csv', index=False)
print(f"Consolidated dataset: {df.shape}")
print(f"Years: {df['Year'].min()}-{df['Year'].max()}, {df['PKR_per_USD'].isna().sum()} missing FX rows")

RELIABLE_CUTOFF = 1991

def loocv_ridge(X, y, alpha=1.0):
    """Leave-one-out CV, the right validation for ~30-80 row datasets.

    Scaling is done INSIDE the CV loop via a Pipeline. Fitting a StandardScaler
    on the full matrix before cross-validation leaks the held-out row's mean and
    variance into its own prediction. With n=79 the effect here is under 0.002
    R2, but Ridge is scale-sensitive and the correct construction costs nothing.
    """
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    preds = cross_val_predict(pipe, X, y, cv=LeaveOneOut())
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    pipe.fit(X, y)  # final fit on all data, for reporting coefficients only
    coefs = dict(zip(X.columns, pipe.named_steps["ridge"].coef_))
    return {"r2_loocv": r2, "mae_loocv": mae, "coefficients": coefs}, preds

results = {}

# =========================================================
# MODEL 1: What drives GDP growth?
# =========================================================
print("\n=== MODEL 1: GDP Growth Rate drivers ===")
feat1 = ['FDI_Billion_USD', 'Inflation_Rate', 'Trade_Balance_Billion_USD',
          'Military_Expenditure_Percent_GDP', 'Agriculture_Share',
          'Industry_Share', 'War_Conflict_Indicator']
d1 = df.dropna(subset=feat1 + ['GDP_Growth_Rate'])
res1_full, preds1_full = loocv_ridge(d1[feat1], d1['GDP_Growth_Rate'])
print(f"Full range (n={len(d1)}): LOOCV R2={res1_full['r2_loocv']:.3f}, MAE={res1_full['mae_loocv']:.2f}")
print("Coefficients (standardized):", {k: round(v,3) for k,v in res1_full['coefficients'].items()})

d1_rel = d1[d1['Year'] >= RELIABLE_CUTOFF]
res1_rel, _ = loocv_ridge(d1_rel[feat1], d1_rel['GDP_Growth_Rate'])
print(f"Reliable window only (n={len(d1_rel)}): LOOCV R2={res1_rel['r2_loocv']:.3f}, MAE={res1_rel['mae_loocv']:.2f}")
results['gdp_growth_model'] = {"full_range": res1_full, "reliable_window": res1_rel, "n_full": len(d1), "n_reliable": len(d1_rel)}

# =========================================================
# MODEL 2: Exchange rate (log PKR/USD) - trend + residual hybrid
# =========================================================
print("\n=== MODEL 2: Exchange rate (PKR/USD) ===")
d2 = df.dropna(subset=['log_fx', 'GDP_Growth_Rate', 'Inflation_Rate', 'Trade_Balance_Billion_USD'])

# Trend component: log(FX) vs time (captures multi-decade depreciation)
trend_lr = LinearRegression()
trend_lr.fit(d2[['time_idx']], d2['log_fx'])
d2 = d2.copy()
d2['fx_trend_pred'] = trend_lr.predict(d2[['time_idx']])
d2['fx_residual'] = d2['log_fx'] - d2['fx_trend_pred']
annual_depreciation_pct = (np.exp(trend_lr.coef_[0]) - 1) * 100
print(f"Trend: currency depreciates {annual_depreciation_pct:.2f}%/year on average (full range)")

feat2 = ['GDP_Growth_Rate', 'Inflation_Rate', 'Trade_Balance_Billion_USD', 'FDI_Billion_USD']
res2, preds2 = loocv_ridge(d2[feat2], d2['fx_residual'])
print(f"Residual model (n={len(d2)}): LOOCV R2={res2['r2_loocv']:.3f} (explains deviations FROM the smooth trend)")
print("Coefficients (standardized):", {k: round(v,3) for k,v in res2['coefficients'].items()})
results['exchange_rate_model'] = {"annual_depreciation_pct": annual_depreciation_pct,
                                    "residual_model": res2, "n": len(d2)}

# =========================================================
# MODEL 3: Unemployment - testing Okun's Law (GDP growth <-> unemployment)
# =========================================================
print("\n=== MODEL 3: Unemployment Rate (Okun's Law test) ===")
feat3 = ['GDP_Growth_Rate', 'Inflation_Rate', 'Military_Expenditure_Percent_GDP', 'War_Conflict_Indicator']
d3 = df.dropna(subset=feat3 + ['Unemployment_Rate'])
res3_full, _ = loocv_ridge(d3[feat3], d3['Unemployment_Rate'])
print(f"Full range (n={len(d3)}): LOOCV R2={res3_full['r2_loocv']:.3f}, MAE={res3_full['mae_loocv']:.2f}")
print("Coefficients (standardized):", {k: round(v,3) for k,v in res3_full['coefficients'].items()})

d3_rel = d3[d3['Year'] >= RELIABLE_CUTOFF]
res3_rel, _ = loocv_ridge(d3_rel[feat3], d3_rel['Unemployment_Rate'])
print(f"Reliable window only (n={len(d3_rel)}): LOOCV R2={res3_rel['r2_loocv']:.3f}, MAE={res3_rel['mae_loocv']:.2f}")

okun_corr_full = d3['GDP_Growth_Rate'].corr(d3['Unemployment_Rate'])
okun_corr_rel = d3_rel['GDP_Growth_Rate'].corr(d3_rel['Unemployment_Rate'])
print(f"Simple correlation GDP growth vs Unemployment: full={okun_corr_full:.3f}, reliable-window={okun_corr_rel:.3f}")
print("(Okun's Law predicts a NEGATIVE correlation - higher growth, lower unemployment)")

# Conflict-year comparison reported in the write-up - computed here so the
# repository substantiates it rather than the prose asserting it.
_conf = d3.groupby('War_Conflict_Indicator')['Unemployment_Rate'].agg(['mean', 'count'])
conflict_stats = {
    "conflict_years_mean_unemployment": float(_conf.loc[1, 'mean']),
    "non_conflict_years_mean_unemployment": float(_conf.loc[0, 'mean']),
    "n_conflict_years": int(_conf.loc[1, 'count']),
    "n_non_conflict_years": int(_conf.loc[0, 'count']),
}
print(f"Conflict years: mean unemployment {conflict_stats['conflict_years_mean_unemployment']:.2f}% "
      f"(n={conflict_stats['n_conflict_years']}) vs non-conflict "
      f"{conflict_stats['non_conflict_years_mean_unemployment']:.2f}% "
      f"(n={conflict_stats['n_non_conflict_years']})")
print("(Correlational only - this data cannot separate state/military employment")
print(" absorption from measurement differences under military-era governments.)")

results['unemployment_model'] = {"full_range": res3_full, "reliable_window": res3_rel,
                                   "conflict_year_comparison": conflict_stats,
                                   "okun_correlation_full": okun_corr_full, "okun_correlation_reliable": okun_corr_rel,
                                   "n_full": len(d3), "n_reliable": len(d3_rel)}

with open(RESULTS_DIR / 'macro_model_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\nSaved consolidated data and results.")
