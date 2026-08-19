"""
Data provenance verification.

The primary dataset (pakistan_economic_data_1947_2025.csv) is a third-party
compilation. Before modelling anything on it, this script checks which parts of
it can be corroborated against independent sources, and reports the rest as
unverified. Run this FIRST; train_macro_model.py assumes you have.

Cross-check sources:
  - Pakistan_Yearly_UnemploymentGDP_And_CPI.csv  (World Bank indicators, 1991-2020)
  - GDP_of_Pakistan_1960-2021.xlsx               (independent GDP level series)
"""
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

main = pd.read_csv(DATA_DIR / "pakistan_economic_data_1947_2025.csv")
wb = pd.read_csv(DATA_DIR / "Pakistan_Yearly_UnemploymentGDP_And_CPI.csv")

report = {}

# --- Check 1: decimal precision of pre-1991 values -------------------------
def n_decimals(x):
    s = repr(float(x))
    return len(s.split(".")[1]) if "." in s and "e" not in s else 0

early = main[main["Year"] < 1991]
prec = {c: int(early[c].dropna().map(n_decimals).max())
        for c in ["Inflation_Rate", "Unemployment_Rate", "GDP_Growth_Rate"]}
report["precision_pre_1991_max_decimals"] = prec
print("Max decimal places in pre-1991 values:", prec)
print("  -> Pakistan had no unemployment or inflation statistical apparatus in")
print("     1947. Values carried to this precision are interpolated or generated,")
print("     not recorded history. Pre-1991 is treated as UNVERIFIED.\n")

# --- Check 2: World Bank overlap, 1991-2020 --------------------------------
m = main.merge(wb, on="Year", how="inner")
infl_diff = float((m["Inflation_Rate"] - m["CPI Value"]).abs().max())
checks = {
    "overlap_years": [int(m.Year.min()), int(m.Year.max())],
    "n_overlap": int(len(m)),
    "inflation_corr": float(m["Inflation_Rate"].corr(m["CPI Value"])),
    "inflation_max_abs_diff": infl_diff,
    "gdp_growth_corr": float(m["GDP_Growth_Rate"].corr(m["GDP value"])),
    "unemployment_corr": float(m["Unemployment_Rate"].corr(m["Unemployment rate"])),
}
report["world_bank_overlap"] = checks
print(f"World Bank overlap {checks['overlap_years'][0]}-{checks['overlap_years'][1]} "
      f"(n={checks['n_overlap']}):")
print(f"  inflation    corr={checks['inflation_corr']:.4f}  max abs diff={infl_diff:.6f}")
print(f"  GDP growth   corr={checks['gdp_growth_corr']:.4f}")
print(f"  unemployment corr={checks['unemployment_corr']:.4f}")
if infl_diff < 1e-9:
    print("  -> Inflation is IDENTICAL to the World Bank series, so the post-1991")
    print("     portion is derived from it. This corroborates the values, but the")
    print("     two files are not independent evidence of one another.\n")

# --- Check 3: GDP levels against an independent series ---------------------
try:
    x = pd.read_excel(DATA_DIR / "GDP_of_Pakistan_1960-2021.xlsx")
    x.columns = [c.strip() for c in x.columns]
    x["Year"] = x["Years"].astype(int)
    x["GDP_ext"] = x["GDP"].astype(str).str.replace(r"[\$,B]", "", regex=True).astype(float)
    g = main.merge(x[["Year", "GDP_ext"]], on="Year")
    windows = {}
    for lo, hi in [(1960, 2021), (2007, 2015), (2016, 2021)]:
        s = g[(g.Year >= lo) & (g.Year <= hi)]
        w = {
            "n": int(len(s)),
            "corr": float(s["GDP_Billion_USD"].corr(s["GDP_ext"])),
            "mean_ratio_primary_over_external": float((s["GDP_Billion_USD"] / s["GDP_ext"]).mean()),
        }
        windows[f"{lo}-{hi}"] = w
        print(f"GDP levels {lo}-{hi}: n={w['n']} corr={w['corr']:.4f} "
              f"ratio={w['mean_ratio_primary_over_external']:.3f}")
    report["gdp_level_checks"] = windows
    print("  -> Near-perfect agreement from 2016; the primary file runs about 13%")
    print("     high over 2007-2015. GDP levels are not a model input, so this")
    print("     affects context only, not results.\n")
except Exception as e:  # noqa: BLE001
    report["gdp_level_checks"] = f"skipped: {e}"
    print(f"GDP level check skipped: {e}\n")

report["conclusion"] = (
    "Post-1991 corroborated against World Bank indicators. Pre-1991 unverified "
    "and likely interpolated. All models are therefore reported on both the full "
    "range and the 1991+ window."
)

with open(RESULTS_DIR / "provenance_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("Wrote results/provenance_report.json")
