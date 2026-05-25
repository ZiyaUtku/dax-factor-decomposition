"""
DAX Factor Decomposition
========================
A 5-year factor decomposition of the DAX index into Market, Value, Momentum,
Size, and Quality factors with return attribution, risk attribution, and
out-of-sample validation.

Features:
  - Newey-West HAC standard errors (robust to autocorrelation and heteroskedasticity)
  - Variance decomposition with Information Ratio per factor
  - Out-of-sample R² validation via 70/30 chronological split
  - Up/down market regime analysis
  - Rolling 126-day factor exposures
  - Unified 3x3 dashboard

Run:
    python dax_factor_decomposition.py

Output:
    outputs/dax_factor_decomposition.png
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime, timedelta

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. UNIVERSE
# ============================================================
DAX40 = [
    "ADS.DE","AIR.DE","ALV.DE","BAS.DE","BAYN.DE","BMW.DE","BEI.DE","CBK.DE",
    "CON.DE","1COV.DE","DTG.DE","DBK.DE","DB1.DE","DHL.DE","DTE.DE","ENR.DE",
    "EOAN.DE","FRE.DE","HNR1.DE","HEI.DE","HEN3.DE","IFX.DE","MBG.DE","MRK.DE",
    "MTX.DE","MUV2.DE","P911.DE","PAH3.DE","QIA.DE","RHM.DE","RWE.DE","SAP.DE",
    "SRT3.DE","SIE.DE","SHL.DE","SY1.DE","VOW3.DE","VNA.DE","ZAL.DE","BNR.DE"
]
INDEX = "^GDAXI"
END = datetime.today()
START = END - timedelta(days=365*5 + 60)

# ============================================================
# 2. PRICES
# ============================================================
print("Downloading prices...")
raw = yf.download(DAX40 + [INDEX], start=START, end=END,
                  auto_adjust=True, progress=False)["Close"]
raw = raw.dropna(axis=1, thresh=int(len(raw) * 0.8))
returns = raw.pct_change().dropna(how="all")
dax_ret = returns[INDEX]
stock_ret = returns.drop(columns=[INDEX])
prices = raw.drop(columns=[INDEX])
print(f"Universe: {len(stock_ret.columns)} names, {len(stock_ret)} trading days")

# ============================================================
# 3. FUNDAMENTALS + SHARES
# ============================================================
print("Downloading fundamentals & shares outstanding...")
fund = {}
for t in stock_ret.columns:
    try:
        info = yf.Ticker(t).info
        fund[t] = {
            "pb":     info.get("priceToBook", np.nan),
            "roe":    info.get("returnOnEquity", np.nan),
            "shares": info.get("sharesOutstanding", np.nan),
        }
    except Exception:
        fund[t] = {"pb": np.nan, "roe": np.nan, "shares": np.nan}
fund = pd.DataFrame(fund).T.astype(float)

# ============================================================
# 4. FACTOR CONSTRUCTION
# ============================================================
def zscore_rows(df):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

# Market: equal-weighted basket (directional equity beta)
market_ret = stock_ret.mean(axis=1)

# Size: time-varying market cap (largely clean)
mcap_ts = prices.mul(fund["shares"], axis=1)
size_panel = zscore_rows((-np.log(mcap_ts)).resample("ME").last())

# Momentum: 12-1 (price-only, fully clean)
mom_panel = zscore_rows(
    prices.shift(21).pct_change(231).resample("ME").last().dropna(how="all")
)

# Value: 1 / P/B (snapshot — look-ahead bias acknowledged)
month_ends = size_panel.index
value_static = zscore_rows(pd.DataFrame([1 / fund["pb"]]))
value_panel = pd.DataFrame(np.tile(value_static.values, (len(month_ends), 1)),
                           index=month_ends, columns=value_static.columns)

# Quality: ROE (snapshot — look-ahead bias acknowledged)
quality_static = zscore_rows(pd.DataFrame([fund["roe"]]))
quality_panel = pd.DataFrame(np.tile(quality_static.values, (len(month_ends), 1)),
                             index=month_ends, columns=quality_static.columns)

# ============================================================
# 5. LONG-SHORT PORTFOLIOS
# ============================================================
def long_short(score_panel, stock_ret, q=0.2):
    """Top-minus-bottom quintile, equal-weighted, monthly rebalanced."""
    score_panel = score_panel.reindex(columns=stock_ret.columns)
    longs  = score_panel.rank(axis=1, pct=True) >= (1 - q)
    shorts = score_panel.rank(axis=1, pct=True) <= q
    w_long  = longs.astype(float).div(longs.sum(axis=1).replace(0, np.nan), axis=0)
    w_short = shorts.astype(float).div(shorts.sum(axis=1).replace(0, np.nan), axis=0)
    w = (w_long - w_short).shift(1).reindex(stock_ret.index).ffill()
    return (w * stock_ret).sum(axis=1)

factor_ret = pd.DataFrame({
    "Market":   market_ret,
    "Value":    long_short(value_panel,   stock_ret),
    "Momentum": long_short(mom_panel,     stock_ret),
    "Size":     long_short(size_panel,    stock_ret),
    "Quality":  long_short(quality_panel, stock_ret),
}).dropna()

# Restrict to last 5 years and align
cutoff = END - timedelta(days=365 * 5)
factor_ret = factor_ret.loc[factor_ret.index >= cutoff]
dax_ret = dax_ret.loc[factor_ret.index].dropna()
factor_ret = factor_ret.loc[dax_ret.index]

# ============================================================
# 6. PERFORMANCE TABLE
# ============================================================
def perf(r):
    ann = r.mean() * 252
    vol = r.std() * np.sqrt(252)
    sr  = ann / vol if vol else np.nan
    cum = (1 + r).cumprod()
    dd  = (cum / cum.cummax() - 1).min()
    return pd.Series({"Ann.Ret": ann, "Ann.Vol": vol, "Sharpe": sr, "MaxDD": dd})

perf_table = factor_ret.apply(perf).T
print("\n=== Factor Performance ===")
print(perf_table.round(3))

# ============================================================
# 7. RETURN ATTRIBUTION (Newey-West HAC)
# ============================================================
X = sm.add_constant(factor_ret)
model = sm.OLS(dax_ret, X, missing="drop").fit(
    cov_type="HAC", cov_kwds={"maxlags": 5}
)
print("\n=== DAX Factor Decomposition (Newey-West SEs) ===")
print(model.summary())

betas = model.params.drop("const")
ann_factor_ret = factor_ret.mean() * 252
ret_contrib = betas * ann_factor_ret
ann_alpha = model.params["const"] * 252
ann_dax = dax_ret.mean() * 252

# ============================================================
# 8. RISK ATTRIBUTION
# ============================================================
Sigma = factor_ret.cov() * 252
betas_v = betas.values
sigma_beta = Sigma.values @ betas_v
total_factor_var = betas_v @ sigma_beta
residual_var = model.mse_resid * 252
total_var = total_factor_var + residual_var
total_vol = np.sqrt(total_var)

var_contrib = pd.Series(betas_v * sigma_beta, index=betas.index)
var_contrib_pct = var_contrib / total_var * 100
residual_pct = residual_var / total_var * 100
vol_contrib = np.sign(var_contrib) * np.sqrt(np.abs(var_contrib))
info_ratio = ret_contrib / vol_contrib.replace(0, np.nan)

risk_table = pd.DataFrame({
    "Return Contrib (%/yr)":    (ret_contrib * 100).round(2),
    "Vol Contrib (%/yr)":       (vol_contrib * 100).round(2),
    "Var Contrib (% of total)": var_contrib_pct.round(2),
    "Info Ratio":               info_ratio.round(2),
})
risk_table.loc["Residual"] = [np.nan, np.sqrt(residual_var)*100, residual_pct, np.nan]
print("\n=== Risk-Adjusted Factor Attribution ===")
print(risk_table)
print(f"\nDAX annualized return:        {ann_dax*100:.2f}%")
print(f"DAX annualized vol:           {total_vol*100:.2f}%")
print(f"Explained (factor) vol:       {np.sqrt(max(total_factor_var,0))*100:.2f}%")
print(f"Residual (idiosyncratic) vol: {np.sqrt(residual_var)*100:.2f}%")

# ============================================================
# 9. OUT-OF-SAMPLE VALIDATION
# ============================================================
split = int(len(dax_ret) * 0.7)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = dax_ret.iloc[:split], dax_ret.iloc[split:]

model_train = sm.OLS(y_train, X_train).fit()
y_pred = model_train.predict(X_test)
ss_res = ((y_test - y_pred) ** 2).sum()
ss_tot = ((y_test - y_test.mean()) ** 2).sum()
oos_r2 = 1 - ss_res / ss_tot

print(f"\n=== Out-of-Sample Validation ===")
print(f"In-sample R² (full):     {model.rsquared:.3f}")
print(f"In-sample R² (train):    {model_train.rsquared:.3f}")
print(f"Out-of-sample R² (test): {oos_r2:.3f}")
print(f"→ Gap: {(model_train.rsquared - oos_r2)*100:.1f} pp "
      f"({'CLEAN — no overfit' if abs(model_train.rsquared-oos_r2) < 0.05 else 'check stability'})")

# ============================================================
# 10. UP / DOWN MARKET REGIME — daily basis
# ============================================================
mkt = factor_ret["Market"]
up_mask = mkt > 0
up_ret_daily   = factor_ret[up_mask].mean() * 100
down_ret_daily = factor_ret[~up_mask].mean() * 100
regime_df = pd.DataFrame({
    "Up Days (avg daily %)":   up_ret_daily,
    "Down Days (avg daily %)": down_ret_daily,
})
print("\n=== Factor Returns by Market Regime (avg daily %) ===")
print(regime_df.round(3))

# ============================================================
# 11. ROLLING FACTOR EXPOSURES
# ============================================================
window = 126
warmup = 60
rolling_betas, idx = [], []
for i in range(window + warmup, len(dax_ret)):
    Xw = sm.add_constant(factor_ret.iloc[i-window:i])
    try:
        m = sm.OLS(dax_ret.iloc[i-window:i], Xw).fit()
        rolling_betas.append(m.params.drop("const").values)
        idx.append(dax_ret.index[i])
    except Exception:
        pass
rolling_betas = pd.DataFrame(rolling_betas, index=idx, columns=factor_ret.columns)

# ============================================================
# 12. UNIFIED DASHBOARD — 3x3
# ============================================================
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5,
                     "axes.titleweight": "bold"})
fig, axes = plt.subplots(3, 3, figsize=(19, 14))
colors = {"Market":"#7f7f7f","Value":"#1f77b4","Momentum":"#ff7f0e",
          "Size":"#2ca02c","Quality":"#9467bd"}

# Row 1
(1+factor_ret.drop(columns=["Market"])).cumprod().plot(
    ax=axes[0,0], lw=1.8,
    color=[colors[c] for c in ["Value","Momentum","Size","Quality"]])
axes[0,0].axhline(1, color="black", lw=0.5)
axes[0,0].set_title("L/S Factor Portfolios — Cumulative Return")
axes[0,0].set_ylabel("Growth of €1")

sns.heatmap(factor_ret.corr(), annot=True, cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, ax=axes[0,1], fmt=".2f", cbar_kws={"shrink": 0.7})
axes[0,1].set_title("Factor Return Correlations")

ci = model.conf_int().drop("const"); ci.columns = ["lo","hi"]
err = np.array([betas - ci["lo"], ci["hi"] - betas])
axes[0,2].bar(betas.index, betas.values, yerr=err,
              color=[colors[c] for c in betas.index],
              capsize=5, edgecolor="black")
axes[0,2].axhline(0, color="black", lw=0.5)
axes[0,2].set_title("DAX Factor Loadings (β) with HAC 95% CI")

# Row 2
rolling_betas.drop(columns=["Market"]).plot(
    ax=axes[1,0], lw=1.5,
    color=[colors[c] for c in ["Value","Momentum","Size","Quality"]])
axes[1,0].axhline(0, color="black", lw=0.5)
axes[1,0].set_ylim(-0.5, 0.3)
axes[1,0].set_title("Rolling 126-day Factor β (excl. Market)")
axes[1,0].set_ylabel("β")

contrib_daily = factor_ret.mul(betas, axis=1)
(1+contrib_daily).cumprod().plot(ax=axes[1,1], lw=1.6,
    color=[colors[c] for c in contrib_daily.columns])
(1+dax_ret).cumprod().plot(ax=axes[1,1], color="black", lw=2.2,
                            linestyle="--", label="DAX (actual)")
axes[1,1].legend(loc="upper left", fontsize=8)
axes[1,1].set_title("Cumulative Factor Contribution to DAX")

attr_plot = ret_contrib.copy(); attr_plot["Alpha"] = ann_alpha
attr_plot = attr_plot.sort_values()
axes[1,2].barh(attr_plot.index, attr_plot.values * 100,
               color=["#d62728" if v<0 else "#2ca02c" for v in attr_plot.values],
               edgecolor="black")
axes[1,2].axvline(0, color="black", lw=0.5)
axes[1,2].set_title(f"Return Attribution (DAX: {ann_dax*100:.1f}%/yr)")
axes[1,2].set_xlabel("Contribution (%/yr)")

# Row 3
var_plot = var_contrib_pct.copy(); var_plot["Residual"] = residual_pct
var_plot = var_plot.sort_values()
axes[2,0].barh(var_plot.index, var_plot.values,
               color=["#d62728" if v<0 else "#1f77b4" for v in var_plot.values],
               edgecolor="black")
axes[2,0].axvline(0, color="black", lw=0.5)
axes[2,0].set_title(f"Variance Attribution (DAX vol: {total_vol*100:.1f}%/yr)")
axes[2,0].set_xlabel("% of total variance")

ir_plot = info_ratio.dropna().sort_values()
axes[2,1].barh(ir_plot.index, ir_plot.values,
               color=["#d62728" if v<0 else "#2ca02c" for v in ir_plot.values],
               edgecolor="black")
axes[2,1].axvline(0, color="black", lw=0.5)
axes[2,1].set_title("Information Ratio per Factor\n(Return Contrib / Vol Contrib)")
axes[2,1].set_xlabel("IR")

x_pos = np.arange(len(regime_df))
w = 0.38
axes[2,2].bar(x_pos - w/2, regime_df["Up Days (avg daily %)"], w,
              color="#2ca02c", edgecolor="black", label="Up days")
axes[2,2].bar(x_pos + w/2, regime_df["Down Days (avg daily %)"], w,
              color="#d62728", edgecolor="black", label="Down days")
axes[2,2].set_xticks(x_pos); axes[2,2].set_xticklabels(regime_df.index, rotation=30)
axes[2,2].axhline(0, color="black", lw=0.5)
axes[2,2].legend(fontsize=8)
axes[2,2].set_title("Avg Daily Factor Return by Market Regime (%)")

plt.suptitle(f"DAX Factor Decomposition — 5Y | In-sample R²={model.rsquared:.3f}, "
             f"OOS R²={oos_r2:.3f}",
             fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "dax_factor_decomposition.png")
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"\n✓ Saved: {output_path}")
