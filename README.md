# DAX Factor Decomposition

A 5-year factor decomposition of the German DAX index into **Market**, **Value**, **Momentum**, **Size**, and **Quality** factors, with return attribution, risk attribution, and out-of-sample validation.

## Key Findings

1. **The DAX is 91% market beta.** Once you control for general European equity exposure, the DAX has essentially no statistically significant tilt to Value, Momentum, or Quality.

2. **The only real factor bet is Size — and it's getting bigger.** The DAX's structural large-cap tilt (β = -0.23, t = -15.7) is its single most meaningful active exposure. Rolling betas show this tilt deepening from -0.20 in 2022 to -0.35 by mid-2026. With small caps down ~5%/yr over the period, that one tilt added +1.1%/yr to the index.

3. **Momentum behaved as a defensive factor.** Classical momentum is pro-cyclical. On the DAX 2021-2026, it wasn't: +0.30% on market down days, -0.13% on up days. The momentum winners of this regime — SAP, Munich Re, Rheinmetall — are defensives, not high-beta names.

## Methodology

- **Universe:** 40 DAX constituents + `^GDAXI` index, 5 years of daily data (yfinance)
- **Factors:**
  - *Market* — equal-weighted basket of all 40 names (captures directional equity beta)
  - *Value* — 1 / Price-to-Book (z-scored cross-sectionally)
  - *Momentum* — 12-1 trailing return (252-day return ending 21 days ago)
  - *Size* — negative log of time-varying market cap (price × shares outstanding)
  - *Quality* — Return on Equity (z-scored cross-sectionally)
- **Portfolio construction:** Top-minus-bottom quintile (8 longs vs 8 shorts), equal-weighted, monthly rebalanced
- **Regression:** OLS of daily DAX returns on factor returns, with Newey-West HAC standard errors (5 lags)
- **Risk attribution:** Variance decomposition via β'Σβ, with Information Ratio = return contribution / vol contribution
- **Out-of-sample validation:** 70/30 chronological split

## Results Summary

| Metric | Value |
|---|---|
| In-sample R² | 0.934 |
| Out-of-sample R² | 0.936 |
| DAX annualized return | 11.28% |
| DAX annualized vol | 17.03% |
| Market β (t-stat) | 0.998 (130) |
| Size β (t-stat) | -0.227 (-15.7) |

### Annualized Return Attribution

| Factor | Contribution |
|---|---|
| Market | +11.08% |
| Size | +1.11% |
| Momentum | +0.29% |
| Value | -0.03% |
| Quality | -0.31% |
| Alpha (α) | -0.85% |
| **Total** | **11.28%** |

### Variance Attribution

| Factor | % of total variance |
|---|---|
| Market | ~91% |
| Residual (idiosyncratic) | ~6% |
| Size | ~3% |
| Others | <1% |

## Caveats

- **Look-ahead bias on Value and Quality.** Fundamentals (P/B, ROE) are current snapshots, not point-in-time. Size uses time-varying market cap and is largely clean. Momentum is price-only and fully clean.
- **Concentrated factor portfolios.** With 40 stocks, quintile portfolios contain only 8 names per leg. Numbers are directional, not Barra-grade.
- **No transaction costs** modeled.
- **Survivorship.** The DAX 40 constituent list is current as of the run date; historical index membership changes are not back-adjusted.

## How to Run

```bash
git clone https://github.com/<your-username>/dax-factor-decomposition.git
cd dax-factor-decomposition
pip install -r requirements.txt
python dax_factor_decomposition.py
```

Output appears in `outputs/dax_factor_decomposition.png` along with regression tables and attribution stats in the terminal.

## Requirements

- Python 3.10+
- See `requirements.txt`

## Possible Extensions

- Add Fama-French European factors as a benchmark for the DIY factors
- Use point-in-time fundamentals (Refinitiv, Compustat) to eliminate look-ahead on Value/Quality
- Add a VSTOXX-based conditional regression (factor exposures in high-vol vs low-vol regimes)
- Sector controls (GICS dummies) to isolate factor effects from sector concentration
- Apply to FTSE 100, CAC 40, or EuroStoxx 50 for cross-market comparison

## License

MIT — see `LICENSE`.

## Acknowledgments

Built as a personal project to explore factor attribution methodology. Inspired by Fama-French and Barra-style risk models, adapted to a small concentrated index.
