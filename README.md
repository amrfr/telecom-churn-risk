# Telecom Customer Churn & Revenue-at-Risk Analysis

A business-analyst-first take on the classic churn problem: instead of stopping at "model accuracy," this quantifies how much monthly recurring revenue is actually at risk, in which segments, how confident we should be in each number, and whether a retention campaign is worth funding.

**v2** adds statistical rigor v1 didn't have: 5-fold cross-validation instead of a single train/test split, a second independent model (Naive Bayes) as a cross-check, permutation-test significance and bootstrap confidence intervals on every segment churn rate, a Kaplan-Meier survival curve driving a real customer-lifetime-value estimate, a campaign ROI sensitivity grid with a solved breakeven rate, and a benchmark against published telecom-industry churn figures. See `METHODOLOGY.md` for full technical detail and `SOURCES.md` for citations.

## Executive summary

**Overall churn rate: 25.3%** (123 of 487 customers), consuming **$8,948/month** against a **$32,196/month** revenue book. That's above both the ~22.5%/yr blended industry midpoint and well above best-in-class US postpaid carriers (T-Mobile 0.90%/mo, AT&T 0.87%/mo) - a sign this sample skews toward higher-churn segments, not a representative national book (see `figures/industry_benchmark_comparison.png`).

**Contract type is the strongest, most certain churn driver.** Month-to-month customers churn at 42.0% (95% bootstrap CI: 36.2%-47.8%) versus 4.1% for one-year and 2.7% for two-year contracts - the confidence intervals don't overlap, and it's the single largest chi-square statistic of 10 drivers tested (94.98, permutation p<0.001). This is the highest-confidence lever in the whole analysis.

**All 10 tested drivers are statistically real, not noise** - contract type, tenure, tech support, online security, payment method, internet type, dependents, paperless billing, partner status, and senior-citizen status all clear p<0.05 on a 5,000-permutation test. But statistical significance isn't the same as size: look at the actual rate gaps and their confidence intervals (`bootstrap_churn_rate_cis.csv`) before prioritizing action, not just the p-value ranking.

**Two independent models agree on the core drivers.** A from-scratch logistic regression (83.7% holdout accuracy, 0.849 AUC) and a from-scratch Naive Bayes model (82.8% AUC) were trained independently; 4 of each model's top-8 churn-raising features overlap (fiber internet, monthly bill size, paperless billing, electronic check payment), which is real cross-method evidence those aren't artifacts of one model's assumptions. Five-fold cross-validation puts the logistic regression's AUC at 0.83 +/- 0.065 across folds - a meaningful spread at n=487, reported honestly rather than hidden behind one lucky split.

**Customer lifetime value, estimated properly.** A Kaplan-Meier survival curve (event = churned, censored = still active) shows more than 60% of customers are still retained at 72 months - median lifetime isn't reached within the observed window, so CLV uses a restricted-mean estimate rather than an extrapolated guess. The active book carries an estimated **$615,218 in remaining customer lifetime value**, with the highest-risk decile alone accounting for **$139,674** of that.

**The retention campaign has a lot of room to fail and still pay off.** At $30/customer targeting the top 2 risk deciles, the campaign only needs a **3.4% success rate to break even** - it needs to fail at saving 29 of every 30 targeted customers before it stops being worth funding. Even the original conservative 25%-success assumption pencils out to a **6.4x ROI multiple**. A full cost x success-rate sensitivity grid (`figures/campaign_roi_sensitivity.png`) shows this conclusion is robust across a wide range of assumptions, not dependent on one optimistic guess - and it's consistent with the widely cited industry rule that acquiring a customer costs 5-10x more than retaining one.

## Repo structure

```
telecom-churn-risk/
├── data/
│   ├── raw/                              # source dataset + industry benchmarks (see SOURCES.md)
│   └── processed/                        # generated: EDA tables, model artifacts, stats, findings
├── figures/                              # generated charts
├── src/
│   ├── 01_churn_model.py                 # EDA + logistic regression (CV) + Naive Bayes cross-check
│   ├── 02_statistical_tests.py           # chi-square/permutation significance + bootstrap CIs
│   ├── 03_revenue_at_risk.py             # scores active customers, survival curve, CLV, campaign ROI
│   └── 04_industry_benchmark.py          # compares this dataset to published churn benchmarks
├── METHODOLOGY.md                        # full methodology & limitations
├── SOURCES.md                            # complete source list
├── requirements.txt
└── README.md
```

## Reproduce it

```bash
pip install -r requirements.txt
python src/01_churn_model.py
python src/02_statistical_tests.py
python src/03_revenue_at_risk.py
python src/04_industry_benchmark.py
```

Everything in `data/processed/` and `figures/` is regenerated from `data/raw/` - nothing downstream is hand-edited. Verified end-to-end from a clean run before packaging.

## Data source

[IBM Telco Customer Churn dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) - see `SOURCES.md` for the full citation and `METHODOLOGY.md` for why this repo uses a 487-row slice of the canonical 7,043-row dataset (a genuine build-environment network constraint, tested against multiple mirrors, not a curation choice).

## Methodology & limitations (read before citing these numbers)

Full detail in `METHODOLOGY.md`. In short: every model and statistical test here is implemented from scratch in numpy/pandas (no scikit-learn/scipy/lifelines) - logistic regression, Naive Bayes, chi-square + permutation p-values, bootstrap confidence intervals, Kaplan-Meier survival, and rank-based ROC/AUC. The main real limitation is sample size (487 of 7,043 canonical rows), which is why this version leans on cross-validation, bootstrap CIs, and a second cross-check model rather than reporting single point estimates as if they were precise.

## Key charts

- `figures/churn_by_contract.png`, `figures/churn_by_tenure.png` - where churn concentrates
- `figures/bootstrap_confidence_intervals.png` - segment churn rates with real uncertainty bands
- `figures/chi_square_significance.png` - which drivers are statistically real vs. noise
- `figures/cross_validation_spread.png` - model performance across 5 folds, not one split
- `figures/model_agreement_comparison.png` - logistic regression vs. Naive Bayes, side by side
- `figures/survival_curve.png` - Kaplan-Meier customer retention curve
- `figures/clv_by_decile.png` - estimated remaining customer lifetime value at risk, by decile
- `figures/campaign_roi_sensitivity.png` - retention campaign ROI across cost/success-rate assumptions
- `figures/industry_benchmark_comparison.png` - this dataset vs. published telecom churn benchmarks
- `figures/roc_curve.png` - holdout ROC curve (AUC 0.85)

## License

MIT - see `LICENSE`.
