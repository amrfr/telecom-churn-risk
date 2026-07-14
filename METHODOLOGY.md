# Methodology & limitations

Read this before citing any number in this repo. Every design choice below exists to be defensible, not impressive.

## Dataset

487 customer records from the [IBM Telco Customer Churn dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) - 21 features covering demographics, account type, services subscribed, billing, and a churn label. The canonical dataset has 7,043 rows; this repo uses a 487-row prefix because this build environment's web-fetch tooling truncates large downloads (confirmed by testing multiple mirror repositories and CDN paths - all hit the same ceiling regardless of source). This is a genuine environment constraint, not a curation choice: the 487 rows are a contiguous, unfiltered prefix of the source file. The observed 25.3% churn rate is close to the canonical dataset's published ~26.5%, which is a reasonable sanity check that this slice isn't badly skewed, but every downstream number should be read as directional at this sample size, not a production estimate. See "What would make this materially better" below.

## No scikit-learn, no scipy, no lifelines

Every model, statistical test, and confidence interval in this repo is implemented from scratch in numpy/pandas:

- **Logistic regression**: batch gradient descent with L2 regularization (`src/01_churn_model.py`).
- **Gaussian Naive Bayes**: closed-form class-conditional means/variances, used as an independent second model (`src/01_churn_model.py`).
- **Chi-square test statistic**: standard Pearson formula, computed manually per categorical driver (`src/02_statistical_tests.py`).
- **Significance (p-values)**: via a 5,000-iteration permutation test (shuffle the churn label, recompute chi-square, see how often the real statistic is beaten), not the chi-square CDF. This sidesteps implementing an incomplete-gamma-function and makes no distributional assumption beyond exchangeability under the null - a defensible tradeoff at n=487, where asymptotic approximations are shakier anyway.
- **Confidence intervals**: 5,000-resample bootstrap rather than the normal (Wald) approximation, which misbehaves when a segment's churn rate is near 0% or 100% (several segments here have single-digit churn rates - see `bootstrap_churn_rate_cis.csv`).
- **Kaplan-Meier survival curve**: standard product-limit estimator, hand-coded over event/censoring times (`src/03_revenue_at_risk.py`).
- **ROC/AUC**: rank-based (Mann-Whitney U statistic), not a library call.

This was originally a build-environment constraint (no package index access to scikit-learn), but the byproduct is that every number in this repo is traceable to explicit, readable code rather than library internals.

## Why cross-validation and a second model, not just one train/test split

At n=487 (390 train / 97 test in the original 80/20 split), a single holdout accuracy or AUC is one draw from a noisy distribution - a slightly different random split could move the headline number meaningfully. Two additions address this:

1. **5-fold stratified cross-validation** reports accuracy/precision/recall/AUC as a mean +/- standard deviation across 5 different splits, not one. The spread (`cross_validation_folds.csv`, `figures/cross_validation_spread.png`) is itself information: a wide spread means "don't trust the single-split number too precisely."
2. **A from-scratch Gaussian Naive Bayes model**, trained and evaluated independently of the logistic regression. Where the two models' top churn-raising features overlap (4 of the top 8 - see `findings_model.json` -> `model_agreement`), that's real cross-method evidence those features matter, not an artifact of one model's assumptions. Where they diverge, treat the logistic regression's ranking as primary (it directly estimates each feature's independent effect controlling for the others; Naive Bayes does not) but note the disagreement rather than hiding it.

## Statistical significance vs. practical significance

All 10 categorical drivers tested come back significant at p<0.05 (`chi_square_significance_tests.csv`) - unsurprising at n=487 with fairly large effect sizes on several of them (Contract, TenureBucket, TechSupport, OnlineSecurity all have large chi-square statistics, not borderline ones). Significance here answers "is this association distinguishable from sampling noise," not "is this association large enough to act on." For the latter, look at the actual churn-rate gaps and their bootstrap confidence intervals (`bootstrap_churn_rate_cis.csv`) - e.g., month-to-month vs. two-year contract churn rates have non-overlapping 95% CIs, which is stronger practical evidence than the p-value alone.

## Kaplan-Meier survival curve and customer lifetime value (CLV)

This dataset is a legitimate (if single-snapshot, cross-sectional) survival-analysis setup: for churned customers, `tenure` is their complete relationship length (an event). For still-active customers, `tenure` is a lower bound on their eventual relationship length (right-censored). The Kaplan-Meier estimator handles both correctly without needing to guess what happens to censored customers after the snapshot date.

At this dataset's overall churn rate, the survival curve never crosses 50% within the observed 0-72 month window (63% of customers are still estimated to be "retained" at 72 months) - so "median customer lifetime" is not reached in this window, and CLV estimates use a **restricted mean residual life** calculation instead (integrate the survival curve forward from a customer's current tenure to the 72-month observed horizon, without extrapolating past it). This is a conservative choice: true CLV is understated for customers who would realistically survive past 72 months, but extrapolating a KM curve past its data window is a well-known way to get overconfident numbers.

## Revenue-at-risk and campaign ROI

`active_customers_scored.csv` ranks currently-active customers by predicted churn probability and reports both a 12-month annualized-revenue view (v1's approach) and a survival-based estimated CLV (v2's addition, `estimated_clv_usd`). Neither is a forecast of which specific customers leave next quarter - both are risk-weighted rankings.

The retention campaign section replaces v1's single illustrative ROI point estimate with a **sensitivity grid** across 5 cost assumptions ($10-$50/customer) x 7 success-rate assumptions (10%-50%) (`campaign_roi_sensitivity_grid.csv`, `figures/campaign_roi_sensitivity.png`), plus the **breakeven success rate** solved directly from the algebra at each cost level (`campaign_breakeven_success_rates.csv`) - the minimum campaign win-rate needed for ROI=0. This is more useful to a decision-maker than a single "6.4x ROI" headline, because it shows how much cushion that number has (at $30/customer, breakeven is 3.4% - meaning the campaign would need to fail at retaining 29 out of every 30 targeted customers before it stopped being worth funding).

## Industry benchmarking

`src/04_industry_benchmark.py` compares this dataset's 25.3% annualized churn rate against published, cited telecom-industry figures (`data/raw/industry_benchmarks.csv`, full citations in `SOURCES.md`). The honest reading: this sample's churn rate sits above both the blended industry midpoint (~22.5%/yr) and well above best-in-class US postpaid carriers (T-Mobile/AT&T, ~10%/yr) - consistent with a sample skewed toward higher-churn segments (month-to-month contracts, fiber internet), not a representative national postpaid book. That does not undermine the within-dataset findings about *which* segments churn more - those directional patterns (month-to-month >> annual contracts; fiber >> DSL) are well documented industry-wide, independent of this sample's overall level.

## What would make this materially better

In rough priority order: (1) the full 7,043-row dataset, which would tighten every confidence interval and cross-validation spread substantially and is the single biggest lever available; (2) true panel/time-series data (repeated snapshots per customer) rather than one cross-sectional snapshot, which would let the survival analysis use actual observed lifetimes past 72 months instead of a restricted estimate; (3) real campaign A/B test data to replace the illustrative success-rate assumptions with an empirically grounded one.
