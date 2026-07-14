# Sources

## Dataset

- [IBM Telco Customer Churn dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) ([repo](https://github.com/IBM/telco-customer-churn-on-icp4d)) - the widely-used public churn benchmark dataset (21 features: demographics, account info, services subscribed, monthly/total charges, churn label). Canonical size 7,043 rows; this repo uses a 487-row prefix (see `METHODOLOGY.md` for why).

## Industry churn benchmarks (`data/raw/industry_benchmarks.csv`)

- [S&P Global Market Intelligence - "US Broadband monthly churn hits 1.3%"](https://www.spglobal.com/market-intelligence/en/news-insights/research/2026/02/us-broadband-monthly-churn-hits-one-point-three-percent) - T-Mobile (0.90%/mo), AT&T (0.87%/mo) Q2 2025 postpaid phone churn; US broadband 1.3%/mo (Q3 2025, reported Feb 2026); US wireless postpaid industry-wide ~1.25%/mo (Q3 2025)
- [ChurnCost.com - "Telecom Churn Rate 2026: Benchmarks and Reduction Math"](https://churncost.com/telecom) - postpaid industry range 0.75%-3.0% monthly (~8.7%-30% annualized)
- [Focus Digital - "Average Churn Rate by Industry 2025"](https://focus-digital.co/average-churn-rate-by-industry/) and [CustomerGauge - "Average Churn Rate by Industry [2025 B2B Benchmarks]"](https://customergauge.com/blog/average-churn-rate-by-industry) - blended (postpaid + prepaid) telecom industry churn commonly cited at 20-25% annually; this repo uses the 22.5% midpoint
- [GitHub - sauravmishra1710/Telecom-Churn-Rate-Analysis](https://github.com/sauravmishra1710/Telecom-Churn-Rate-Analysis) - cites the widely-used industry rule of thumb that acquiring a new telecom customer costs 5-10x more than retaining an existing one, and a 15-25% annual churn range for the telecom industry broadly
- [Tridens Technology - "Why Telecom Customers Churn and How to Measure it?"](https://tridenstechnology.com/telecom-churn/) - general context on telecom churn measurement and drivers
- [ainvest.com - "The Telecom Churn: How Carriers Are Battling for Customer Loyalty in 2025"](https://www.ainvest.com/news/telecom-churn-carriers-battling-customer-loyalty-2025-2507/) - competitive context on 2025 carrier retention strategy

## Note on scraping vs. research-compiled figures

Industry benchmark figures were sourced via web search and direct citation of the reporting outlet above, not scraped from a live dashboard - they are point-in-time published figures, cited with as-of dates in `industry_benchmarks.csv`. This build environment's network access is restricted to a small domain allowlist (see `METHODOLOGY.md`), which also constrained the dataset size; where a number could not be independently verified against a second source, it is presented with a single citation rather than implied consensus.
