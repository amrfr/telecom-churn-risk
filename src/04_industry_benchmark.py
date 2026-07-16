"""
Step 4: put this dataset's churn numbers next to real, cited, published
telecom-industry churn benchmarks. v1 never did this - a 25.3% churn rate
means nothing on its own without knowing what "normal" looks like for a
telecom book. See SOURCES.md for full citations.

Run: python src/04_industry_benchmark.py  (after 01_churn_model.py)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
PROCESSED = os.path.join(BASE, "data", "processed")
FIGURES = os.path.join(BASE, "figures")

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                      "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

bench = pd.read_csv(os.path.join(RAW, "industry_benchmarks.csv"))

with open(os.path.join(PROCESSED, "findings_model.json")) as f:
    model_findings = json.load(f)
this_dataset_annual_pct = model_findings["overall_churn_rate"] * 100

plot_rows = bench.dropna(subset=["value_pct_annualized"]).copy()
plot_rows = plot_rows[["segment", "value_pct_annualized"]].rename(columns={"value_pct_annualized": "annual_churn_pct"})
plot_rows = pd.concat([plot_rows, pd.DataFrame([{"segment": "This dataset (487-customer sample)",
                                                   "annual_churn_pct": this_dataset_annual_pct}])],
                       ignore_index=True)
plot_rows = plot_rows.sort_values("annual_churn_pct")
plot_rows.to_csv(os.path.join(PROCESSED, "industry_benchmark_comparison.csv"), index=False)

fig, ax = plt.subplots(figsize=(9.5, 6))
colors = ["#d62728" if s.startswith("This dataset") else "#1f77b4" for s in plot_rows["segment"]]
ax.barh(plot_rows["segment"], plot_rows["annual_churn_pct"], color=colors)
for i, (s, v) in enumerate(zip(plot_rows["segment"], plot_rows["annual_churn_pct"])):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlabel("Annualized churn rate (%)")
ax.set_title("This dataset vs. published telecom industry churn benchmarks")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "industry_benchmark_comparison.png"), dpi=150)
plt.close(fig)

best_in_class = bench[bench["segment"].isin(["T-Mobile US postpaid phone", "AT&T postpaid phone"])]["value_pct_annualized"].mean()
blended_industry = bench[bench["segment"].str.contains("Blended")]["value_pct_annualized"].iloc[0]

findings = {
    "this_dataset_annualized_churn_pct": round(this_dataset_annual_pct, 1),
    "best_in_class_us_postpaid_avg_annualized_pct": round(float(best_in_class), 1),
    "blended_industry_annualized_pct": round(float(blended_industry), 1),
    "how_this_dataset_compares": (
        f"This dataset's {this_dataset_annual_pct:.1f}% churn rate sits above the ~{blended_industry:.0f}% "
        f"midpoint of the commonly cited blended (postpaid+prepaid) industry range, and well above "
        f"best-in-class US postpaid carriers (T-Mobile/AT&T average ~{best_in_class:.0f}%/yr). Read this as "
        "evidence the sample is skewed toward higher-churn segments (month-to-month contracts, fiber "
        "internet - see churn_by_contract.csv / churn_by_internet.csv) rather than a representative "
        "national postpaid book. It does NOT mean the segment-level findings (contract type, tenure, "
        "internet type as churn drivers) are wrong - those directional patterns are well documented "
        "industry-wide, independent of this sample's overall level."
    ),
    "acquisition_vs_retention_cost_context": (
        "Widely cited industry rule of thumb: acquiring a new customer costs 5-10x more than retaining "
        "an existing one (see SOURCES.md). This is the economic backdrop for why even a modest-probability "
        "retention campaign (see campaign_roi_sensitivity_grid.csv) tends to pencil out."
    ),
}
with open(os.path.join(PROCESSED, "findings_industry_benchmark.json"), "w") as f:
    json.dump(findings, f, indent=2, default=str)

print("Step 4 complete. This dataset:", f"{this_dataset_annual_pct:.1f}%/yr",
      "| Best-in-class US postpaid avg:", f"{best_in_class:.1f}%/yr",
      "| Blended industry:", f"{blended_industry:.1f}%/yr")
