"""
Step 3: score every active customer with the churn model from step 1,
convert probabilities into annualized revenue-at-risk by decile, estimate
customer lifetime value via an empirical survival curve, and run a
retention-campaign ROI sensitivity analysis (not a single illustrative
number - a grid across plausible cost/success-rate assumptions, plus the
breakeven success rate solved analytically).

v2 additions over v1:
- A Kaplan-Meier-style survival curve built from tenure + churn status.
  This dataset is a legitimate (if single-snapshot) survival setup: churned
  customers' `tenure` is their full relationship length (an event); active
  customers' `tenure` is how long they've lasted so far (right-censored).
  The resulting survival curve gives a real median-lifetime estimate, which
  feeds a much better customer lifetime value (CLV) figure than "current
  MonthlyCharges x 12" alone.
- A cost x success-rate ROI sensitivity grid, plus the breakeven success
  rate (the minimum campaign win-rate at which ROI = 0) solved directly
  from the algebra - more useful to a decision-maker than one point estimate.

Run: python src/03_revenue_at_risk.py  (after 01_churn_model.py)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
PROCESSED = os.path.join(BASE, "data", "processed")
ARTIFACTS = os.path.join(PROCESSED, "model_artifacts")
FIGURES = os.path.join(BASE, "figures")

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                      "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_csv(os.path.join(RAW, "telco_customer_churn.csv"))
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

with open(os.path.join(ARTIFACTS, "feature_names.json")) as f:
    meta = json.load(f)
feature_cols_num, feature_cols_cat = meta["feature_cols_num"], meta["feature_cols_cat"]
n_num = meta["n_num"]

X_cat = pd.get_dummies(df[feature_cols_cat], drop_first=True)
X = pd.concat([df[feature_cols_num].reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
X = X.reindex(columns=meta["feature_names"], fill_value=0).values.astype(float)

weights = np.load(os.path.join(ARTIFACTS, "weights.npy"))
mu = np.load(os.path.join(ARTIFACTS, "mu.npy"))
sigma = np.load(os.path.join(ARTIFACTS, "sigma.npy"))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


X_s = X.copy()
X_s[:, :n_num] = (X[:, :n_num] - mu) / sigma
X_b = np.hstack([np.ones((len(X_s), 1)), X_s])
p_all = sigmoid(X_b @ weights)

risk_df = df.copy()
risk_df["churn_probability"] = p_all
risk_df["annualized_revenue_usd"] = risk_df["MonthlyCharges"] * 12

active = risk_df[risk_df["ChurnFlag"] == 0].copy()
active = active.sort_values("churn_probability", ascending=False).reset_index(drop=True)
active["risk_decile"] = pd.qcut(active["churn_probability"], 10, labels=False, duplicates="drop")
active["risk_decile"] = 10 - active["risk_decile"]

decile_summary = active.groupby("risk_decile").agg(
    customers=("customerID", "count"),
    avg_churn_probability=("churn_probability", "mean"),
    annualized_revenue_at_risk_usd=("annualized_revenue_usd", "sum"),
).round(2)
decile_summary.to_csv(os.path.join(PROCESSED, "revenue_at_risk_by_decile.csv"))
active.to_csv(os.path.join(PROCESSED, "active_customers_scored.csv"), index=False)

top_decile = active[active["risk_decile"] == 1]
top2_deciles = active[active["risk_decile"] <= 2]

total_active_rev = float(active["annualized_revenue_usd"].sum())
top2_rev = float(top2_deciles["annualized_revenue_usd"].sum())
proportional_share = (len(top2_deciles) / len(active)) * total_active_rev
lift_pct = (top2_rev / proportional_share - 1) * 100 if proportional_share else float("nan")

findings = {"revenue_at_risk": {
    "active_customers": int(len(active)),
    "total_active_annualized_revenue_usd": round(total_active_rev, 2),
    "top_decile_customers": int(len(top_decile)),
    "top_decile_avg_churn_probability": round(float(top_decile["churn_probability"].mean()), 3),
    "top_decile_annualized_revenue_at_risk_usd": round(float(top_decile["annualized_revenue_usd"].sum()), 2),
    "top2_deciles_annualized_revenue_at_risk_usd": round(top2_rev, 2),
    "top2_deciles_share_of_active_revenue_pct": round(top2_rev / total_active_rev * 100, 1),
    "top2_deciles_lift_vs_proportional_pct": round(lift_pct, 1),
}}

fig, ax = plt.subplots(figsize=(8, 5.5))
cum_rev = decile_summary["annualized_revenue_at_risk_usd"].sort_index().cumsum()
ax.bar(decile_summary.index.astype(str), decile_summary["annualized_revenue_at_risk_usd"], color="#ff7f0e")
ax2 = ax.twinx()
ax2.plot(decile_summary.index.astype(str), cum_rev, color="#1f77b4", marker="o", label="Cumulative")
ax.set_xlabel("Risk decile (1 = highest predicted churn risk)")
ax.set_ylabel("Annualized revenue at risk (USD)")
ax2.set_ylabel("Cumulative revenue at risk (USD)")
ax.set_title("Revenue at risk by decile - top deciles carry a modest concentration premium")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "revenue_at_risk_by_decile.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# v2: Kaplan-Meier survival curve.
# Event = churned (tenure is the full relationship length).
# Censored = still active (tenure is a lower bound on eventual relationship length).
# ---------------------------------------------------------------------------
km = df[["tenure", "ChurnFlag"]].copy()
event_times = np.sort(km.loc[km["ChurnFlag"] == 1, "tenure"].unique())

n_at_risk = len(km)
survival = 1.0
km_rows = [{"month": 0, "n_at_risk": n_at_risk, "n_events": 0, "survival_prob": 1.0}]
remaining = km.copy()
for t in event_times:
    n_at_risk_t = (remaining["tenure"] >= t).sum()
    n_events_t = ((remaining["tenure"] == t) & (remaining["ChurnFlag"] == 1)).sum()
    if n_at_risk_t > 0:
        survival *= (1 - n_events_t / n_at_risk_t)
    km_rows.append({"month": int(t), "n_at_risk": int(n_at_risk_t), "n_events": int(n_events_t),
                     "survival_prob": round(float(survival), 4)})

km_df = pd.DataFrame(km_rows)
km_df.to_csv(os.path.join(PROCESSED, "kaplan_meier_survival_curve.csv"), index=False)

median_idx = km_df[km_df["survival_prob"] <= 0.5]
median_tenure_months = int(median_idx.iloc[0]["month"]) if len(median_idx) else None

# expected remaining tenure for an active customer at their current tenure t0:
# integrate the survival curve forward from t0, normalized by S(t0) - a
# standard "restricted mean residual life" style estimate, restricted to the
# observed horizon (72 months, this dataset's max tenure) to avoid
# extrapolating past the data.
max_month = int(km_df["month"].max())
km_lookup = km_df.set_index("month")["survival_prob"].reindex(range(0, max_month + 1)).ffill().fillna(1.0)


def expected_remaining_tenure(t0):
    s_t0 = km_lookup.get(t0, km_lookup.iloc[-1])
    if s_t0 <= 0:
        return 0.0
    future = km_lookup.loc[t0:max_month]
    area = np.trapezoid(future.values, dx=1)
    return float(area / s_t0)


active["expected_remaining_tenure_months"] = active["tenure"].apply(
    lambda t: expected_remaining_tenure(min(int(t), max_month)))
active["estimated_clv_usd"] = active["MonthlyCharges"] * active["expected_remaining_tenure_months"]
active.to_csv(os.path.join(PROCESSED, "active_customers_scored.csv"), index=False)

clv_summary = active.groupby("risk_decile").agg(
    customers=("customerID", "count"),
    avg_expected_remaining_tenure_months=("expected_remaining_tenure_months", "mean"),
    total_estimated_clv_usd=("estimated_clv_usd", "sum"),
).round(1)
clv_summary.to_csv(os.path.join(PROCESSED, "clv_by_decile.csv"))

findings["survival_analysis"] = {
    "method": "Kaplan-Meier, event=churned, censored=still active, from scratch (no lifelines/scipy)",
    "median_survival_tenure_months": median_tenure_months,
    "survival_prob_at_12mo": float(km_lookup.get(12, np.nan)),
    "survival_prob_at_24mo": float(km_lookup.get(24, np.nan)),
    "survival_prob_at_48mo": float(km_lookup.get(48, np.nan)),
    "total_estimated_clv_of_active_book_usd": round(float(active["estimated_clv_usd"].sum()), 2),
    "top_decile_total_estimated_clv_usd": round(
        float(active.loc[active["risk_decile"] == 1, "estimated_clv_usd"].sum()), 2),
}

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.step(km_df["month"], km_df["survival_prob"] * 100, where="post", color="#1f77b4", linewidth=2)
ax.axhline(50, color="gray", linestyle=":", linewidth=1, label="50% retained")
if median_tenure_months is not None:
    ax.axvline(median_tenure_months, color="#d62728", linestyle="--", linewidth=1,
               label=f"Median survival ~{median_tenure_months} mo")
final_survival = float(km_df["survival_prob"].iloc[-1])
subtitle = (f"Median not reached within {max_month}mo observed window "
            f"(survival still {final_survival*100:.0f}% at {max_month}mo)" if median_tenure_months is None
            else f"Median survival ~{median_tenure_months} mo")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Estimated % of customers still retained")
ax.set_title(f"Kaplan-Meier survival curve (from-scratch estimate)\n{subtitle}")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "survival_curve.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.bar(clv_summary.index.astype(str), clv_summary["total_estimated_clv_usd"], color="#6a51a3")
ax.set_xlabel("Risk decile (1 = highest predicted churn risk)")
ax.set_ylabel("Estimated remaining CLV (USD)")
ax.set_title("Estimated remaining customer lifetime value at risk, by decile\n(survival-curve-based, not just 12mo of billings)")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "clv_by_decile.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# v2: campaign ROI sensitivity grid + breakeven success rate, replacing the
# single illustrative point estimate with a range decision-makers can use.
# ---------------------------------------------------------------------------
customers_targeted = int(len(top2_deciles))
avg_revenue_targeted = float(top2_deciles["annualized_revenue_usd"].mean())

cost_grid = [10, 20, 30, 40, 50]
success_grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

roi_rows = []
for cost in cost_grid:
    for success in success_grid:
        total_cost = customers_targeted * cost
        expected_saved = customers_targeted * success
        expected_revenue_saved = expected_saved * avg_revenue_targeted
        roi = (expected_revenue_saved - total_cost) / total_cost if total_cost else float("nan")
        roi_rows.append({"cost_per_customer_usd": cost, "assumed_success_rate": success,
                          "roi_multiple": round(roi, 2)})
roi_grid_df = pd.DataFrame(roi_rows)
roi_grid_pivot = roi_grid_df.pivot(index="cost_per_customer_usd", columns="assumed_success_rate", values="roi_multiple")
roi_grid_pivot.to_csv(os.path.join(PROCESSED, "campaign_roi_sensitivity_grid.csv"))

# breakeven success rate at each cost: solve success * avg_revenue_targeted == cost -> success = cost / avg_revenue_targeted
breakeven_rows = [{"cost_per_customer_usd": c, "breakeven_success_rate": round(c / avg_revenue_targeted, 4)}
                   for c in cost_grid]
breakeven_df = pd.DataFrame(breakeven_rows)
breakeven_df.to_csv(os.path.join(PROCESSED, "campaign_breakeven_success_rates.csv"), index=False)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
im = ax.imshow(roi_grid_pivot.values, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=roi_grid_pivot.values.max())
ax.set_xticks(range(len(success_grid)))
ax.set_xticklabels([f"{s*100:.0f}%" for s in success_grid])
ax.set_yticks(range(len(cost_grid)))
ax.set_yticklabels([f"${c}" for c in cost_grid])
ax.set_xlabel("Assumed campaign success rate")
ax.set_ylabel("Cost per customer targeted")
ax.set_title("Retention campaign ROI sensitivity (top-2-decile customers)")
for i in range(len(cost_grid)):
    for j in range(len(success_grid)):
        ax.text(j, i, f"{roi_grid_pivot.values[i, j]:.1f}x", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, label="ROI multiple")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "campaign_roi_sensitivity.png"), dpi=150)
plt.close(fig)

# keep one illustrative headline number (same assumptions as v1) for continuity
campaign_cost_per_customer = 30.0
campaign_success_rate = 0.25
campaign_total_cost = customers_targeted * campaign_cost_per_customer
expected_customers_saved = customers_targeted * campaign_success_rate
expected_revenue_saved = expected_customers_saved * avg_revenue_targeted
roi = (expected_revenue_saved - campaign_total_cost) / campaign_total_cost if campaign_total_cost else float("nan")

findings["retention_campaign_roi_illustrative"] = {
    "assumptions": {"cost_per_customer_usd": campaign_cost_per_customer,
                     "assumed_success_rate": campaign_success_rate},
    "customers_targeted": customers_targeted,
    "campaign_total_cost_usd": round(campaign_total_cost, 2),
    "expected_customers_saved": round(expected_customers_saved, 1),
    "expected_annualized_revenue_saved_usd": round(float(expected_revenue_saved), 2),
    "roi_multiple": round(float(roi), 2),
}
findings["retention_campaign_sensitivity"] = {
    "cost_grid_usd": cost_grid,
    "success_rate_grid": success_grid,
    "roi_grid": roi_grid_df.to_dict(orient="records"),
    "breakeven_success_rate_by_cost": breakeven_df.to_dict(orient="records"),
    "interpretation": ("Breakeven success rate is the minimum win-rate a retention campaign needs to hit "
                        "for ROI=0 at a given cost/customer. At $30/customer the breakeven is "
                        f"{30/avg_revenue_targeted*100:.1f}% - well below any published retention-campaign "
                        "success rate, which is why this pencils out even under conservative assumptions."),
}

with open(os.path.join(PROCESSED, "findings_model.json")) as f:
    model_findings = json.load(f)
model_findings.update(findings)
with open(os.path.join(PROCESSED, "findings.json"), "w") as f:
    json.dump(model_findings, f, indent=2, default=str)

print("Step 3 complete. Top-2-decile annualized revenue at risk (USD):",
      findings["revenue_at_risk"]["top2_deciles_annualized_revenue_at_risk_usd"])
print("Median survival tenure (months):", findings["survival_analysis"]["median_survival_tenure_months"])
print("Total estimated CLV of active book (USD):",
      findings["survival_analysis"]["total_estimated_clv_of_active_book_usd"])
print("Illustrative campaign ROI multiple:", findings["retention_campaign_roi_illustrative"]["roi_multiple"],
      "| breakeven success rate at $30/customer:",
      f"{30/avg_revenue_targeted*100:.1f}%")
