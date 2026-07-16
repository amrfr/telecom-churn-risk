"""
Step 2: statistical significance testing, entirely from scratch.

Two things v1 never established: (1) whether the churn-rate gaps between
segments (e.g. month-to-month vs two-year contracts) are statistically
distinguishable from noise at n=480, and (2) how uncertain each segment's
churn rate estimate actually is. Both matter more here than in a
million-row dataset, because 480 rows leaves real sampling noise.

Method choices (see METHODOLOGY.md for why):
- Chi-square statistic for each categorical driver x churn, but the p-value
  comes from a PERMUTATION test (shuffle the churn label 5,000 times,
  recompute chi-square, see how often the real statistic is beaten) rather
  than the chi-square CDF - this avoids needing an incomplete-gamma-function
  implementation and makes no distributional assumption beyond exchangeability
  under the null.
- 95% confidence intervals via bootstrap resampling (5,000 resamples) rather
  than the normal (Wald) approximation, which can misbehave at small n or
  when a segment's churn rate is near 0% or 100%.

Run: python src/02_statistical_tests.py  (after 01_churn_model.py)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(7)
N_PERM = 5000
N_BOOT = 5000

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
PROCESSED = os.path.join(BASE, "data", "processed")
FIGURES = os.path.join(BASE, "figures")

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                      "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_csv(os.path.join(RAW, "telco_customer_churn.csv"))
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)


def tenure_bucket(t):
    if t <= 12:
        return "0-12 mo"
    if t <= 24:
        return "13-24 mo"
    if t <= 48:
        return "25-48 mo"
    return "49+ mo"


df["TenureBucket"] = df["tenure"].apply(tenure_bucket)

rng = np.random.default_rng(7)


def chi_square_stat(group_vals, y):
    """Pearson chi-square statistic for a categorical column vs binary y."""
    cats = pd.unique(group_vals)
    n = len(y)
    stat = 0.0
    for c in cats:
        mask = group_vals == c
        n_c = mask.sum()
        if n_c == 0:
            continue
        observed_churn = y[mask].sum()
        observed_retain = n_c - observed_churn
        expected_churn = n_c * y.mean()
        expected_retain = n_c * (1 - y.mean())
        if expected_churn > 0:
            stat += (observed_churn - expected_churn) ** 2 / expected_churn
        if expected_retain > 0:
            stat += (observed_retain - expected_retain) ** 2 / expected_retain
    return stat


def permutation_pvalue(group_vals, y, n_perm=N_PERM):
    observed = chi_square_stat(group_vals, y)
    count_ge = 0
    y_arr = y.copy()
    for _ in range(n_perm):
        y_perm = rng.permutation(y_arr)
        stat = chi_square_stat(group_vals, y_perm)
        if stat >= observed:
            count_ge += 1
    p = (count_ge + 1) / (n_perm + 1)  # add-one smoothing, standard for permutation tests
    return observed, p


categorical_drivers = ["Contract", "InternetService", "PaymentMethod", "TenureBucket",
                        "OnlineSecurity", "TechSupport", "PaperlessBilling", "SeniorCitizen",
                        "Partner", "Dependents"]

y = df["ChurnFlag"].values
test_rows = []
for col in categorical_drivers:
    group_vals = df[col].astype(str).values
    n_cats = len(pd.unique(group_vals))
    dof = n_cats - 1
    stat, p = permutation_pvalue(group_vals, y)
    test_rows.append({"driver": col, "n_categories": n_cats, "degrees_of_freedom": dof,
                       "chi_square_statistic": round(float(stat), 2),
                       "permutation_p_value": round(float(p), 4),
                       "significant_at_0.05": bool(p < 0.05)})

sig_df = pd.DataFrame(test_rows).sort_values("chi_square_statistic", ascending=False).reset_index(drop=True)
sig_df.to_csv(os.path.join(PROCESSED, "chi_square_significance_tests.csv"), index=False)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
colors = ["#2ca02c" if s else "#bbbbbb" for s in sig_df["significant_at_0.05"]]
bars = ax.barh(sig_df["driver"], sig_df["chi_square_statistic"], color=colors)
ax.invert_yaxis()
for b, p in zip(bars, sig_df["permutation_p_value"]):
    label = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    ax.text(b.get_width() + max(sig_df["chi_square_statistic"]) * 0.01, b.get_y() + b.get_height() / 2,
            label, va="center", fontsize=9)
ax.set_xlabel("Chi-square statistic (higher = stronger association with churn)")
ax.set_title("Which churn drivers are statistically real vs. noise?\n(5,000-permutation test; green = p<0.05)")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "chi_square_significance.png"), dpi=150)
plt.close(fig)


def bootstrap_ci(mask, n_boot=N_BOOT, ci=0.95):
    sub = y[mask]
    n = len(sub)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(sub, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(sub.mean()), float(lo), float(hi)


ci_rows = []
overall_rate, overall_lo, overall_hi = bootstrap_ci(np.ones(len(df), dtype=bool))
ci_rows.append({"segment": "Overall book", "n": len(df), "churn_rate": round(overall_rate, 4),
                 "ci95_low": round(overall_lo, 4), "ci95_high": round(overall_hi, 4)})

for col in ["Contract", "InternetService", "TenureBucket"]:
    for val in df[col].unique():
        mask = (df[col] == val).values
        rate, lo, hi = bootstrap_ci(mask)
        ci_rows.append({"segment": f"{col} = {val}", "n": int(mask.sum()), "churn_rate": round(rate, 4),
                         "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)})

ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv(os.path.join(PROCESSED, "bootstrap_churn_rate_cis.csv"), index=False)

plot_df = ci_df[ci_df["segment"] != "Overall book"].copy()
plot_df = plot_df.sort_values("churn_rate")
fig, ax = plt.subplots(figsize=(8.5, 7))
y_pos = np.arange(len(plot_df))
ax.errorbar(plot_df["churn_rate"] * 100, y_pos,
            xerr=[(plot_df["churn_rate"] - plot_df["ci95_low"]) * 100,
                  (plot_df["ci95_high"] - plot_df["churn_rate"]) * 100],
            fmt="o", color="#08519c", ecolor="#9ecae1", elinewidth=3, capsize=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_df["segment"])
ax.axvline(overall_rate * 100, color="gray", linestyle="--", linewidth=1,
           label=f"Book average ({overall_rate*100:.1f}%)")
ax.set_xlabel("Churn rate (%) with 95% bootstrap CI")
ax.set_title(f"Segment churn rates with uncertainty (n={len(df)}, {N_BOOT:,} bootstrap resamples)")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "bootstrap_confidence_intervals.png"), dpi=150)
plt.close(fig)

findings = {
    "chi_square_significance_tests": sig_df.to_dict(orient="records"),
    "bootstrap_confidence_intervals": ci_df.to_dict(orient="records"),
    "methodology_note": ("p-values from 5,000-permutation tests (shuffle Churn label, recompute chi-square); "
                          "confidence intervals from 5,000-resample bootstrap. No scipy/statsmodels used."),
}
with open(os.path.join(PROCESSED, "findings_statistical_tests.json"), "w") as f:
    json.dump(findings, f, indent=2, default=str)

n_sig = int(sig_df["significant_at_0.05"].sum())
print(f"Step 2 (stats) complete. {n_sig}/{len(sig_df)} drivers significant at p<0.05.")
print(sig_df[["driver", "chi_square_statistic", "permutation_p_value", "significant_at_0.05"]].to_string(index=False))
