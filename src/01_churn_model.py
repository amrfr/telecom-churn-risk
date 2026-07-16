"""
Step 1: EDA + churn modeling, entirely from scratch (numpy/pandas only,
no scikit-learn or scipy - every number here is traceable to explicit code).

v2 additions over v1:
- 5-fold stratified cross-validation for the logistic regression (in addition
  to the original 80/20 holdout), so performance is reported as a mean +/-
  std across folds rather than a single lucky/unlucky split.
- A from-scratch Gaussian Naive Bayes model as an independent second method.
  If both models agree on which features matter most, that's real evidence
  the drivers aren't an artifact of one modeling choice - important given a
  modest sample size (see METHODOLOGY.md).

Run: python src/01_churn_model.py
Writes model artifacts to data/processed/model_artifacts/ for step 2.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
PROCESSED = os.path.join(BASE, "data", "processed")
ARTIFACTS = os.path.join(PROCESSED, "model_artifacts")
FIGURES = os.path.join(BASE, "figures")
for d in (PROCESSED, ARTIFACTS, FIGURES):
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                      "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

findings = {}

df = pd.read_csv(os.path.join(RAW, "telco_customer_churn.csv"))
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

findings["n_customers"] = int(len(df))
findings["overall_churn_rate"] = round(float(df["ChurnFlag"].mean()), 4)
findings["total_monthly_recurring_revenue_usd"] = round(float(df["MonthlyCharges"].sum()), 2)
findings["mrr_already_lost_to_churn_usd"] = round(float(df.loc[df["ChurnFlag"] == 1, "MonthlyCharges"].sum()), 2)


def tenure_bucket(t):
    if t <= 12:
        return "0-12 mo"
    if t <= 24:
        return "13-24 mo"
    if t <= 48:
        return "25-48 mo"
    return "49+ mo"


df["TenureBucket"] = df["tenure"].apply(tenure_bucket)

churn_by_contract = df.groupby("Contract")["ChurnFlag"].mean().sort_values(ascending=False)
churn_by_tenure = df.groupby("TenureBucket")["ChurnFlag"].mean().reindex(
    ["0-12 mo", "13-24 mo", "25-48 mo", "49+ mo"])
churn_by_internet = df.groupby("InternetService")["ChurnFlag"].mean().sort_values(ascending=False)
churn_by_payment = df.groupby("PaymentMethod")["ChurnFlag"].mean().sort_values(ascending=False)

for name, s in [("contract", churn_by_contract), ("tenure", churn_by_tenure),
                ("internet", churn_by_internet), ("payment", churn_by_payment)]:
    s.to_csv(os.path.join(PROCESSED, f"churn_rate_by_{name}.csv"))
    findings[f"churn_rate_by_{name}"] = (s * 100).round(1).to_dict()

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(churn_by_contract.index, churn_by_contract.values * 100, color="#d62728")
ax.axhline(findings["overall_churn_rate"] * 100, color="gray", linestyle="--", linewidth=1,
           label=f"Book average ({findings['overall_churn_rate']*100:.1f}%)")
for b, v in zip(bars, churn_by_contract.values):
    ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1, f"{v*100:.1f}%", ha="center")
ax.set_ylabel("Churn rate (%)")
ax.set_title("Month-to-month contracts churn at multiples of the book average")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "churn_by_contract.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(churn_by_tenure.index, churn_by_tenure.values * 100, color="#1f77b4")
for b, v in zip(bars, churn_by_tenure.values):
    ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1, f"{v*100:.1f}%", ha="center")
ax.set_ylabel("Churn rate (%)")
ax.set_title("Churn risk is concentrated in the first year of the relationship")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "churn_by_tenure.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(df.loc[df.ChurnFlag == 0, "MonthlyCharges"], bins=25, alpha=0.6, label="Retained", color="#2ca02c", density=True)
ax.hist(df.loc[df.ChurnFlag == 1, "MonthlyCharges"], bins=25, alpha=0.6, label="Churned", color="#d62728", density=True)
ax.set_xlabel("Monthly charges (USD)")
ax.set_ylabel("Density")
ax.set_title("Churned customers skew toward higher monthly bills")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "monthly_charges_by_churn.png"), dpi=150)
plt.close(fig)

feature_cols_num = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
feature_cols_cat = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                     "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                     "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                     "PaperlessBilling", "PaymentMethod"]

X_cat = pd.get_dummies(df[feature_cols_cat], drop_first=True)
X_full = pd.concat([df[feature_cols_num].reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
y_full = df["ChurnFlag"].values.astype(float)
feature_names = X_full.columns.tolist()
X_full = X_full.values.astype(float)
n_num = len(feature_cols_num)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_logreg(Xb, yv, lr=0.1, n_iter=3000, l2=0.5):
    w = np.zeros(Xb.shape[1])
    n = Xb.shape[0]
    for _ in range(n_iter):
        p = sigmoid(Xb @ w)
        grad = (Xb.T @ (p - yv)) / n
        grad[1:] += (l2 / n) * w[1:]
        w -= lr * grad
    return w


def standardize_fit(Xin, n_num):
    mu = Xin[:, :n_num].mean(axis=0)
    sigma = Xin[:, :n_num].std(axis=0)
    sigma[sigma == 0] = 1.0
    return mu, sigma


def standardize_apply(Xin, mu, sigma, n_num):
    Xout = Xin.copy()
    Xout[:, :n_num] = (Xin[:, :n_num] - mu) / sigma
    return Xout


def add_bias(X):
    return np.hstack([np.ones((len(X), 1)), X])


def eval_binary(y_true, p_pred, threshold=0.5):
    pred = (p_pred >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    accuracy = (tp + tn) / len(y_true) if len(y_true) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    pos_scores, neg_scores = p_pred[y_true == 1], p_pred[y_true == 0]
    if len(pos_scores) and len(neg_scores):
        auc = float(np.mean([np.mean(ps > neg_scores) + 0.5 * np.mean(ps == neg_scores) for ps in pos_scores]))
    else:
        auc = float("nan")
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1, auc=auc,
                tp=tp, tn=tn, fp=fp, fn=fn)


idx_pos = np.where(y_full == 1)[0]
idx_neg = np.where(y_full == 0)[0]
rng = np.random.default_rng(42)
rng.shuffle(idx_pos)
rng.shuffle(idx_neg)
split_pos = int(len(idx_pos) * 0.8)
split_neg = int(len(idx_neg) * 0.8)
train_idx = np.concatenate([idx_pos[:split_pos], idx_neg[:split_neg]])
test_idx = np.concatenate([idx_pos[split_pos:], idx_neg[split_neg:]])
rng.shuffle(train_idx)
rng.shuffle(test_idx)

X_train, y_train = X_full[train_idx], y_full[train_idx]
X_test, y_test = X_full[test_idx], y_full[test_idx]

mu, sigma = standardize_fit(X_train, n_num)
X_train_b = add_bias(standardize_apply(X_train, mu, sigma, n_num))
X_test_b = add_bias(standardize_apply(X_test, mu, sigma, n_num))

weights = train_logreg(X_train_b, y_train)
p_test = sigmoid(X_test_b @ weights)
holdout_metrics = eval_binary(y_test, p_test)

findings["model"] = {
    "n_train": int(len(y_train)), "n_test": int(len(y_test)),
    "accuracy": round(holdout_metrics["accuracy"], 3), "precision": round(holdout_metrics["precision"], 3),
    "recall": round(holdout_metrics["recall"], 3), "f1": round(holdout_metrics["f1"], 3),
    "auc": round(holdout_metrics["auc"], 3),
    "confusion_matrix": {"tp": holdout_metrics["tp"], "tn": holdout_metrics["tn"],
                          "fp": holdout_metrics["fp"], "fn": holdout_metrics["fn"]},
}

K = 5
fold_pos = np.array_split(rng.permutation(idx_pos), K)
fold_neg = np.array_split(rng.permutation(idx_neg), K)
cv_rows = []
for k in range(K):
    val_idx = np.concatenate([fold_pos[k], fold_neg[k]])
    tr_idx = np.concatenate([f for i, f in enumerate(fold_pos) if i != k] +
                             [f for i, f in enumerate(fold_neg) if i != k])
    Xtr, ytr = X_full[tr_idx], y_full[tr_idx]
    Xval, yval = X_full[val_idx], y_full[val_idx]
    mu_k, sigma_k = standardize_fit(Xtr, n_num)
    Xtr_b = add_bias(standardize_apply(Xtr, mu_k, sigma_k, n_num))
    Xval_b = add_bias(standardize_apply(Xval, mu_k, sigma_k, n_num))
    w_k = train_logreg(Xtr_b, ytr)
    p_val = sigmoid(Xval_b @ w_k)
    m = eval_binary(yval, p_val)
    m["fold"] = k + 1
    m["n_val"] = int(len(yval))
    cv_rows.append(m)

cv_df = pd.DataFrame(cv_rows)[["fold", "n_val", "accuracy", "precision", "recall", "f1", "auc"]]
cv_df.to_csv(os.path.join(PROCESSED, "cross_validation_folds.csv"), index=False)

cv_summary = {}
for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
    cv_summary[metric] = {"mean": round(float(cv_df[metric].mean()), 3),
                           "std": round(float(cv_df[metric].std(ddof=1)), 3)}
findings["cross_validation"] = {"k": K, "per_fold": cv_df.to_dict(orient="records"), "summary": cv_summary}

fig, ax = plt.subplots(figsize=(7, 5))
metrics_to_plot = ["accuracy", "precision", "recall", "auc"]
box_data = [cv_df[m].values for m in metrics_to_plot]
bp = ax.boxplot(box_data, tick_labels=[m.upper() for m in metrics_to_plot], patch_artist=True, widths=0.5)
for patch in bp["boxes"]:
    patch.set_facecolor("#9ecae1")
for i, m in enumerate(metrics_to_plot):
    ax.scatter([i + 1] * len(cv_df), cv_df[m].values, color="#08519c", zorder=3, s=25)
ax.set_title(f"{K}-fold cross-validation spread (n={len(y_full)} customers)\nSingle-split numbers above are one point in this spread")
ax.set_ylabel("Score")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "cross_validation_spread.png"), dpi=150)
plt.close(fig)

coef_series = pd.Series(weights[1:], index=feature_names).sort_values()
top_drivers = pd.concat([coef_series.head(8), coef_series.tail(8)])
fig, ax = plt.subplots(figsize=(9, 8))
colors = ["#2ca02c" if v < 0 else "#d62728" for v in top_drivers.values]
ax.barh(top_drivers.index, top_drivers.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Standardized coefficient\nred = raises churn odds, green = lowers churn odds")
ax.set_title("What actually drives churn, ranked (logistic regression)")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "churn_drivers.png"), dpi=150)
plt.close(fig)

thresholds = np.linspace(0, 1, 101)
tpr_list, fpr_list = [], []
for th in thresholds:
    pred = (p_test >= th).astype(int)
    tp_ = ((pred == 1) & (y_test == 1)).sum()
    fn_ = ((pred == 0) & (y_test == 1)).sum()
    fp_ = ((pred == 1) & (y_test == 0)).sum()
    tn_ = ((pred == 0) & (y_test == 0)).sum()
    tpr_list.append(tp_ / (tp_ + fn_) if (tp_ + fn_) else 0)
    fpr_list.append(fp_ / (fp_ + tn_) if (fp_ + tn_) else 0)
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.plot(fpr_list, tpr_list, color="#1f77b4", linewidth=2, label=f"Model (AUC={holdout_metrics['auc']:.2f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curve - holdout test set")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "roc_curve.png"), dpi=150)
plt.close(fig)


def train_gnb(Xb_nobias, yv):
    classes = [0, 1]
    stats = {}
    priors = {}
    for c in classes:
        Xc = Xb_nobias[yv == c]
        priors[c] = len(Xc) / len(Xb_nobias)
        mean = Xc.mean(axis=0)
        var = Xc.var(axis=0)
        var[var < 1e-6] = 1e-6
        stats[c] = (mean, var)
    return priors, stats


def gnb_log_likelihood(X, mean, var):
    return -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)


def predict_gnb(X, priors, stats):
    ll0 = gnb_log_likelihood(X, *stats[0]) + np.log(priors[0])
    ll1 = gnb_log_likelihood(X, *stats[1]) + np.log(priors[1])
    m = np.maximum(ll0, ll1)
    p1 = np.exp(ll1 - m) / (np.exp(ll0 - m) + np.exp(ll1 - m))
    return p1


X_train_s = standardize_apply(X_train, mu, sigma, n_num)
X_test_s = standardize_apply(X_test, mu, sigma, n_num)
priors, stats = train_gnb(X_train_s, y_train)
p_test_gnb = predict_gnb(X_test_s, priors, stats)
gnb_metrics = eval_binary(y_test, p_test_gnb)

mean_diff = stats[1][0] - stats[0][0]
gnb_importance = pd.Series(mean_diff, index=feature_names).sort_values()

findings["naive_bayes_crosscheck"] = {
    "accuracy": round(gnb_metrics["accuracy"], 3), "precision": round(gnb_metrics["precision"], 3),
    "recall": round(gnb_metrics["recall"], 3), "auc": round(gnb_metrics["auc"], 3),
    "top_drivers_raising_churn": gnb_importance.tail(8).round(3).to_dict(),
    "top_drivers_lowering_churn": gnb_importance.head(8).round(3).to_dict(),
}

logreg_top_raise = set(coef_series.tail(8).index)
gnb_top_raise = set(gnb_importance.tail(8).index)
overlap = logreg_top_raise & gnb_top_raise
findings["model_agreement"] = {
    "logreg_top8_raise_churn": sorted(logreg_top_raise),
    "naive_bayes_top8_raise_churn": sorted(gnb_top_raise),
    "overlap_count_of_8": len(overlap),
    "overlap_features": sorted(overlap),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=False)
lr_top = pd.concat([coef_series.head(8), coef_series.tail(8)])
colors_lr = ["#2ca02c" if v < 0 else "#d62728" for v in lr_top.values]
axes[0].barh(lr_top.index, lr_top.values, color=colors_lr)
axes[0].axvline(0, color="black", linewidth=0.8)
axes[0].set_title(f"Logistic regression\n(holdout AUC {holdout_metrics['auc']:.2f})")
gnb_top = pd.concat([gnb_importance.head(8), gnb_importance.tail(8)])
colors_gnb = ["#2ca02c" if v < 0 else "#d62728" for v in gnb_top.values]
axes[1].barh(gnb_top.index, gnb_top.values, color=colors_gnb)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_title(f"Naive Bayes cross-check\n(holdout AUC {gnb_metrics['auc']:.2f})")
fig.suptitle("Two independent methods, compared: do they agree on what drives churn?")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "model_agreement_comparison.png"), dpi=150)
plt.close(fig)

np.save(os.path.join(ARTIFACTS, "weights.npy"), weights)
np.save(os.path.join(ARTIFACTS, "mu.npy"), mu)
np.save(os.path.join(ARTIFACTS, "sigma.npy"), sigma)
with open(os.path.join(ARTIFACTS, "feature_names.json"), "w") as f:
    json.dump({"feature_cols_num": feature_cols_num, "feature_cols_cat": feature_cols_cat,
               "feature_names": feature_names, "n_num": n_num}, f, indent=2)
with open(os.path.join(PROCESSED, "findings_model.json"), "w") as f:
    json.dump(findings, f, indent=2, default=str)

print("Step 1 complete. Holdout accuracy:", findings["model"]["accuracy"], "| AUC:", findings["model"]["auc"])
print(f"CV ({K}-fold) AUC: {cv_summary['auc']['mean']} +/- {cv_summary['auc']['std']}")
print("Naive Bayes cross-check AUC:", findings["naive_bayes_crosscheck"]["auc"],
      "| driver overlap with logreg top-8:", findings["model_agreement"]["overlap_count_of_8"], "/8")
