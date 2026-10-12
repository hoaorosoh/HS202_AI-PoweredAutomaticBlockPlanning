import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score, matthews_corrcoef,
                             log_loss, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)

TRAIN_PATH, TEST_PATH = "data/train.csv", "data/test.csv"
MODEL_DIR, OUT_DIR = "models", "results"
N_BOOT = 300
os.makedirs(OUT_DIR, exist_ok=True)

model = joblib.load(f"{MODEL_DIR}/railway_model.joblib")
meta = joblib.load(f"{MODEL_DIR}/model_meta.joblib")
THR = meta["threshold"]


def predict_proba(X):
    parts = []
    if model["binary"] is not None:
        parts.append(model["binary"].predict_proba(X)[:, 1])
    if model["multi"] is not None:
        classes = list(model["multi"].named_steps["clf"].classes_)
        parts.append(1 - model["multi"].predict_proba(X)[:, classes.index("None")])
    return np.mean(parts, axis=0)


df_tr, df_te = pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
X_tr, y_tr = df_tr[meta["features"]], df_tr[meta["target"]].astype(int).values
X_te, y_te = df_te[meta["features"]], df_te[meta["target"]].astype(int).values
p_tr, p_te = predict_proba(X_tr), predict_proba(X_te)
pred_tr, pred_te = (p_tr >= THR).astype(int), (p_te >= THR).astype(int)
baseline = max(y_te.mean(), 1 - y_te.mean())


def all_metrics(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"Accuracy": accuracy_score(y, pred), "Balanced Accuracy": balanced_accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall (Sensitivity)": recall_score(y, pred, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "F1 Score": f1_score(y, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y, prob), "PR-AUC (Avg Precision)": average_precision_score(y, prob),
            "MCC": matthews_corrcoef(y, pred),
            "Log Loss": log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6))}


m_tr, m_te = all_metrics(y_tr, pred_tr, p_tr), all_metrics(y_te, pred_te, p_te)

# bootstrap 95 % confidence intervals on the test set
rng = np.random.RandomState(0)
boot = {k: [] for k in ("Accuracy", "Precision", "Recall (Sensitivity)", "F1 Score", "ROC-AUC", "PR-AUC (Avg Precision)")}
for _ in range(N_BOOT):
    i = rng.randint(0, len(y_te), len(y_te))
    yb, pb, qb = y_te[i], pred_te[i], p_te[i]
    boot["Accuracy"].append(accuracy_score(yb, pb))
    boot["Precision"].append(precision_score(yb, pb, zero_division=0))
    boot["Recall (Sensitivity)"].append(recall_score(yb, pb, zero_division=0))
    boot["F1 Score"].append(f1_score(yb, pb, zero_division=0))
    boot["ROC-AUC"].append(roc_auc_score(yb, qb))
    boot["PR-AUC (Avg Precision)"].append(average_precision_score(yb, qb))

table = pd.DataFrame({"Train": m_tr, "Test": m_te})
table["Gap (Train - Test)"] = table["Train"] - table["Test"]
table["Test 95% CI low"] = pd.Series({k: np.percentile(v, 2.5) for k, v in boot.items()})
table["Test 95% CI high"] = pd.Series({k: np.percentile(v, 97.5) for k, v in boot.items()})
table.to_csv(f"{OUT_DIR}/metrics_comparison.csv")

# every operating point on the test set
ops = []
for name, t in {**meta["thresholds"]}.items():
    pr = (p_te >= t).astype(int)
    m = all_metrics(y_te, pr, p_te)
    ops.append({"operating_point": name, "threshold": t, "flagged_share": pr.mean(),
                **{k: m[k] for k in ("Accuracy", "Balanced Accuracy", "Precision", "Recall (Sensitivity)", "F1 Score", "MCC")}})
ops = pd.DataFrame(ops).set_index("operating_point")
ops.to_csv(f"{OUT_DIR}/operating_points.csv")

print(f"Model: {meta['candidate']} | threshold mode '{meta['threshold_mode']}' = {THR:.3f}")
print(f"Majority-class baseline accuracy (test): {baseline:.4f}\n")
print("=== TRAIN VS TEST ===")
print(table.round(4).to_string())
lo, hi = table.loc["Accuracy", "Test 95% CI low"], table.loc["Accuracy", "Test 95% CI high"]
print(f"\nTest accuracy {m_te['Accuracy']:.4f} (95% CI {lo:.4f}-{hi:.4f}) is "
      f"{(m_te['Accuracy'] - baseline) * 100:+.2f} points vs the {baseline:.4f} baseline.")
if m_tr["Accuracy"] - m_te["Accuracy"] > 0.05:
    print("WARNING: train/test accuracy gap > 5 points -> overfitting.")
if lo <= baseline:
    print("NOTE: the baseline lies inside the confidence interval -> no reliable accuracy gain over guessing.")
print("\n=== ALL OPERATING POINTS (test) ===")
print(ops.round(4).to_string())
rep = classification_report(y_te, pred_te, target_names=["No Maintenance", "Maintenance"], zero_division=0)
print("\n--- TEST CLASSIFICATION REPORT ---\n" + rep)
open(f"{OUT_DIR}/classification_report_test.txt", "w").write(rep)

# detection by failure type / severity
det = []
for label, col in (("type", meta.get("failure_type_col")), ("severity", meta.get("failure_severity_col"))):
    if col and col in df_te.columns:
        g = df_te[col].fillna("None").astype(str)
        for v in sorted(g.unique()):
            mk = (g == v).values
            rate = pred_te[mk].mean()
            det.append({"group": label, "value": v, "rows": int(mk.sum()),
                        "flagged_rate": rate, "meaning": "correctly cleared" if v == "None" else "recall (detected)"})
            if v == "None":
                det[-1]["flagged_rate"] = 1 - rate
det = pd.DataFrame(det)
if len(det):
    det.to_csv(f"{OUT_DIR}/detection_by_failure.csv", index=False)
    print("\n=== DETECTION BY FAILURE TYPE / SEVERITY (test) ===")
    print(det.round(4).to_string(index=False))

pd.DataFrame({"true": y_te, "probability": p_te, "prediction": pred_te,
              "failure_type": df_te[meta["failure_type_col"]].fillna("None") if meta.get("failure_type_col") else ""}
             ).to_csv(f"{OUT_DIR}/test_predictions.csv", index=False)

#  evaluation.png 
fig, ax = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle(f"Railway Predictive Maintenance | Train Acc {m_tr['Accuracy']*100:.2f}% | "
             f"Test Acc {m_te['Accuracy']*100:.2f}% | Baseline {baseline*100:.2f}% | "
             f"Test ROC-AUC {m_te['ROC-AUC']:.3f}", fontsize=14)

ConfusionMatrixDisplay.from_predictions(y_te, pred_te, display_labels=["No Maint.", "Maint."],
                                        cmap="Blues", values_format="d", ax=ax[0, 0])
ax[0, 0].set_title(f"Test Confusion Matrix (threshold {THR:.3f})")

RocCurveDisplay.from_predictions(y_tr, p_tr, ax=ax[0, 1], name="Train")
RocCurveDisplay.from_predictions(y_te, p_te, ax=ax[0, 1], name="Test")
ax[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.5)
ax[0, 1].set_title("ROC Curve")

PrecisionRecallDisplay.from_predictions(y_tr, p_tr, ax=ax[0, 2], name="Train")
PrecisionRecallDisplay.from_predictions(y_te, p_te, ax=ax[0, 2], name="Test")
ax[0, 2].axhline(y_te.mean(), color="k", ls="--", alpha=0.5, label="Chance")
ax[0, 2].legend()
ax[0, 2].set_title("Precision-Recall Curve")

ts = np.unique(np.quantile(p_te, np.linspace(0, 0.9995, 300)))
sw = {k: [] for k in ("Accuracy", "Precision", "Recall", "F1")}
for t in ts:
    pr = (p_te >= t).astype(int)
    sw["Accuracy"].append(accuracy_score(y_te, pr))
    sw["Precision"].append(precision_score(y_te, pr, zero_division=1))
    sw["Recall"].append(recall_score(y_te, pr))
    sw["F1"].append(f1_score(y_te, pr, zero_division=0))
for k, v in sw.items():
    ax[1, 0].plot(ts, v, label=k)
ax[1, 0].axhline(baseline, color="gray", ls=":", label="Baseline acc")
ax[1, 0].axvline(THR, color="red", ls="--", label="Chosen threshold")
ax[1, 0].set_xlabel("Decision threshold")
ax[1, 0].set_title("Threshold Sweep (test)")
ax[1, 0].legend()

fr, mp = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")
ax[1, 1].plot(mp, fr, "o-", label="Model")
ax[1, 1].plot([0, 1], [0, 1], "k--", label="Perfect")
ax[1, 1].set_xlabel("Predicted probability")
ax[1, 1].set_ylabel("Observed failure rate")
ax[1, 1].set_title("Calibration (reliability)")
ax[1, 1].legend()

if len(det):
    dt = det[det.group == "type"].sort_values("flagged_rate")
    colors = ["#2ca02c" if v == "None" else "#d62728" for v in dt["value"]]
    ax[1, 2].barh(dt["value"], dt["flagged_rate"], color=colors)
    for i, (r, n) in enumerate(zip(dt["flagged_rate"], dt["rows"])):
        ax[1, 2].text(r + 0.01, i, f"{r:.2f}  (n={n})", va="center", fontsize=9)
    ax[1, 2].set_xlim(0, 1.25)
    ax[1, 2].set_title("Detection rate by failure type\n(red = recall, green = healthy rows correctly cleared)")
else:
    ax[1, 2].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/evaluation.png", dpi=150)

#  metrics_comparison.png 
fig2 = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig2, height_ratios=[1.2, 1])
axb = fig2.add_subplot(gs[0, :])
names = ["Accuracy", "Balanced Accuracy", "Precision", "Recall (Sensitivity)", "Specificity",
         "F1 Score", "ROC-AUC", "PR-AUC (Avg Precision)", "MCC"]
x, w = np.arange(len(names)), 0.38
b1 = axb.bar(x - w / 2, [m_tr[k] for k in names], w, label="Train", color="#1f77b4")
b2 = axb.bar(x + w / 2, [m_te[k] for k in names], w, label="Test", color="#ff7f0e")
axb.bar_label(b1, fmt="%.2f", fontsize=8)
axb.bar_label(b2, fmt="%.2f", fontsize=8)
axb.axhline(baseline, color="red", ls="--", lw=1, label=f"Majority baseline acc ({baseline:.2f})")
axb.set_xticks(x)
axb.set_xticklabels(names, rotation=20, ha="right")
axb.set_ylim(0, 1.1)
axb.set_title("Train vs Test Metrics")
axb.legend()
for col, (nm, yy, pp) in enumerate([("Train", y_tr, pred_tr), ("Test", y_te, pred_te)]):
    a = fig2.add_subplot(gs[1, col])
    ConfusionMatrixDisplay.from_predictions(yy, pp, display_labels=["No Maint.", "Maint."], normalize="true",
                                            cmap="Blues", values_format=".2f", ax=a)
    a.set_title(f"{nm} Confusion Matrix (row-normalised)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/metrics_comparison.png", dpi=150)

#  permutation importance 
idx = np.random.RandomState(1).choice(len(X_te), size=min(5000, len(X_te)), replace=False)
Xs, ys = X_te.iloc[idx].reset_index(drop=True), y_te[idx]
base_auc = roc_auc_score(ys, predict_proba(Xs))
imp = []
for f in meta["features"]:
    d = []
    for r in range(3):
        Xp = Xs.copy()
        Xp[f] = np.random.RandomState(r).permutation(Xp[f].values)
        d.append(base_auc - roc_auc_score(ys, predict_proba(Xp)))
    imp.append((f, np.mean(d), np.std(d)))
imp = pd.DataFrame(imp, columns=["feature", "importance_mean", "importance_std"]).sort_values("importance_mean", ascending=False)
imp.to_csv(f"{OUT_DIR}/feature_importance.csv", index=False)
top = imp.head(15).iloc[::-1]
plt.figure(figsize=(9, 7))
plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="teal")
plt.axvline(0, color="k", lw=0.8)
plt.title("Top 15 Permutation Importances (drop in test ROC-AUC)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)

print(f"\nSaved to {OUT_DIR}/: metrics_comparison.csv/.png, operating_points.csv, evaluation.png, "
      f"feature_importance.csv/.png, detection_by_failure.csv, test_predictions.csv, classification_report_test.txt")
if matplotlib.get_backend().lower() != "agg":
    plt.show()