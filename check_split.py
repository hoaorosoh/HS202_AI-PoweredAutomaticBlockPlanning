import re
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())


def _col(df, name):
    return next((c for c in df.columns if norm(c) == norm(name)), None)


def _is_id(c):
    return "id" in [t for t in re.split(r"[^a-z0-9]+", c.lower()) if t] or norm(c).endswith("id")


def run_checks(train, test):
    status = []

    def report(level, msg):
        status.append(level)
        print(f"  [{level}] {msg}")

    print("=" * 64 + "\nSPLIT BALANCE REPORT\n" + "=" * 64)

    # 1. sizes
    n_tr, n_te = len(train), len(test)
    print(f"\n1) Sizes: train={n_tr:,} ({n_tr/(n_tr+n_te):.1%})  test={n_te:,} ({n_te/(n_tr+n_te):.1%})")

    # 2. target
    target = _col(train, "Maintenance Required")
    r_tr, r_te = train[target].mean(), test[target].mean()
    print(f"\n2) Target '{target}': train={r_tr:.4f}  test={r_te:.4f}  diff={abs(r_tr-r_te)*100:.3f} pts")
    report("PASS" if abs(r_tr - r_te) < 0.005 else "FAIL", "target rate matches within 0.5 points")

    # 3. failure type / severity
    for name in ("Failure Type", "Failure Severity"):
        c = _col(train, name)
        if c is None:
            continue
        a = train[c].fillna("None").astype(str).value_counts()
        b = test[c].fillna("None").astype(str).value_counts()
        tab = pd.concat([a, b], axis=1, keys=["train", "test"]).fillna(0)
        prop = tab / tab.sum()
        print(f"\n3) {c} distribution:")
        print(prop.round(4).assign(diff=(prop.train - prop.test).abs().round(4)).to_string())
        p = chi2_contingency(tab.values)[1]
        report("PASS" if (prop.train - prop.test).abs().max() < 0.005 and p > 0.01 else "WARN",
               f"{c}: max diff {(prop.train - prop.test).abs().max()*100:.2f} pts, chi2 p={p:.3f}")

    skip = {target} | {c for c in train.columns if _is_id(c)} | \
           {c for c in train.columns if norm(c) in ("failuretype", "failureseverity")}
    cat_cols = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c]) and c not in skip
                and train[c].nunique() <= 50]
    num_cols = [c for c in train.select_dtypes("number").columns if c not in skip]

    # 4. categorical features
    print("\n4) Categorical features (largest proportion gap per column):")
    worst = []
    for c in cat_cols:
        pa = train[c].fillna("__nan__").value_counts(normalize=True)
        pb = test[c].fillna("__nan__").value_counts(normalize=True)
        worst.append((c, pa.sub(pb, fill_value=0).abs().max()))
    for c, d in sorted(worst, key=lambda x: -x[1])[:5]:
        print(f"     {c:<28} {d*100:.2f} pts")
    report("PASS" if all(d < 0.01 for _, d in worst) else "WARN", "categorical mix within 1 point")

    # 5. numeric features (KS) + missingness
    print("\n5) Numeric features (KS statistic, 0 = identical):")
    ks = []
    for c in num_cols:
        a, b = train[c].dropna(), test[c].dropna()
        if len(a) and len(b):
            ks.append((c, ks_2samp(a, b).statistic))
    for c, d in sorted(ks, key=lambda x: -x[1])[:5]:
        print(f"     {c:<28} D={d:.4f}")
    report("PASS" if all(d < 0.02 for _, d in ks) else "WARN", "numeric distributions match (all D < 0.02)")
    miss = (train[num_cols + cat_cols].isna().mean() - test[num_cols + cat_cols].isna().mean()).abs()
    report("PASS" if miss.max() < 0.01 else "WARN",
           f"missing-value rates match (max gap {miss.max()*100:.2f} pts, column '{miss.idxmax()}')")

    # 6. duplicates shared between sets
    cols = [c for c in train.columns if c in test.columns]
    h_tr = set(pd.util.hash_pandas_object(train[cols], index=False))
    h_te = pd.util.hash_pandas_object(test[cols], index=False)
    shared = int(h_te.isin(h_tr).sum())
    print(f"\n6) Test rows that also appear (identical) in train: {shared:,}")
    report("PASS" if shared == 0 else "WARN", "no identical rows leak between train and test")

    # 7. ID overlap
    for c in [c for c in train.columns if _is_id(c)]:
        overlap = test[c].isin(set(train[c])).mean()
        reuse = len(train) / max(train[c].nunique(), 1)
        print(f"\n7) ID column '{c}': {overlap:.1%} of test IDs also in train "
              f"(each ID appears ~{reuse:.1f}x in train)")
        if overlap > 0.5 and reuse > 2:
            report("INFO", f"'{c}' repeats across rows; if rows of one {c} are related, set "
                           f"GROUP_COL = '{c}' in split_data.py to avoid leakage")

    # 8. adversarial validation
    print("\n8) Adversarial validation (can a model tell train rows from test rows?)")
    feat = [c for c in cat_cols + num_cols]
    both = pd.concat([train[feat], test[feat]], ignore_index=True)
    for c in cat_cols:
        both[c] = pd.factorize(both[c])[0]
    lab = np.r_[np.zeros(len(train)), np.ones(len(test))]
    if len(both) > 60000:
        idx = np.random.RandomState(0).choice(len(both), 60000, replace=False)
        both, lab = both.iloc[idx], lab[idx]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=100, random_state=0), both, lab,
                          cv=StratifiedKFold(3, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    auc = roc_auc_score(lab, p)
    print(f"     AUC = {auc:.4f}  (0.50 = indistinguishable = perfect split)")
    report("PASS" if auc < 0.52 else ("WARN" if auc < 0.55 else "FAIL"), "train and test look alike to a classifier")

    verdict = "FAIL" if "FAIL" in status else ("WARN" if "WARN" in status else "PASS")
    print(f"\nOVERALL: {verdict}  ({status.count('PASS')} pass, {status.count('WARN')} warn, {status.count('FAIL')} fail)")
    return verdict


if __name__ == "__main__":
    run_checks(pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv"))