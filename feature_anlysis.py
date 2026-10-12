import re
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
train = pd.read_csv("data/train.csv", keep_default_na=False, na_values=[""])
test = pd.read_csv("data/test.csv", keep_default_na=False, na_values=[""])
target = next(c for c in train.columns if norm(c) == "maintenancerequired")
fail_type = next(c for c in train.columns if norm(c) == "failuretype")
fail_sev = next(c for c in train.columns if norm(c) == "failureseverity")
id_cols = [c for c in train.columns if "id" in re.split(r"[^a-z0-9]+", c.lower()) or "date" in c.lower()]
drop = [target, fail_type, fail_sev] + id_cols


def make_X(a, b):
    Xa, Xb = a.drop(columns=drop), b.drop(columns=drop)
    cat = Xa.select_dtypes(exclude="number").columns
    if len(cat):
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xa[cat] = enc.fit_transform(Xa[cat].astype(str))
        Xb[cat] = enc.transform(Xb[cat].astype(str))
    imp = SimpleImputer(strategy="median")
    return (pd.DataFrame(imp.fit_transform(Xa), columns=Xa.columns),
            pd.DataFrame(imp.transform(Xb), columns=Xb.columns))


Xtr, Xte = make_X(train, test)
ytr, yte = train[target].astype(int).values, test[target].astype(int).values
base = max(yte.mean(), 1 - yte.mean())

# ---- 1. rule leaves: learned on train, judged on test ----
tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=50, random_state=0).fit(Xtr, ytr)
leaf_tr, leaf_te = tree.apply(Xtr), tree.apply(Xte)
rate = pd.Series(ytr).groupby(leaf_tr).agg(["mean", "size"])
rule_leaves = rate[(rate["mean"] >= 0.98) & (rate["size"] >= 50)].index
pred = np.isin(leaf_te, rule_leaves).astype(int)
tp, fp = ((pred == 1) & (yte == 1)).sum(), ((pred == 1) & (yte == 0)).sum()
print(f"Majority baseline accuracy        : {base:.4f}")
print(f"Rule-leaf classifier accuracy     : {(pred == yte).mean():.4f}")
print(f"  rule precision on test          : {tp / max(tp + fp, 1):.4f}")
print(f"  share of failures covered (recall): {tp / yte.sum():.4f}")
print(f"  => ~{(1 - tp / yte.sum()) * 100:.0f}% of failures are NOT explained by sensor rules\n")

shallow = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=0).fit(Xtr, ytr)
print("Top-level rules (depth 3):")
print(export_text(shallow, feature_names=list(Xtr.columns), show_weights=True))

# ---- 2. can the failure TYPE be predicted among failure rows? ----
mask = train[target] == 1
Xf, yf = Xtr[mask.values], train.loc[mask, fail_type].astype(str)
maj = yf.value_counts(normalize=True).iloc[0]
acc = cross_val_score(HistGradientBoostingClassifier(max_depth=5, random_state=0), Xf, yf,
                      cv=StratifiedKFold(3, shuffle=True, random_state=0)).mean()
print(f"Failure-type accuracy among failure rows: {acc:.4f}  (majority class = {maj:.4f})")
print("-> close to majority = failure type is essentially random given the sensors."
      if acc < maj + 0.05 else "-> failure type IS predictable: build a failure-type classifier.")