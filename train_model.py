import os
import re
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# ------------------------------ settings ------------------------------
TRAIN_PATH = "data/train.csv"
MODEL_DIR = "models"
RANDOM_STATE = 42
N_FOLDS = 5
TUNE = True              # False = skip the hyper-parameter search (faster)
N_ITER = 12              # random-search candidates
TUNE_ROWS = 40000        # rows used for the search
THRESHOLD_MODE = "accuracy"   # "accuracy" | "f1" | "precision95" | "default"
# ----------------------------------------------------------------------

norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())


def find_col(df, name, required=True):
    for c in df.columns:
        if norm(c) == norm(name):
            return c
    if required:
        raise KeyError(f"Column '{name}' not found. Available: {list(df.columns)}")
    return None


def is_id_or_date(col):
    tokens = [t for t in re.split(r"[^a-z0-9]+", col.lower()) if t]
    return ("id" in tokens or norm(col).endswith("id")
            or any(t in ("date", "time", "timestamp", "datetime") for t in tokens))


t0 = time.time()
os.makedirs(MODEL_DIR, exist_ok=True)
df = pd.read_csv(TRAIN_PATH)

target = find_col(df, "Maintenance Required")
ft_col = find_col(df, "Failure Type", required=False)
fs_col = find_col(df, "Failure Severity", required=False)
leak_cols = [c for c in (ft_col, fs_col) if c]

y = df[target].astype(int).values
y_type = df[ft_col].fillna("None").astype(str).values if ft_col else None
use_multi = y_type is not None
if use_multi:
    consistent = (y_type != "None").astype(int)
    print(f"Maintenance Required == (failure type present) for {(consistent == y).mean():.2%} of rows")

X = df.drop(columns=[target] + leak_cols)
drop_cols = [c for c in X.columns if is_id_or_date(c)
             or (not pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 50)]
X = X.drop(columns=drop_cols)
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]
print(f"Dropped as leakage: {leak_cols} | dropped IDs/dates/high-cardinality: {drop_cols}")
print(f"Features: {len(num_cols)} numeric, {len(cat_cols)} categorical | rows: {len(X):,}")


def make_pipeline(params):
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
    ])
    prep = ColumnTransformer([("num", "passthrough", num_cols), ("cat", cat_pipe, cat_cols)])
    mask = [False] * len(num_cols) + [True] * len(cat_cols)
    clf = HistGradientBoostingClassifier(
        categorical_features=mask if cat_cols else None,
        max_iter=600, early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
        random_state=RANDOM_STATE, **params)
    return Pipeline([("prep", prep), ("clf", clf)])


# ---------------- 1. hyper-parameter search (binary target) ----------------
DEFAULT_PARAMS = dict(learning_rate=0.05, max_depth=6, max_leaf_nodes=31,
                      min_samples_leaf=100, l2_regularization=1.0)
best_params = DEFAULT_PARAMS
if TUNE:
    print(f"\nHyper-parameter search ({N_ITER} candidates, 3-fold, scored on PR-AUC)...")
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(TUNE_ROWS, len(X)), replace=False)
    space = {"clf__learning_rate": [0.03, 0.05, 0.08, 0.12],
             "clf__max_depth": [None, 4, 6, 8],
             "clf__max_leaf_nodes": [15, 31, 63],
             "clf__min_samples_leaf": [20, 50, 100, 200],
             "clf__l2_regularization": [0.0, 0.5, 2.0, 5.0]}
    search = RandomizedSearchCV(make_pipeline({}), space, n_iter=N_ITER, scoring="average_precision",
                                cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
                                random_state=RANDOM_STATE, refit=False, n_jobs=1)
    search.fit(X.iloc[idx], y[idx])
    best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    print(f"Best params: {best_params} (CV PR-AUC {search.best_score_:.4f})")

# ---------------- 2. out-of-fold predictions for each candidate ----------------
print(f"\n{N_FOLDS}-fold cross-validation on the training data...")
skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
strat = y_type if use_multi else y
oof_bin = np.zeros(len(X))
oof_multi = np.zeros(len(X))
for k, (tr, va) in enumerate(skf.split(X, strat), 1):
    m = make_pipeline(best_params).fit(X.iloc[tr], y[tr])
    oof_bin[va] = m.predict_proba(X.iloc[va])[:, 1]
    msg = f"  fold {k}: binary AUC={roc_auc_score(y[va], oof_bin[va]):.4f}"
    if use_multi:
        mm = make_pipeline(best_params).fit(X.iloc[tr], y_type[tr])
        classes = list(mm.named_steps["clf"].classes_)
        oof_multi[va] = 1 - mm.predict_proba(X.iloc[va])[:, classes.index("None")]
        msg += f" | type-model AUC={roc_auc_score(y[va], oof_multi[va]):.4f}"
    print(msg)

candidates = {"binary": oof_bin}
if use_multi:
    candidates["multiclass"] = oof_multi
    candidates["blend"] = (oof_bin + oof_multi) / 2

rows = []
for name, p in candidates.items():
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    rows.append({"candidate": name, "PR-AUC": average_precision_score(y, p),
                 "ROC-AUC": roc_auc_score(y, p), "LogLoss": log_loss(y, pc)})
cv_table = pd.DataFrame(rows).set_index("candidate")
print(f"\nCross-validation (chance PR-AUC = {y.mean():.4f}):\n{cv_table.round(4).to_string()}")
best_name = cv_table["PR-AUC"].idxmax()
oof = candidates[best_name]
print(f"-> selected candidate: {best_name}")


# ---------------- 3. decision thresholds from out-of-fold predictions ----------------
def pick_thresholds(y, p):
    ts = np.unique(np.quantile(p, np.linspace(0.30, 0.9995, 700)))
    acc, prec, rec, f1 = [], [], [], []
    for t in ts:
        pred = p >= t
        tp = (pred & (y == 1)).sum()
        fp = (pred & (y == 0)).sum()
        pr = tp / max(tp + fp, 1)
        rc = tp / y.sum()
        acc.append((pred == y).mean())
        prec.append(pr)
        rec.append(rc)
        f1.append(2 * pr * rc / max(pr + rc, 1e-9))
    acc, prec, rec, f1 = map(np.array, (acc, prec, rec, f1))
    plateau = np.where(acc >= acc.max() - 1e-4)[0]          # middle of the best-accuracy plateau
    out = {"accuracy": float(ts[plateau[len(plateau) // 2]]),
           "f1": float(ts[np.argmax(f1)]), "default": 0.5}
    ok = np.where(prec >= 0.95)[0]
    out["precision95"] = float(ts[ok[0]]) if len(ok) else float(ts[np.argmax(prec)])
    return out


thresholds = pick_thresholds(y, oof)
print("\nOut-of-fold operating points (train data only):")
for name, t in thresholds.items():
    pred = oof >= t
    tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
    print(f"  {name:<12} thr={t:.3f}  acc={(pred == y).mean():.4f}  "
          f"precision={tp / max(tp + fp, 1):.4f}  recall={tp / y.sum():.4f}")
print(f"  majority-class baseline accuracy = {max(y.mean(), 1 - y.mean()):.4f}")
selected_threshold = thresholds[THRESHOLD_MODE]
print(f"Selected: THRESHOLD_MODE='{THRESHOLD_MODE}' -> {selected_threshold:.3f}")

# ---------------- 4. final fit on all training data ----------------
print("\nFitting final model(s) on all training data...")
final = {"mode": best_name, "binary": None, "multi": None}
if best_name in ("binary", "blend"):
    final["binary"] = make_pipeline(best_params).fit(X, y)
if best_name in ("multiclass", "blend"):
    final["multi"] = make_pipeline(best_params).fit(X, y_type)

# ---------------- 5. readable rules (explainability) ----------------
ref = final["binary"] or make_pipeline(best_params).fit(X, y)
Z = SimpleImputer(strategy="median").fit_transform(ref.named_steps["prep"].transform(X))
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, random_state=RANDOM_STATE).fit(Z, y)
with open(f"{MODEL_DIR}/rules_depth4.txt", "w") as f:
    f.write(export_text(tree, feature_names=num_cols + cat_cols, show_weights=True))

# ---------------- 6. save ----------------
joblib.dump(final, f"{MODEL_DIR}/railway_model.joblib")
joblib.dump({"target": target, "features": X.columns.tolist(), "num_cols": num_cols,
             "cat_cols": cat_cols, "leak_cols": leak_cols, "failure_type_col": ft_col,
             "failure_severity_col": fs_col, "thresholds": thresholds,
             "threshold_mode": THRESHOLD_MODE, "threshold": selected_threshold,
             "best_params": best_params, "candidate": best_name},
            f"{MODEL_DIR}/model_meta.joblib")
cv_table.to_csv(f"{MODEL_DIR}/cv_candidates.csv")
json.dump({k: (float(v) if not isinstance(v, str) and v is not None else v)
           for k, v in best_params.items()}, open(f"{MODEL_DIR}/best_params.json", "w"), indent=2)
print(f"Saved model to {MODEL_DIR}/railway_model.joblib  (total time {time.time() - t0:.0f}s)")