"""
Step 2: Train a model on data/train.csv and save it to models/.

Install:  pip install pandas scikit-learn joblib
"""
import os
import re
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TRAIN_PATH = "data/train.csv"
MODEL_DIR = "models"
RANDOM_STATE = 42


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_col(df, wanted):
    for c in df.columns:
        if norm(c) == norm(wanted):
            return c
    raise KeyError(f"Column '{wanted}' not found. Available: {list(df.columns)}")


os.makedirs(MODEL_DIR, exist_ok=True)
df = pd.read_csv(TRAIN_PATH)

target = find_col(df, "Maintenance Required")

# Other targets would leak the answer -> drop them from the features
leak_cols = [c for c in df.columns if norm(c) in ("failuretype", "failureseverity")]

y = df[target]
X = df.drop(columns=[target] + leak_cols)

# Drop ID / date-like / very high-cardinality text columns
drop_cols = []
for c in X.columns:
    n = norm(c)
    if any(k in n for k in ("id", "date", "timestamp")) and X[c].dtype == "object":
        drop_cols.append(c)
    elif X[c].dtype == "object" and X[c].nunique() > 50:
        drop_cols.append(c)
X = X.drop(columns=drop_cols)
print("Dropped leakage columns:", leak_cols)
print("Dropped ID/high-cardinality columns:", drop_cols)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
print(f"Features: {len(num_cols)} numeric, {len(cat_cols)} categorical")

preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])

model = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        class_weight="balanced",   # handles any class imbalance
        random_state=RANDOM_STATE,
    )),
])

print("Training...")
model.fit(X, y)
print(f"Training accuracy: {model.score(X, y):.4f}")

joblib.dump(model, f"{MODEL_DIR}/railway_model.joblib")
joblib.dump({"target": target, "features": X.columns.tolist(), "leak_cols": leak_cols},
            f"{MODEL_DIR}/model_meta.joblib")
print(f"Model saved to {MODEL_DIR}/railway_model.joblib")
