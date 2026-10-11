import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from check_split import run_checks

HF_PATH = "hf://datasets/samyuktha01/Indian_Railway_maintance/indian_railway_predictive_maintenance_100k.csv"
OUT_DIR = "data"
FULL_PATH = f"{OUT_DIR}/full_dataset.csv"
TEST_SIZE = 0.25
RANDOM_STATE = 42
GROUP_COL = None          # e.g. "train_id" -> all rows of one group go to train OR test
DROP_DUPLICATES = False   # True removes exact duplicate rows before splitting

norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())


def find_col(df, name):
    return next((c for c in df.columns if norm(c) == norm(name)), None)


os.makedirs(OUT_DIR, exist_ok=True)
if os.path.exists(FULL_PATH):
    df = pd.read_csv(FULL_PATH)
    print(f"Loaded local copy {FULL_PATH}: {df.shape}")
else:
    df = pd.read_csv(HF_PATH)
    df.to_csv(FULL_PATH, index=False)
    print(f"Downloaded and saved {FULL_PATH}: {df.shape}")

if DROP_DUPLICATES:
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicate rows")

target = find_col(df, "Maintenance Required")
ft, fs = find_col(df, "Failure Type"), find_col(df, "Failure Severity")

# stratification key: failure type | severity (falls back to the target)
if ft and fs:
    strata = df[ft].fillna("None").astype(str) + "|" + df[fs].fillna("None").astype(str)
    rare = strata.map(strata.value_counts()) < 10
    strata[rare] = df.loc[rare, ft].fillna("None").astype(str)
else:
    strata = df[target].astype(str)
print(f"Stratifying on {strata.nunique()} groups")

if GROUP_COL:
    sgkf = StratifiedGroupKFold(n_splits=round(1 / TEST_SIZE), shuffle=True, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(sgkf.split(df, strata, groups=df[GROUP_COL]))
    train_df, test_df = df.iloc[tr_idx], df.iloc[te_idx]
else:
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=strata, random_state=RANDOM_STATE)

train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)
print(f"\nSaved {OUT_DIR}/train.csv {train_df.shape} and {OUT_DIR}/test.csv {test_df.shape}\n")

run_checks(pd.read_csv(f"{OUT_DIR}/train.csv"), pd.read_csv(f"{OUT_DIR}/test.csv"))