"""
Step 1: Load the dataset, split into train/test (stratified = balanced class ratio
in both parts) and save both as offline CSV files.

Install:  pip install pandas scikit-learn huggingface_hub fsspec
"""
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

HF_PATH = "hf://datasets/samyuktha01/Indian_Railway_maintance/indian_railway_predictive_maintenance_100k.csv"
OUT_DIR = "data"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_col(df, wanted):
    for c in df.columns:
        if norm(c) == norm(wanted):
            return c
    raise KeyError(f"Column '{wanted}' not found. Available: {list(df.columns)}")


os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(HF_PATH)
print("Loaded:", df.shape)

# Save a full offline copy of the raw data
df.to_csv(f"{OUT_DIR}/full_dataset.csv", index=False)

target = find_col(df, "Maintenance Required")
print(f"Target column: {target}")
print("Class distribution:\n", df[target].value_counts(normalize=True).round(3))

# Stratified split -> train and test have the same class proportions
train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, stratify=df[target], random_state=RANDOM_STATE
)

train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)

print(f"\nTrain: {train_df.shape} | Test: {test_df.shape}")
print("Train class ratio:\n", train_df[target].value_counts(normalize=True).round(3))
print("Test class ratio:\n", test_df[target].value_counts(normalize=True).round(3))
print(f"\nSaved to ./{OUT_DIR}/ (full_dataset.csv, train.csv, test.csv)")
