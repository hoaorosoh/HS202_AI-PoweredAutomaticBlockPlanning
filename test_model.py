"""
Step 3: Load the saved model, evaluate on data/test.csv, show accuracy + graphs.

Install:  pip install pandas scikit-learn joblib matplotlib
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay,
                             PrecisionRecallDisplay, roc_auc_score)

TEST_PATH = "data/test.csv"
MODEL_DIR = "models"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

model = joblib.load(f"{MODEL_DIR}/railway_model.joblib")
meta = joblib.load(f"{MODEL_DIR}/model_meta.joblib")

df = pd.read_csv(TEST_PATH)
X_test = df[meta["features"]]
y_test = df[meta["target"]]

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
print(f"Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print(f"ROC-AUC       : {auc:.4f}\n")
print(classification_report(y_test, y_pred, target_names=["No Maintenance", "Maintenance"]))

fig, ax = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(f"Railway Predictive Maintenance  |  Accuracy = {acc*100:.2f}%", fontsize=15)

# 1. Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["No Maint.", "Maint."], cmap="Blues", ax=ax[0, 0])
ax[0, 0].set_title("Confusion Matrix")

# 2. ROC curve
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax[0, 1])
ax[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.5)
ax[0, 1].set_title("ROC Curve")

# 3. Precision-Recall curve
PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax[1, 0])
ax[1, 0].set_title("Precision-Recall Curve")

# 4. Top 15 feature importances
names = model.named_steps["prep"].get_feature_names_out()
imp = model.named_steps["clf"].feature_importances_
top = np.argsort(imp)[-15:]
ax[1, 1].barh([names[i].split("__")[-1] for i in top], imp[top], color="teal")
ax[1, 1].set_title("Top 15 Feature Importances")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/evaluation.png", dpi=150)
print(f"Graph saved to {OUT_DIR}/evaluation.png")
plt.show()
