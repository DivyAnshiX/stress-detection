"""Evidence for the presentation and viva. Run after train_model.py.

Creates model/analysis/ with charts and tables:
  confusion_matrix.png, confidence_buckets.png/.csv, top_words.png/.csv, errors.csv
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from train_model import MODEL_FILE, MODEL_DIR, RANDOM_STATE, load_data

OUT = MODEL_DIR / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

data = load_data()
# Same split as train_model.py, so this is genuinely unseen test data.
_, x_test, _, y_test = train_test_split(
    data["text"], data["label"], test_size=0.20, random_state=RANDOM_STATE, stratify=data["label"]
)
pipe = joblib.load(MODEL_FILE)
proba = pipe.predict_proba(x_test)[:, list(pipe.classes_).index(1)]
pred = (proba >= 0.5).astype(int)

# 1. Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=["Not stressed", "Stressed"], cmap="Blues", ax=ax
)
ax.set_title("Confusion matrix (test set)")
fig.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=200)
plt.close(fig)

# 2. Accuracy by confidence bucket: justifies the "unclear" band
df = pd.DataFrame({"p": proba, "y": y_test.values, "pred": pred})
df["bucket"] = pd.cut(df["p"], [0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0], include_lowest=True)
table = (
    df.groupby("bucket", observed=True)
    .apply(lambda g: pd.Series({"n": len(g), "accuracy": (g["y"] == g["pred"]).mean()}), include_groups=False)
    .reset_index()
)
table.to_csv(OUT / "confidence_buckets.csv", index=False)
print("\nAccuracy by stress-score bucket (test set):")
print(table.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(table["bucket"].astype(str), table["accuracy"], color="#4C78A8")
ax.axhline(0.5, color="gray", ls="--", lw=1)
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Stress score range")
ax.set_title("Model accuracy by confidence")
for i, (a, n) in enumerate(zip(table["accuracy"], table["n"])):
    ax.text(i, a + 0.02, f"{a:.0%}\n(n={n})", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "confidence_buckets.png", dpi=200)
plt.close(fig)

# 3. Most influential words (Logistic Regression only)
clf = pipe.named_steps["classifier"]
if hasattr(clf, "coef_"):
    names = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coef = clf.coef_[0]
    order = np.argsort(coef)
    top = pd.DataFrame({
        "word": np.concatenate([names[order[-15:]][::-1], names[order[:15]]]),
        "weight": np.concatenate([coef[order[-15:]][::-1], coef[order[:15]]]),
        "pushes_towards": ["stress"] * 15 + ["not stress"] * 15,
    })
    top.to_csv(OUT / "top_words.csv", index=False)
    print("\nTop words pushing towards STRESS:", ", ".join(top["word"][:15]))
    print("Top words pushing towards NOT STRESS:", ", ".join(top["word"][15:]))
    fig, ax = plt.subplots(figsize=(7, 8))
    colors = ["#E45756" if t == "stress" else "#54A24B" for t in top["pushes_towards"]]
    ax.barh(top["word"][::-1], top["weight"][::-1], color=colors[::-1])
    ax.set_title("Words that push the prediction (red = stress, green = not stress)")
    fig.tight_layout()
    fig.savefig(OUT / "top_words.png", dpi=200)
    plt.close(fig)

# 4. Errors for the limitations slide
errors = pd.DataFrame({"text": x_test.values, "true": y_test.values, "predicted": pred, "score": proba.round(3)})
errors = errors[errors["true"] != errors["predicted"]]
errors.to_csv(OUT / "errors.csv", index=False)
print(f"\n{len(errors)} misclassified test rows saved to errors.csv")
print(f"Charts saved in: {OUT}")
