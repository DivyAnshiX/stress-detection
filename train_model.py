"""Train, compare, evaluate, and save the stress text classifier."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "stress.csv"
MODEL_DIR = ROOT / "model"
MODEL_FILE = MODEL_DIR / "stress_pipeline.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"
RANDOM_STATE = 42


def make_pipeline(classifier) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=100_000,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Could not find {DATA_FILE.name} in {ROOT}")
    data = pd.read_csv(DATA_FILE).dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    data["label"] = data["label"].astype(int)
    data = data[data["text"].ne("")]
    if set(data["label"].unique()) != {0, 1}:
        raise ValueError("The label column must contain both 0 and 1 classes.")
    return data


def main() -> None:
    data = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    candidates = {
        "logistic_regression": make_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "complement_naive_bayes": make_pipeline(ComplementNB(alpha=0.5)),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    comparison = {}
    for name, candidate in candidates.items():
        scores = cross_val_score(candidate, x_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        comparison[name] = {
            "cv_f1_macro_mean": round(float(scores.mean()), 4),
            "cv_f1_macro_std": round(float(scores.std()), 4),
            "fold_scores": [round(float(score), 4) for score in scores],
        }

    selected_name = max(comparison, key=lambda name: comparison[name]["cv_f1_macro_mean"])
    selected_pipeline = candidates[selected_name]
    selected_pipeline.fit(x_train, y_train)
    predictions = selected_pipeline.predict(x_test)

    report = classification_report(
        y_test,
        predictions,
        target_names=["Not stressed", "Stress-related"],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "rows_used": int(len(data)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "random_state": RANDOM_STATE,
        "selected_model": selected_name,
        "selection_metric": "3-fold training-only macro F1",
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": report,
        "model_comparison": comparison,
        "label_meaning": {"0": "not strongly stress-related", "1": "stress-related"},
        "warning": "This is a text-pattern estimate, not a medical diagnosis.",
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(selected_pipeline, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Selected model: {selected_name}")
    print(f"Saved model to: {MODEL_FILE}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print("Model comparison (training CV macro F1):")
    for name, result in comparison.items():
        print(f"  {name}: {result['cv_f1_macro_mean']:.4f} +/- {result['cv_f1_macro_std']:.4f}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
