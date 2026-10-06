"""Train and compare spam classifiers, then save the best one.

Usage:
    python src/train.py
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data import load

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "spam_classifier.joblib"
REPORTS = ROOT / "reports"
SEED = 42

MODELS = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=10, class_weight="balanced"),
    "Linear SVM": LinearSVC(C=1.0, class_weight="balanced"),
}


def build_pipeline(model) -> Pipeline:
    # Word unigrams + bigrams; sublinear TF dampens repeated words like "FREE FREE FREE".
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                 min_df=2, sublinear_tf=True)
    return Pipeline([("tfidf", vectorizer), ("clf", model)])


def main() -> None:
    df = load()
    y = (df["label"] == "spam").astype(int)
    print(f"Loaded {len(df)} unique messages ({y.mean():.1%} spam)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], y, test_size=0.2, stratify=y, random_state=SEED)

    results = {}
    pipelines = {}
    for name, model in MODELS.items():
        pipe = build_pipeline(model).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        }
        pipelines[name] = pipe

    print(f"{'Model':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    for name, m in results.items():
        print(f"{name:<22}{m['accuracy']:>10.3f}{m['precision']:>11.3f}"
              f"{m['recall']:>9.3f}{m['f1']:>8.3f}")

    # F1 on the spam class is the fairest single number on an imbalanced dataset.
    best = max(results, key=lambda n: results[n]["f1"])
    best_pipe = pipelines[best]
    print(f"\nBest model: {best}\n")
    print(classification_report(y_test, best_pipe.predict(X_test),
                                target_names=["ham", "spam"], digits=3))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(
        json.dumps({"best_model": best, "results": results}, indent=2))
    ConfusionMatrixDisplay.from_estimator(
        best_pipe, X_test, y_test, display_labels=["ham", "spam"], cmap="Blues")
    plt.title(f"Confusion matrix: {best}")
    plt.savefig(REPORTS / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    print(f"Saved model to {MODEL_PATH.relative_to(ROOT)} and reports to reports/")


if __name__ == "__main__":
    main()
