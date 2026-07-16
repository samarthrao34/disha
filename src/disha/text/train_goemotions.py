"""Train and benchmark a reproducible GoEmotions TF-IDF + SGD baseline."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(path: Path):
    texts, labels = [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 2:
                continue
            texts.append(row[0])
            labels.append(tuple(int(value) for value in row[1].split(",")))
    return texts, labels


def threshold_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    predictions = (probabilities >= threshold).astype(np.int8)
    empty = predictions.sum(axis=1) == 0
    if np.any(empty):
        predictions[empty, probabilities[empty].argmax(axis=1)] = 1
    return predictions


def metrics(y_true, y_pred):
    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/goemotions")
    parser.add_argument("--output", default="checkpoints/text_goemotions_tfidf_sgd.joblib")
    parser.add_argument("--metrics", default="experiments/text_goemotions_tfidf_sgd_metrics.json")
    parser.add_argument("--max-features", type=int, default=60000)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    labels = [line.strip() for line in (data_dir / "emotions.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    train_texts, train_labels = load_split(data_dir / "train.tsv")
    dev_texts, dev_labels = load_split(data_dir / "dev.tsv")
    test_texts, test_labels = load_split(data_dir / "test.tsv")

    mlb = MultiLabelBinarizer(classes=list(range(len(labels))))
    y_train = mlb.fit_transform(train_labels)
    y_dev = mlb.transform(dev_labels)
    y_test = mlb.transform(test_labels)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=args.max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    started = time.perf_counter()
    x_train = vectorizer.fit_transform(train_texts)
    x_dev = vectorizer.transform(dev_texts)
    x_test = vectorizer.transform(test_texts)

    classifier = OneVsRestClassifier(
        SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            max_iter=50,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
        ),
        n_jobs=args.n_jobs,
    )
    classifier.fit(x_train, y_train)
    training_seconds = time.perf_counter() - started

    dev_probabilities = classifier.predict_proba(x_dev)
    candidates = np.arange(0.15, 0.56, 0.05)
    threshold_scores = {
        float(threshold): f1_score(
            y_dev,
            threshold_predictions(dev_probabilities, float(threshold)),
            average="micro",
            zero_division=0,
        )
        for threshold in candidates
    }
    threshold = max(threshold_scores, key=threshold_scores.get)

    test_probabilities = classifier.predict_proba(x_test)
    y_pred = threshold_predictions(test_probabilities, threshold)
    test_metrics = metrics(y_test, y_pred)

    latency_samples = test_texts[:500]
    latency_started = time.perf_counter()
    for text in latency_samples:
        classifier.predict_proba(vectorizer.transform([text]))
    latency_ms = (time.perf_counter() - latency_started) * 1000.0 / len(latency_samples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "labels": labels,
            "threshold": threshold,
            "dataset": "GoEmotions simplified split",
            "random_state": 42,
        },
        output_path,
        compress=3,
    )

    result = {
        "experiment": "text_goemotions_tfidf_sgd",
        "dataset": "GoEmotions simplified",
        "source": "https://github.com/google-research/google-research/tree/master/goemotions",
        "license": "CC BY 4.0 (dataset repository statement)",
        "train_samples": len(train_texts),
        "dev_samples": len(dev_texts),
        "test_samples": len(test_texts),
        "num_labels": len(labels),
        "multi_label": True,
        "model": "TF-IDF (1-2 grams) + one-vs-rest SGD logistic loss",
        "vocabulary_size": len(vectorizer.vocabulary_),
        "decision_threshold": threshold,
        "dev_threshold_micro_f1": float(threshold_scores[threshold]),
        "training_seconds": training_seconds,
        "latency_ms_per_text": latency_ms,
        "model_size_mb": output_path.stat().st_size / (1024 * 1024),
        "test_metrics": test_metrics,
        "sha256": {
            name: sha256(data_dir / name)
            for name in ("train.tsv", "dev.tsv", "test.tsv", "emotions.txt")
        },
    }
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
