"""Train and evaluate a conversational text model on official MELD splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from disha.evaluation.statistics import classification_metrics, cluster_bootstrap_interval


LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
MELD_TO_DISHA = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(data_dir: Path, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    path = data_dir / f"{split}_sent_emo.csv"
    frame = pd.read_csv(path)
    labels = frame["Emotion"].map(MELD_TO_DISHA)
    if labels.isna().any():
        raise ValueError(f"unknown MELD labels in {path}")
    return frame, np.array([LABELS.index(label) for label in labels], dtype=np.int64)


def build_pipeline(c: float, class_weight: str | None, n_jobs: int = 1) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=80_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=80_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=c,
        class_weight=class_weight,
        max_iter=500,
        random_state=42,
        n_jobs=n_jobs,
    )
    return Pipeline([("vectorizer", features), ("classifier", classifier)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/meld/MELD-repo/data/MELD"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/text_meld_tfidf_logreg.joblib"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("experiments/text_meld_tfidf_logreg_metrics.json"),
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    train, y_train = load_split(args.data_dir, "train")
    dev, y_dev = load_split(args.data_dir, "dev")
    test, y_test = load_split(args.data_dir, "test")
    candidates = []
    started = time.perf_counter()
    for c in (0.5, 1.0, 2.0):
        for class_weight in (None, "balanced"):
            model = build_pipeline(c, class_weight, args.n_jobs)
            model.fit(train["Utterance"].fillna(""), y_train)
            prediction = model.predict(dev["Utterance"].fillna(""))
            candidates.append(
                {
                    "c": c,
                    "class_weight": class_weight,
                    "macro_f1": float(
                        f1_score(y_dev, prediction, average="macro", zero_division=0)
                    ),
                    "weighted_f1": float(
                        f1_score(y_dev, prediction, average="weighted", zero_division=0)
                    ),
                    "model": model,
                }
            )
    best = max(candidates, key=lambda item: (item["macro_f1"], item["weighted_f1"]))
    model = best.pop("model")
    probabilities = model.predict_proba(test["Utterance"].fillna(""))
    prediction = probabilities.argmax(axis=1)
    metrics = classification_metrics(y_test, probabilities, labels=LABELS)
    metrics["accuracy_dialogue_bootstrap_ci"] = cluster_bootstrap_interval(
        y_test,
        prediction,
        test["Dialogue_ID"].to_numpy(),
        accuracy_score,
    )
    metrics["macro_f1_dialogue_bootstrap_ci"] = cluster_bootstrap_interval(
        y_test,
        prediction,
        test["Dialogue_ID"].to_numpy(),
        lambda actual, predicted: f1_score(
            actual, predicted, average="macro", zero_division=0
        ),
    )
    payload = {
        "vectorizer": model.named_steps["vectorizer"],
        "classifier": model.named_steps["classifier"],
        "labels": LABELS,
        "canonical_labels": True,
        "dataset": "MELD",
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.checkpoint)
    files = {
        split: args.data_dir / f"{split}_sent_emo.csv"
        for split in ("train", "dev", "test")
    }
    report = {
        "experiment": "text_meld_tfidf_logreg",
        "dataset": "MELD official splits",
        "source": "https://github.com/declare-lab/MELD",
        "samples": {"train": len(train), "dev": len(dev), "test": len(test)},
        "labels": LABELS,
        "selection_metric": "development macro-F1",
        "selected_hyperparameters": best,
        "candidate_results": [
            {key: value for key, value in item.items() if key != "model"}
            for item in candidates
        ],
        "test_metrics": metrics,
        "majority_test_accuracy": float(
            np.max(np.bincount(y_test, minlength=len(LABELS))) / len(y_test)
        ),
        "training_and_selection_seconds": time.perf_counter() - started,
        "checkpoint_size_mb": args.checkpoint.stat().st_size / (1024 * 1024),
        "sha256": {split: sha256(path) for split, path in files.items()},
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
