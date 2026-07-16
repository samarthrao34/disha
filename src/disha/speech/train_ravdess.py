"""Speaker-independent RAVDESS MFCC + SVM benchmark."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from disha.speech.features import extract_file_features


EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pipeline(probability: bool = False) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svm",
                SVC(
                    C=10.0,
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced",
                    probability=probability,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ravdess/audio_speech")
    parser.add_argument("--archive", default="data/raw/ravdess/Audio_Speech_Actors_01-24.zip")
    parser.add_argument("--cache", default="data/processed/ravdess_mfcc_features.npz")
    parser.add_argument("--output", default="checkpoints/speech_ravdess_mfcc_svm.joblib")
    parser.add_argument("--metrics", default="experiments/speech_ravdess_mfcc_svm_metrics.json")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    paths = sorted(Path(args.data_dir).rglob("*.wav"))
    if len(paths) != 1440:
        raise ValueError(f"expected 1440 RAVDESS speech files, found {len(paths)}")

    labels = [EMOTIONS[path.stem.split("-")[2]] for path in paths]
    groups = np.array([int(path.stem.split("-")[6]) for path in paths])
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feature_started = time.perf_counter()
    if cache_path.is_file():
        cached = np.load(cache_path)
        x = cached["features"]
    else:
        x = np.stack([extract_file_features(path) for path in tqdm(paths, desc="MFCC")])
        np.savez_compressed(cache_path, features=x)
    feature_seconds = time.perf_counter() - feature_started
    y = np.array(labels)

    cv = GroupKFold(n_splits=6)
    benchmark_started = time.perf_counter()
    predictions = cross_val_predict(
        build_pipeline(False), x, y, groups=groups, cv=cv, n_jobs=args.n_jobs
    )
    benchmark_seconds = time.perf_counter() - benchmark_started

    classes = sorted(EMOTIONS.values())
    cm = confusion_matrix(y, predictions, labels=classes)
    accuracy = float(accuracy_score(y, predictions))
    macro_f1 = float(f1_score(y, predictions, average="macro"))
    weighted_f1 = float(f1_score(y, predictions, average="weighted"))

    final_pipeline = build_pipeline(True)
    final_pipeline.fit(x, y)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": final_pipeline,
            "classes": list(final_pipeline.classes_),
            "feature": "40 MFCC + delta + delta-delta mean/std (240 dimensions)",
        },
        output_path,
        compress=3,
    )

    latency_paths = paths[:100]
    latency_started = time.perf_counter()
    for path in latency_paths:
        feature = extract_file_features(path).reshape(1, -1)
        final_pipeline.predict_proba(feature)
    latency_ms = (time.perf_counter() - latency_started) * 1000.0 / len(latency_paths)

    result = {
        "experiment": "speech_ravdess_mfcc_svm",
        "dataset": "RAVDESS Audio Speech Actors 01-24",
        "source": "https://zenodo.org/records/1188976",
        "license": "CC BY-NC-SA 4.0",
        "archive_md5": md5(Path(args.archive)),
        "samples": len(paths),
        "actors": len(set(groups)),
        "classes": classes,
        "evaluation": "6-fold speaker-independent GroupKFold (actors never cross folds)",
        "model": "MFCC statistics + StandardScaler + RBF SVM",
        "feature_dimensions": int(x.shape[1]),
        "feature_extraction_seconds": feature_seconds,
        "cross_validation_seconds": benchmark_seconds,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "latency_ms_per_file": latency_ms,
        "model_size_mb": output_path.stat().st_size / (1024 * 1024),
        "confusion_matrix": cm.tolist(),
    }
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("RAVDESS speaker-independent 6-fold confusion matrix")
    for row in range(len(classes)):
        for col in range(len(classes)):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(metrics_path.with_name("speech_ravdess_mfcc_svm_confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
