"""Measure warm integrated inference latency on one real sample per modality.

This is a systems latency benchmark, not an accuracy benchmark because the
three independent datasets do not provide aligned multimodal examples.
"""

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disha.runtime import DishaEngine


def main() -> None:
    face = next((ROOT / "data/raw/fer2013/test/happy").glob("*.jpg"))
    audio = next((ROOT / "data/raw/ravdess/audio_speech").rglob("03-01-03-*.wav"))
    text = "I am happy and excited today"
    engine = DishaEngine()

    cold_started = time.perf_counter()
    cold = engine.process(text=text, image_path=str(face), audio_path=str(audio))
    cold_ms = (time.perf_counter() - cold_started) * 1000.0

    timings = []
    result = cold
    for _ in range(20):
        started = time.perf_counter()
        result = engine.process(text=text, image_path=str(face), audio_path=str(audio))
        timings.append((time.perf_counter() - started) * 1000.0)

    payload = {
        "benchmark": "integrated_demo_latency",
        "accuracy_claim": None,
        "accuracy_note": "No aligned multimodal ground-truth dataset was evaluated.",
        "samples_per_iteration": {"text": 1, "face": 1, "speech": 1},
        "iterations": len(timings),
        "cold_start_ms": cold_ms,
        "warm_latency_mean_ms": statistics.mean(timings),
        "warm_latency_median_ms": statistics.median(timings),
        "warm_latency_p95_ms": sorted(timings)[int(0.95 * (len(timings) - 1))],
        "warm_latency_min_ms": min(timings),
        "warm_latency_max_ms": max(timings),
        "modalities_used": result.reasoning_trace.get("modalities_used"),
        "final_demo_fused_probabilities": result.reasoning_trace.get("fused_probabilities"),
        "hardware": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
    }
    output = ROOT / "experiments/integrated_demo_latency_metrics.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
