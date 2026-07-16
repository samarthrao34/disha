"""Run DISHA training jobs with cloud-friendly defaults.

The orchestrator runs selected training jobs with cloud-friendly defaults,
skips jobs whose datasets are not present, and writes a manifest describing what
was launched. It does not fabricate metrics; each child trainer owns its own
experiment output.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CloudJob:
    name: str
    command: list[str]
    required_paths: list[Path]

    def missing_paths(self) -> list[str]:
        return [str(path) for path in self.required_paths if not path.exists()]


def _python() -> str:
    return sys.executable or "python"


def build_jobs(args: argparse.Namespace) -> list[CloudJob]:
    return [
        CloudJob(
            name="face_fer2013",
            command=[
                _python(),
                "-m",
                "src.face.train",
                "--data-dir",
                str(args.fer2013_dir),
                "--model-name",
                args.face_model,
                "--epochs",
                str(args.face_epochs),
                "--batch-size",
                str(args.face_batch_size),
                "--num-workers",
                str(args.num_workers),
            ],
            required_paths=[args.fer2013_dir / "train", args.fer2013_dir / "test"],
        ),
        CloudJob(
            name="text_goemotions",
            command=[
                _python(),
                "-m",
                "disha.text.train_goemotions",
                "--data-dir",
                str(args.goemotions_dir),
                "--n-jobs",
                str(args.n_jobs),
            ],
            required_paths=[
                args.goemotions_dir / "train.tsv",
                args.goemotions_dir / "dev.tsv",
                args.goemotions_dir / "test.tsv",
                args.goemotions_dir / "emotions.txt",
            ],
        ),
        CloudJob(
            name="text_meld",
            command=[
                _python(),
                "-m",
                "disha.text.train_meld",
                "--data-dir",
                str(args.meld_dir),
                "--n-jobs",
                str(args.n_jobs),
            ],
            required_paths=[
                args.meld_dir / "train_sent_emo.csv",
                args.meld_dir / "dev_sent_emo.csv",
                args.meld_dir / "test_sent_emo.csv",
            ],
        ),
        CloudJob(
            name="speech_ravdess",
            command=[
                _python(),
                "-m",
                "disha.speech.train_ravdess",
                "--data-dir",
                str(args.ravdess_dir),
                "--archive",
                str(args.ravdess_archive),
                "--n-jobs",
                str(args.n_jobs),
            ],
            required_paths=[args.ravdess_dir, args.ravdess_archive],
        ),
    ]


def select_jobs(jobs: Sequence[CloudJob], requested: set[str]) -> list[CloudJob]:
    if "all" in requested:
        return list(jobs)
    known = {job.name for job in jobs}
    unknown = sorted(requested - known)
    if unknown:
        raise SystemExit(f"unknown job(s): {', '.join(unknown)}; choose from all, {', '.join(sorted(known))}")
    return [job for job in jobs if job.name in requested]


def cloud_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("OMP_NUM_THREADS", str(args.n_jobs))
    env.setdefault("MKL_NUM_THREADS", str(args.n_jobs))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def run_job(job: CloudJob, env: dict[str, str], dry_run: bool) -> dict[str, object]:
    missing = job.missing_paths()
    if missing:
        return {
            "name": job.name,
            "status": "skipped_missing_data",
            "missing_paths": missing,
            "command": job.command,
        }
    if dry_run:
        return {"name": job.name, "status": "planned", "command": job.command}
    started = time.perf_counter()
    completed = subprocess.run(job.command, cwd=ROOT, env=env, check=False)
    return {
        "name": job.name,
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
        "command": job.command,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DISHA cloud training jobs")
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=["all"],
        help="Jobs to run: all, face_fer2013, text_goemotions, text_meld, speech_ravdess",
    )
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--num-workers", type=int, default=max(2, min(8, os.cpu_count() or 2)))
    parser.add_argument("--face-model", default="resnet18")
    parser.add_argument("--face-epochs", type=int, default=15)
    parser.add_argument("--face-batch-size", type=int, default=128)
    parser.add_argument("--fer2013-dir", type=Path, default=ROOT / "data/raw/fer2013")
    parser.add_argument("--goemotions-dir", type=Path, default=ROOT / "data/raw/goemotions")
    parser.add_argument("--meld-dir", type=Path, default=ROOT / "data/raw/meld/MELD-repo/data/MELD")
    parser.add_argument("--ravdess-dir", type=Path, default=ROOT / "data/raw/ravdess/audio_speech")
    parser.add_argument(
        "--ravdess-archive",
        type=Path,
        default=ROOT / "data/raw/ravdess/Audio_Speech_Actors_01-24.zip",
    )
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/cloud_training_manifest.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = select_jobs(build_jobs(args), set(args.jobs))
    env = cloud_environment(args)
    results = [run_job(job, env, args.dry_run) for job in selected]
    manifest = {
        "created_at_unix": time.time(),
        "dry_run": args.dry_run,
        "n_jobs": args.n_jobs,
        "num_workers": args.num_workers,
        "results": results,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if any(item.get("status") == "failed" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
