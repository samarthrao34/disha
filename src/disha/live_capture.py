"""Local webcam and microphone capture for the research demo.

Captured files are temporary modality inputs. They are never embedded in
EvidenceObject instances or exported session reports.
"""

from pathlib import Path
import time

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf


def capture_webcam_frame(
    output_path: str | Path,
    *,
    camera_index: int = 0,
    warmup_frames: int = 20,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("could not open the webcam")
    frame = None
    try:
        for _ in range(max(1, warmup_frames)):
            ok, candidate = camera.read()
            if ok and candidate is not None and candidate.size:
                frame = candidate
            time.sleep(0.02)
    finally:
        camera.release()
    if frame is None:
        raise RuntimeError("the webcam opened but did not return a frame")
    if not cv2.imwrite(str(target), frame):
        raise RuntimeError(f"could not save webcam frame: {target}")
    return target


def record_microphone(
    output_path: str | Path,
    *,
    duration_seconds: float = 4.0,
    sample_rate: int = 16_000,
) -> Path:
    if not 1.0 <= duration_seconds <= 30.0:
        raise ValueError("duration_seconds must be between 1 and 30")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = int(round(duration_seconds * sample_rate))
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not np.isfinite(audio).all() or not np.any(np.abs(audio) > 1e-6):
        raise RuntimeError("microphone recording was empty")
    sf.write(target, audio, sample_rate, subtype="PCM_16")
    return target
