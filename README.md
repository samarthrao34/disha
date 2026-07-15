# DISHA Reference Implementation

DISHA is a safety-governed multimodal emotional-support research framework.
The repository is an early research implementation, not a therapist, medical
device, diagnostic system, or production crisis service.

## Current status

- A FER2013 face-emotion baseline has been trained and evaluated with a
  ResNet-18 backbone.
- Face probabilities can be temperature-scaled and converted into the
  canonical `EvidenceObject` used by SUTRA.
- A trained GoEmotions text baseline provides a runnable
  text-to-SUTRA-to-response path.
- A conversational MELD text model is selected on the official development
  split and evaluated once on the official test split with dialogue-clustered
  confidence intervals.
- A speaker-independent RAVDESS speech baseline converts WAV files into
  canonical evidence using MFCC statistics and an RBF SVM.
- A Tkinter application supports files, live webcam capture, four-second
  microphone recording, conservative SUTRA fusion, session trends,
  safety-policy overrides, and sanitized JSON report export.
- SUTRA performs conservative evidence fusion and selects bounded response
  actions.
- Safety policy can override normal actions when explicit crisis indicators
  are present.
- Aligned multimodal accuracy evaluation, qualified human evaluation,
  persistent consent-aware storage, and production safety validation remain
  future work.

## Research rules

1. No result without an executed experiment.
2. No threshold without clearly marking whether it is calibrated or a
   conservative research default.
3. No clinical or diagnostic claims.
4. SUTRA receives structured Evidence Objects, never raw image, audio, or text.
5. Crisis-risk signals are kept separate from ordinary emotion labels.

See [docs/NO_FAKE_RESULTS_POLICY.md](docs/NO_FAKE_RESULTS_POLICY.md) and
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run tests

```powershell
python -m pytest -q
```

## Run the text integration baseline

```powershell
$env:PYTHONPATH="src"
python -m disha.runtime --text "I had a difficult day at work"
```

The command returns the bounded response, selected SUTRA action, safety
decision, and sanitized evidence metadata.

## Run the professor demo

Double-click `RUN_DEMO.bat`, or run:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe demo_app.py
```

The first multimodal analysis may take about 30 seconds to load all models.
The measured warm mean was 68.45 ms over 20 runs on the local RTX 3050 laptop.
See `PROFESSOR_DEMO_GUIDE.md` for the exact presentation sequence and
`output/pdf/DISHA_Genuine_Research_Paper.pdf` for the evidence-backed paper.

The private repository includes the trained checkpoints required by the demo.
Raw datasets and temporary captures are intentionally excluded. Checkpoint
hashes, training sources, and limitations are recorded in
`MODEL_ARTIFACTS.md`.

## Face experiment

The FER2013 dataset is expected at:

```text
data/raw/fer2013/train/<class>/*.jpg
data/raw/fer2013/test/<class>/*.jpg
```

Train and evaluate:

```powershell
python -m src.face.train --data-dir data/raw/fer2013 --model-name resnet18 --epochs 15
python -m src.face.temperature_scaling --data-dir data/raw/fer2013
python -m src.face.evaluate --data-dir data/raw/fer2013 --model-name resnet18
python -m src.face.test_evidence_real --data-dir data/raw/fer2013 --index 0
```

Dataset provenance, licensing, and checksums must be verified and recorded
before publishing or reproducing any experiment.

## Architecture

```text
modality input
  -> modality model
  -> quality and uncertainty assessment
  -> canonical EvidenceObject (no raw input)
  -> SUTRA conservative fusion and action selection
  -> safety-policy override
  -> bounded response renderer
```

Operational thresholds, crisis behavior, and response quality require
dedicated validation before use with real users.
