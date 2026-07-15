# DISHA Implementation Plan

Status as of 2026-07-15. Saturday, 2026-07-18 is the current presentation
deadline.

## Completed and reproducible

1. Canonical, validated Evidence Objects that exclude raw text, pixels, and
   audio samples from the fusion boundary.
2. FER2013 ResNet-18 face baseline, temperature scaling, quality assessment,
   and evaluation artifacts.
3. GoEmotions multilabel text baseline and a MELD conversational seven-class
   text baseline with development-only model selection and dialogue-clustered
   test confidence intervals.
4. RAVDESS MFCC/RBF-SVM speech baseline with speaker-independent GroupKFold.
5. SUTRA reliability-aware fusion, uncertainty behavior, bounded actions,
   explicit crisis-rule separation, safety override, and in-memory temporal
   tracking.
6. Desktop demo with file input, webcam capture, microphone recording,
   temporary-media cleanup, and sanitized report export.
7. Executed unimodal and warm-runtime benchmarks plus an evidence-backed
   five-page research paper.

## Critical work before Saturday

1. Finish and verify the official MELD raw archive, then run aligned
   text/audio/video evaluation on untouched official test clips.
2. Train MELD-domain speech and visual adaptation heads using train only;
   tune fusion on development only; report test metrics, ablations, confusion
   matrices, calibration, and dialogue-clustered confidence intervals.
3. Test live webcam and microphone capture on presentation hardware and add
   graceful missing-device behavior.
4. Expand automated coverage for live-capture boundaries, modality failure,
   missing checkpoints, fusion ablations, and sanitized export invariants.
5. Update the paper and presentation only after the aligned benchmark is
   executed and visually verify every final PDF page.

## Research limitations that cannot honestly be called complete by a date

- FER2013 source, license, archive hash, and split provenance remain
  unverified for the local copy.
- Crisis rules and generated responses have not received qualified clinical
  validation and must not be described as a medical or crisis service.
- No human-subject effectiveness claim is permitted without ethics approval,
  a preregistered protocol, consent, and qualified reviewers.
