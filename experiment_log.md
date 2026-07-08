# DISHA Experiment Log

All entries must correspond to real runs. Do not record results that were not produced by `src/face/evaluate.py` (or the equivalent script for other modalities).

## Experiments

| Date | Commit | Dataset | Model | Params | Acc | Macro-F1 | ECE | Latency | Notes |
| ---- | ------ | ------- | ----- | ------ | --- | -------- | --- | ------- | ----- |
| 2026-07-08 | pending | FER2013 | ResNet-18 | 11.18M | 0.6649 | 0.6557 | 0.1907 | 5.86 ms | Initial face module baseline (CUDA) |
Temperature scaling applied, T=2.385616