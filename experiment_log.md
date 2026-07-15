# DISHA Experiment Log

All entries correspond to executed runs. Do not record results that were not
produced by the relevant evaluation script.

## Experiments

| Date | Commit/state | Dataset | Model | Params | Accuracy | Macro-F1 | ECE | Latency | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 2026-07-08 | historical run | FER2013 | ResNet-18 | 11.18M | 0.6649 | 0.6557 | 0.1907 | 5.86 ms | Initial CUDA face baseline |
| 2026-07-08 | `f5f9362` lineage | FER2013 | ResNet-18 + temperature scaling | 11.18M | 0.6649 | 0.6557 | 0.1907 to 0.0162 | 5.12 ms CUDA | Validation-fitted temperature `T=2.385616` |
| 2026-07-13 | `4ffc960` + local artifact rerun | FER2013 | ResNet-18 + temperature scaling | 11.18M | 0.6649 | 0.6557 | 0.1907 to 0.0162 | 8.29 ms CUDA | Current local artifact; latency varies by environment |
| 2026-07-15 | local working tree | GoEmotions simplified | TF-IDF + one-vs-rest logistic SGD | 58,340 terms | 0.3193 subset | 0.4501 | N/A | 14.33 ms/text | Multilabel micro-F1 0.5222; threshold 0.55 selected on dev |
| 2026-07-15 | local working tree | MELD official text splits | Word/character TF-IDF + logistic regression | N/A | 0.5000 | 0.3703 | 0.0694 | N/A | Weighted-F1 0.5161; development-selected; 95% dialogue-bootstrap accuracy CI [0.4789, 0.5205] |
| 2026-07-15 | local working tree | RAVDESS speech | MFCC statistics + RBF SVM | 240 features | 0.4896 | 0.4680 | N/A | 32.42 ms/file | Six-fold speaker-independent GroupKFold over 24 actors |
| 2026-07-15 | local working tree | One text + face + speech demo tuple | Integrated DISHA runtime | N/A | N/A | N/A | N/A | 68.45 ms warm mean | 20 runs; systems benchmark only; no aligned multimodal accuracy claim |

Full machine-readable metrics and dataset hashes are stored in `experiments/`.
