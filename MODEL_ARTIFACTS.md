# DISHA Model Artifacts

The private repository includes the trained checkpoints needed to run the
demo without immediate retraining. Raw datasets are not included.

| Artifact | Training source | Size | SHA-256 |
| --- | --- | ---: | --- |
| `face_resnet18_fer2013.pt` | Local FER2013 split; provenance still unverified | 44,801,867 B | `b344bb2d50bddfd389b5a06da5a16183c75785e8ec05428308e5625f98c8f3ce` |
| `face_resnet18_temperature.pt` | Local FER2013 validation split | 1,689 B | `14733aa6bd2d4620bc4f583ed9b673f5e6cc14cbab7097734c4af518ff65a5cc` |
| `speech_ravdess_mfcc_svm.joblib` | RAVDESS speech, CC BY-NC-SA 4.0 | 1,649,813 B | `59d1516f24f59ffed5f9d2e0b07ea8bcf626b2a0451d43f6393e4687f7b25384` |
| `text_goemotions_tfidf_sgd.joblib` | GoEmotions, CC BY 4.0 | 13,036,337 B | `ae429642a1a8272bc9a412936f327ebf741bb1d8303fcad21054b04eb47b61b3` |
| `text_meld_tfidf_logreg.joblib` | MELD official text splits | 2,779,335 B | `035db19bf95249ced163bce13216f315f4152321723ff9ccb9a5f6ff50c05050` |

These are research artifacts, not clinically validated models. The face
checkpoint must not be redistributed from a public repository until the local
FER2013 source and terms are verified. Keep this repository private until that
governance task is resolved.
