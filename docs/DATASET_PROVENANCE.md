# Dataset Provenance Register

No dataset may be redistributed, published, or used for a public benchmark
until all `UNVERIFIED` fields below are resolved.

## FER2013

| Field | Current record |
| --- | --- |
| Local path | `data/raw/fer2013` (ignored by Git) |
| Train images observed | 28,709 |
| Test images observed | 7,178 |
| Classes | angry, disgust, fear, happy, neutral, sad, surprise |
| Download source | **UNVERIFIED** |
| Dataset version | **UNVERIFIED** |
| License / terms | **UNVERIFIED** |
| Canonical citation | **UNVERIFIED** |
| Archive checksum | **UNVERIFIED** |
| Split checksum | **UNVERIFIED** |

The counts describe the current local files; they do not establish provenance
or permission to redistribute them.

## RAVDESS

| Field | Verified record |
| --- | --- |
| Local archive | `data/raw/ravdess/Audio_Speech_Actors_01-24.zip` (ignored by Git) |
| Source | `https://zenodo.org/records/1188976` |
| Archive MD5 | `bc696df654c87fed845eb13823edef8a` (matches Zenodo record) |
| License | CC BY-NC-SA 4.0 |
| Speech files | 1,440 |
| Speakers | 24 actors |
| Evaluation split | Six-fold `GroupKFold`; actors never cross train/test within a fold |

The experiment uses only the audio-speech archive. Full metrics are in
`experiments/speech_ravdess_mfcc_svm_metrics.json`.

## GoEmotions

| Field | Verified record |
| --- | --- |
| Local path | `data/raw/goemotions` (ignored by Git) |
| Source | `https://github.com/google-research/google-research/tree/master/goemotions` |
| License | CC BY 4.0 per the dataset repository |
| Train / dev / test | 43,410 / 5,426 / 5,427 examples |
| Labels | 28, multilabel |
| Split | Official simplified train/dev/test TSV files |

SHA-256 hashes for every downloaded TSV and the label file are recorded in
`experiments/text_goemotions_tfidf_sgd_metrics.json`.

## MELD

| Field | Current record |
| --- | --- |
| Annotation source | `https://github.com/declare-lab/MELD` |
| Repository license | GPL-3.0 for the released repository code |
| Official annotation counts | 9,989 train / 1,109 development / 2,610 test utterances |
| Labels | anger, disgust, fear, joy, neutral, sadness, surprise |
| Dialogue counts | 1,038 train / 114 development / 280 test |
| Raw archive source | `https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz` |
| Raw archive expected size | 10,878,146,150 bytes |
| Local raw archive status | Partial, resumable download; ignored by Git |

The repository contains a text-only experiment using official annotation
splits. No audio/video MELD result may be reported until the raw archive is
complete, verified, extracted, and evaluated.
