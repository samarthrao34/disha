# DISHA: An Evidence-Governed Multimodal Emotion-Aware Research Prototype

**Samarth Rao**  
Independent research prototype, July 15, 2026

## Abstract

This paper presents DISHA, a reproducible research prototype for multimodal
emotion-aware interaction. DISHA processes facial images, speech recordings,
and text messages through separately evaluated classifiers, converts their
outputs into a common Evidence Object, performs reliability-weighted late
fusion through the SUTRA reasoner, tracks session-level emotion state, and
selects a bounded conversational action under a deterministic safety policy.
Unlike an earlier design manuscript, this report contains only measurements
executed in the accompanying repository. The face baseline is a temperature-
scaled ResNet-18 evaluated on 7,178 FER2013 test images (66.49% accuracy,
0.6557 macro-F1). The speech baseline uses 240-dimensional MFCC statistics and
an RBF SVM evaluated with six-fold speaker-independent GroupKFold on all 1,440
RAVDESS speech utterances (48.96% accuracy, 0.4680 macro-F1). The text baseline
uses TF-IDF bigrams and one-vs-rest SGD logistic classifiers on the official
GoEmotions simplified split (43,410 train; 5,426 validation; 5,427 test),
obtaining 0.5222 micro-F1 and 0.4501 macro-F1 on the 28-label multilabel task.
On one real sample per modality, the integrated warm pipeline averaged 68.45
ms over 20 runs on an RTX 3050 Laptop GPU. No fused-accuracy or clinical-safety
claim is made because no aligned multimodal or clinically annotated evaluation
set was used. The result is a working, inspectable MVP and an honest baseline
for subsequent research rather than a validated mental-health intervention.

**Keywords:** multimodal emotion recognition, affective computing, quality-
weighted fusion, FER2013, RAVDESS, GoEmotions, safety routing, reproducibility

## 1. Introduction

Emotion recognition systems commonly report a label from a single sensor. A
conversational system needs additional machinery: uncertain inputs must be
identified, multiple modalities must be reconciled, state must persist across
turns, and response behavior must remain bounded. DISHA explores this systems
problem through a modular research implementation.

The principal research question is not whether a prototype can claim a single
high accuracy number. Face, speech, and text benchmarks have different label
spaces and data-generating processes, making such a number invalid without an
aligned multimodal corpus. Instead, this work asks whether independently tested
components can communicate through a transparent evidence contract and support
a measurable end-to-end demo without hiding uncertainty.

The contributions implemented and evaluated here are:

1. A canonical Evidence Object that excludes raw sensor content and separates
   input reliability from model uncertainty.
2. Three reproducible emotion baselines trained or evaluated on real public
   datasets.
3. A quality-weighted late-fusion implementation named SUTRA.
4. Session-level exponential smoothing and bounded response-action routing.
5. Explicit crisis-indicator routing kept separate from emotion prediction.
6. Reproducible metrics, dataset checksums, artifacts, and a desktop demo.

DISHA is not presented as a therapist, diagnostic system, medical device, or
validated crisis service.

## 2. Related Work

Multimodal machine learning combines complementary sources while handling
alignment, representation, and fusion challenges [1]. FER2013 introduced a
widely used in-the-wild facial-expression benchmark [2]. RAVDESS provides
acted, perceptually validated audio-visual emotion recordings from 24 actors
[3]. GoEmotions provides 58k English Reddit comments annotated with 27 emotion
categories plus neutral [4]. These datasets are not aligned with one another;
therefore, their independent scores cannot be averaged into a fused accuracy.

Quality-aware late fusion is attractive for a resource-constrained prototype
because it preserves modality-specific models and makes weights inspectable.
DISHA uses this strategy instead of claiming a learned cross-modal model.

## 3. System Architecture

Each modality produces an Evidence Object containing modality, normalized
emotion probabilities, optional calibrated confidence, normalized predictive
entropy, an input-reliability score and category, availability, timestamp,
quality metadata, and model identity. Raw image pixels, waveforms, and message
text are deliberately excluded from the object passed to SUTRA.

For usable modality i, SUTRA assigns weight:

    w_i = r_i c_i

where r_i is input reliability and c_i is calibrated confidence when available
(otherwise 1). Fused probabilities are:

    p_fused(e) = sum_i w_i p_i(e) / sum_i w_i

This is an implemented heuristic, not a learned or calibrated fusion model.
The text model's 28 probabilities and the speech model's eight probabilities
are transparently projected into the seven FER-style labels used by fusion.

Session tracking applies an exponential update with alpha=0.35 and records no
raw input. SUTRA selects one of a fixed set of actions: clarify, acknowledge,
explore, encourage coping, recommend human support, provide crisis resources,
or safe fallback. A separate deterministic scan for explicit self-harm phrases
can override ordinary emotion routing. These rules are integration safeguards,
not clinically validated risk estimates.

## 4. Experimental Method

### 4.1 Face

The local FER2013 layout contains 28,709 training and 7,178 test images across
seven classes. Ten percent of the training directory was selected with seed 42
for validation. An ImageNet-initialized ResNet-18 was fine-tuned for 15 epochs;
the epoch-14 checkpoint with highest validation macro-F1 was retained.
Temperature scaling was fitted only on validation logits. Evaluation reports
accuracy, macro/weighted F1, 15-bin expected calibration error (ECE), and
single-image CUDA latency.

### 4.2 Speech

The official RAVDESS Audio Speech archive was downloaded from Zenodo. Its MD5
matched the publisher value `bc696df654c87fed845eb13823edef8a`. All 1,440
utterances and eight classes were used. Audio was resampled to 16 kHz and
trimmed. Forty MFCCs plus delta and delta-delta coefficients were summarized by
mean and standard deviation, producing 240 features. A standardized RBF SVM
with C=10 and balanced class weights was evaluated with six-fold GroupKFold by
actor. No actor crossed train/evaluation folds.

### 4.3 Text

The official GoEmotions simplified files were downloaded from the Google
Research repository and SHA-256 hashes were recorded. TF-IDF word unigrams and
bigrams (58,340 retained features) were classified by 28 one-vs-rest SGD
classifiers with logistic loss and balanced class weights. The multilabel
decision threshold was chosen from 0.15 to 0.55 using validation micro-F1, then
fixed for the test evaluation. Samples with no threshold crossing use their
highest-probability label.

### 4.4 Integrated pipeline

The integration benchmark uses one real positive sample from each independent
dataset and the message "I am happy and excited today." It measures systems
latency only. After one cold start, 20 warm iterations were executed in one
process. Because the inputs are not an aligned multimodal observation, this
experiment cannot estimate fused accuracy.

## 5. Results

| Component | Dataset/protocol | Accuracy | Macro-F1 | Additional result | Latency |
|---|---|---:|---:|---|---:|
| Face ResNet-18 | FER2013 held-out test | 66.49% | 0.6557 | ECE 0.1907 to 0.0162 | 8.29 ms/image |
| Speech MFCC+SVM | RAVDESS 6-fold actor GroupKFold | 48.96% | 0.4680 | Weighted-F1 0.4857 | 32.42 ms/file |
| Text TF-IDF+SGD | GoEmotions official test | N/A | 0.4501 | Micro-F1 0.5222 | 14.33 ms/text |
| Integrated system | 1 sample/modality, 20 warm runs | N/A | N/A | Median 66.30 ms | 68.45 ms mean |

GoEmotions subset accuracy was 31.93%, micro precision 0.4518, micro recall
0.6186, and Hamming loss 0.04715. Subset accuracy is strict: all labels for a
comment must match exactly.

Face calibration improved substantially after temperature scaling; however,
ECE depends on binning and does not establish real-world reliability. The
speaker-independent speech score is lower than values often obtained using
random utterance splits, illustrating the importance of preventing speaker
leakage. The three-modality demo produced a fused happy evidence value of
0.9274 for its selected positive examples, but this is one interaction and is
not reported as accuracy.

Cold start was 26.34 seconds, dominated by Python and ML-library import/model
initialization. Once loaded, warm latency ranged from 61.92 to 79.24 ms, with
95th-percentile latency 76.20 ms.

## 6. Safety, Privacy, and Ethics

The prototype uses bounded templates rather than unconstrained neural response
generation. Explicit crisis phrases trigger a conservative emergency-support
message independently of emotion output. This reduces one class of integration
risk but does not measure crisis sensitivity, specificity, or clinical value.
No crisis dataset or expert annotation was evaluated, and no crisis F1 is
reported.

Evidence Objects omit raw inputs, but the demo still reads local files and is
not a complete privacy architecture. There is no authentication, encrypted
persistent database, consent workflow, retention control, human escalation
service, or clinical oversight. Real-user deployment would be inappropriate
without those controls and prospective safety evaluation.

RAVDESS is licensed CC BY-NC-SA 4.0. The Google Research repository states CC
BY 4.0 for its datasets. FER2013 provenance for the local copy must be formally
recorded before redistribution. Dataset demographics and acted or online
expressions limit cultural and ecological validity.

## 7. Limitations

The modalities use different datasets and label taxonomies. Projection into
seven labels is manually specified and has not been empirically validated.
FER2013 facial labels are noisy; RAVDESS is acted North American English;
GoEmotions contains English Reddit text. The speech model is a compact baseline
and the text model is linear rather than transformer-based. Fusion thresholds
and reliability formulas are research defaults. Face detection falls back to a
center crop when the installed OpenCV build lacks a detector. The desktop demo
has a long cold start. No human conversation-quality study, reinforcement
learning, model compression study, crisis benchmark, clinical trial, or fused
accuracy experiment has been performed.

## 8. Conclusion

DISHA demonstrates a functioning, evidence-governed path from three real-data
emotion classifiers through reliability-weighted fusion, state tracking,
safety routing, and a bounded response. Its measured component results are
moderate and its limitations are substantial, but every number in this report
is tied to an executed artifact. The prototype provides a defensible baseline
for future aligned multimodal evaluation, trained fusion, stronger models,
calibrated uncertainty, culturally diverse validation, and expert-reviewed
safety research.

## References

[1] T. Baltrusaitis, C. Ahuja, and L.-P. Morency, "Multimodal Machine Learning:
A Survey and Taxonomy," IEEE TPAMI, 2019.

[2] I. J. Goodfellow et al., "Challenges in Representation Learning: A Report on
Three Machine Learning Contests," ICML Workshop, 2013.

[3] S. R. Livingstone and F. A. Russo, "The Ryerson Audio-Visual Database of
Emotional Speech and Song (RAVDESS)," PLOS ONE 13(5), 2018.
https://doi.org/10.1371/journal.pone.0196391

[4] D. Demszky et al., "GoEmotions: A Dataset of Fine-Grained Emotions," ACL,
2020. https://github.com/google-research/google-research/tree/master/goemotions

[5] J. Garcia and F. Fernandez, "A Comprehensive Survey on Safe Reinforcement
Learning," JMLR 16, 2015.

[6] DISHA experiment artifacts and source code, local repository, commit state
dated July 15, 2026.
