# DISHA Professor Demo Guide

## Before the meeting

1. Open `output/pdf/DISHA_Genuine_Research_Paper.pdf`.
2. Double-click `RUN_DEMO.bat` at least five minutes before presenting.
3. Keep the demo open. The first multimodal analysis may take about 30 seconds; warm analyses average about 68 ms on this machine.

## Recommended live demonstration

1. Enter: `I am happy and excited today.`
2. Choose this face image:
   `data/raw/fer2013/test/happy/PrivateTest_10077120.jpg`
3. Choose this speech file:
   `data/raw/ravdess/audio_speech/Actor_01/03-01-03-01-01-01-01.wav`
4. Click **Analyze with DISHA**.
5. Point out that text, face, and speech each produce a probability distribution and quality score. SUTRA fuses only those structured Evidence Objects; it does not receive raw media.
6. Point out the session trend and the safety decision. Explain that session state is in memory only and contains probabilities, not raw text, images, or audio.

## What to say about the results

- Face: FER2013 ResNet-18, 66.49% test accuracy and 65.57% macro-F1.
- Text: GoEmotions TF-IDF plus one-vs-rest logistic SGD, 52.22% test micro-F1 and 45.01% macro-F1.
- Speech: RAVDESS MFCC plus RBF SVM, speaker-independent 6-fold accuracy 48.96% and macro-F1 46.80%.
- Integrated runtime: 68.45 ms mean warm latency over 20 runs on this laptop; cold model loading was 26.34 seconds.
- There is no fused-accuracy claim because no aligned text-face-speech ground-truth dataset was evaluated.

## Honest scope statement

Say this plainly: “DISHA is a research prototype for emotion-aware support. It is not a therapist, diagnostic system, or validated crisis service. Today’s contribution is the auditable evidence architecture, reproducible unimodal baselines, conservative fusion, and a working end-to-end demonstration.”

Do not present the older claimed figures such as 91.4% fused accuracy, 95% crisis F1, 500 expert-rated conversations, or 10,000 learning iterations. Those experiments were not found in the repository and are not supported by artifacts.

## If the professor asks what remains

1. Verify FER2013 source, license, and split provenance.
2. Evaluate fusion on a genuinely aligned multimodal benchmark.
3. Compare SUTRA against late-fusion and learned-fusion baselines with confidence intervals and ablations.
4. Validate safety rules and response quality with qualified human reviewers under an approved protocol.
5. Add consent, retention controls, encryption, accessibility, and deployment monitoring before any real-user study.
