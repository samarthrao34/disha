"""Build the professor-ready DISHA research paper PDF with ReportLab."""

import json
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/DISHA_Genuine_Research_Paper.pdf"


def architecture_drawing():
    width, height = 500, 145
    drawing = Drawing(width, height)
    box_w, box_h = 92, 34
    labels = ["Face\nFER2013", "Speech\nRAVDESS", "Text\nGoEmotions"]
    colors_fill = [colors.HexColor("#E6F0FF"), colors.HexColor("#E8F7F0"), colors.HexColor("#FFF2DD")]
    for index, (label, fill) in enumerate(zip(labels, colors_fill)):
        x = 8 + index * 112
        drawing.add(Rect(x, 98, box_w, box_h, rx=5, ry=5, fillColor=fill, strokeColor=colors.HexColor("#47627A")))
        first, second = label.split("\n")
        drawing.add(String(x + box_w / 2, 117, first, textAnchor="middle", fontName="Helvetica-Bold", fontSize=9))
        drawing.add(String(x + box_w / 2, 105, second, textAnchor="middle", fontName="Helvetica", fontSize=7))
        drawing.add(Line(x + box_w / 2, 98, 174, 75, strokeColor=colors.HexColor("#47627A")))
    drawing.add(Rect(125, 42, 100, 35, rx=5, ry=5, fillColor=colors.HexColor("#EEE9FF"), strokeColor=colors.HexColor("#604A8B")))
    drawing.add(String(175, 62, "Evidence Objects", textAnchor="middle", fontName="Helvetica-Bold", fontSize=9))
    drawing.add(String(175, 50, "quality + uncertainty", textAnchor="middle", fontSize=7))
    drawing.add(Line(225, 59, 270, 59, strokeColor=colors.HexColor("#47627A")))
    drawing.add(Rect(270, 42, 95, 35, rx=5, ry=5, fillColor=colors.HexColor("#F1EBFF"), strokeColor=colors.HexColor("#604A8B")))
    drawing.add(String(317, 62, "SUTRA Fusion", textAnchor="middle", fontName="Helvetica-Bold", fontSize=9))
    drawing.add(String(317, 50, "bounded action", textAnchor="middle", fontSize=7))
    drawing.add(Line(365, 59, 405, 59, strokeColor=colors.HexColor("#47627A")))
    drawing.add(Rect(405, 42, 88, 35, rx=5, ry=5, fillColor=colors.HexColor("#FFE9E9"), strokeColor=colors.HexColor("#9C4A4A")))
    drawing.add(String(449, 62, "Safety + State", textAnchor="middle", fontName="Helvetica-Bold", fontSize=9))
    drawing.add(String(449, 50, "response", textAnchor="middle", fontSize=7))
    return drawing


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(20 * mm, 12 * mm, "DISHA genuine measured-results edition - July 15, 2026")
    canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PaperTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=colors.HexColor("#17324D"), alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(name="PaperAuthor", parent=styles["Normal"], fontSize=10, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#475467"), spaceAfter=18))
    styles.add(ParagraphStyle(name="Abstract", parent=styles["BodyText"], fontSize=9, leading=12, alignment=TA_JUSTIFY, leftIndent=12, rightIndent=12, borderColor=colors.HexColor("#B8C7D9"), borderWidth=0.5, borderPadding=10, backColor=colors.HexColor("#F7FAFC")))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=colors.HexColor("#17324D"), spaceBefore=12, spaceAfter=7))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=colors.HexColor("#28577E"), spaceBefore=9, spaceAfter=5))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9, leading=12.5, alignment=TA_JUSTIFY, spaceAfter=6))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.5, leading=10, spaceAfter=4))
    styles.add(ParagraphStyle(name="Callout", parent=styles["BodyText"], fontSize=9, leading=12, textColor=colors.HexColor("#7A2E0B"), backColor=colors.HexColor("#FFF5ED"), borderColor=colors.HexColor("#FDBA74"), borderWidth=0.6, borderPadding=8, spaceBefore=6, spaceAfter=8))

    face = json.loads((ROOT / "experiments/face_resnet18_fer2013_metrics.json").read_text())
    speech = json.loads((ROOT / "experiments/speech_ravdess_mfcc_svm_metrics.json").read_text())
    text = json.loads((ROOT / "experiments/text_goemotions_tfidf_sgd_metrics.json").read_text())
    integrated = json.loads((ROOT / "experiments/integrated_demo_latency_metrics.json").read_text())

    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=18 * mm, bottomMargin=20 * mm, title="DISHA: An Evidence-Governed Multimodal Emotion-Aware Research Prototype", author="Samarth Rao")
    story = []
    story.append(Paragraph("DISHA: An Evidence-Governed Multimodal Emotion-Aware Research Prototype", styles["PaperTitle"]))
    story.append(Paragraph("Samarth Rao<br/>Independent research prototype - July 15, 2026", styles["PaperAuthor"]))
    abstract = (
        "<b>Abstract -</b> This paper presents DISHA, a reproducible research prototype for multimodal emotion-aware interaction. "
        "Separately evaluated face, speech, and text classifiers emit a common Evidence Object, followed by reliability-weighted SUTRA fusion, session tracking, and bounded safety routing. "
        f"Measured results are: FER2013 face accuracy {face['accuracy']*100:.2f}% (macro-F1 {face['macro_f1']:.4f}); "
        f"speaker-independent RAVDESS speech accuracy {speech['accuracy']*100:.2f}% (macro-F1 {speech['macro_f1']:.4f}); and "
        f"GoEmotions multilabel text micro-F1 {text['test_metrics']['micro_f1']:.4f} (macro-F1 {text['test_metrics']['macro_f1']:.4f}). "
        f"The integrated warm pipeline averaged {integrated['warm_latency_mean_ms']:.2f} ms over 20 runs. "
        "No fused-accuracy or clinical-safety claim is made because no aligned multimodal or clinically annotated evaluation set was used."
    )
    story.append(Paragraph(abstract, styles["Abstract"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Keywords: multimodal emotion recognition, affective computing, quality-weighted fusion, FER2013, RAVDESS, GoEmotions, reproducibility", styles["Smallx"]))
    story.append(Paragraph("1. Introduction", styles["H1x"]))
    for paragraph in [
        "Emotion recognition systems commonly report a label from a single sensor. A conversational system requires additional machinery: uncertain inputs must be identified, multiple modalities must be reconciled, state must persist across turns, and response behavior must remain bounded. DISHA explores this systems problem through a modular research implementation.",
        "The research question is not whether a prototype can claim one high accuracy number. Face, speech, and text benchmarks have different label spaces and data-generating processes, so a fused score is invalid without an aligned multimodal corpus. Instead, this work asks whether independently tested components can communicate through a transparent evidence contract and support a measurable end-to-end demo without hiding uncertainty.",
    ]:
        story.append(Paragraph(paragraph, styles["Bodyx"]))
    contributions = [
        "Canonical Evidence Object excluding raw sensor content and separating input reliability from predictive uncertainty.",
        "Three reproducible baselines trained or evaluated on public datasets.",
        "Implemented quality-weighted late fusion, session smoothing, bounded response actions, and crisis-indicator override.",
        "Executed metrics, dataset hashes, confusion matrices, serialized models, and a desktop demonstration.",
    ]
    story.append(Paragraph("<b>Implemented contributions</b>", styles["Bodyx"]))
    for idx, item in enumerate(contributions, 1):
        story.append(Paragraph(f"{idx}. {item}", styles["Bodyx"]))
    story.append(Paragraph("DISHA is not presented as a therapist, diagnostic system, medical device, or validated crisis service.", styles["Callout"]))

    story.append(Paragraph("2. Architecture", styles["H1x"]))
    story.append(architecture_drawing())
    story.append(Paragraph("Figure 1. Implemented evidence-governed late-fusion pipeline.", styles["Smallx"]))
    story.append(Paragraph("Each modality emits normalized emotion probabilities, optional calibrated confidence, normalized entropy, input reliability, availability, timestamp, quality metadata, and model identity. Raw pixels, waveforms, and message text are excluded from the object passed to SUTRA.", styles["Bodyx"]))
    story.append(Paragraph("For modality i, the implemented heuristic uses weight w_i = r_i c_i, where r_i is input reliability and c_i is calibrated confidence when available (otherwise 1). The fused probability is p(e) = sum_i w_i p_i(e) / sum_i w_i. This is inspectable late fusion, not a trained cross-modal model.", styles["Bodyx"]))
    story.append(Paragraph("Session state uses exponential smoothing (alpha=0.35). A deterministic crisis phrase scan can override ordinary emotion routing. This is an integration safeguard and has not been clinically validated.", styles["Bodyx"]))

    story.append(Paragraph("3. Experimental Method", styles["H1x"]))
    methods = [
        ("3.1 Face - FER2013", "The local layout contains 28,709 training and 7,178 test images across seven classes. Ten percent of the training directory was selected with seed 42 for validation. An ImageNet-initialized ResNet-18 was fine-tuned for 15 epochs; epoch 14 was retained by validation macro-F1. Temperature scaling was fitted on validation logits. Test metrics include accuracy, macro/weighted F1, 15-bin ECE, and synchronized CUDA latency."),
        ("3.2 Speech - RAVDESS", "The official 208.5 MB RAVDESS Audio Speech archive was downloaded from Zenodo and matched MD5 bc696df654c87fed845eb13823edef8a. All 1,440 utterances from 24 actors were used. Audio was resampled to 16 kHz and trimmed. Forty MFCCs plus delta and delta-delta coefficients were summarized by mean and standard deviation (240 features). A standardized balanced RBF SVM (C=10) was evaluated with six-fold GroupKFold by actor; no actor crossed folds."),
        ("3.3 Text - GoEmotions", "Official simplified train, validation, and test TSV files were downloaded from Google Research and SHA-256 hashes recorded. TF-IDF word unigrams and bigrams (58,340 retained features) were classified by 28 one-vs-rest SGD classifiers with logistic loss. The multilabel threshold was tuned only on validation micro-F1 and fixed for test evaluation."),
        ("3.4 Integrated latency", "One real positive sample from each independent dataset and the message 'I am happy and excited today' were used for a systems benchmark. After one cold start, 20 warm iterations were measured in one process on an RTX 3050 6 GB Laptop GPU. These samples are not aligned and therefore cannot estimate fused accuracy."),
    ]
    for heading, body in methods:
        story.append(Paragraph(heading, styles["H2x"]))
        story.append(Paragraph(body, styles["Bodyx"]))

    story.append(Paragraph("4. Measured Results", styles["H1x"]))
    table_data = [
        ["Component", "Protocol", "Accuracy", "Macro-F1", "Other", "Latency"],
        ["Face ResNet-18", "FER2013 test", f"{face['accuracy']*100:.2f}%", f"{face['macro_f1']:.4f}", f"ECE {face['ece_uncalibrated']:.4f} -> {face['ece_calibrated']:.4f}", f"{face['latency_ms_per_image']:.2f} ms"],
        ["Speech MFCC+SVM", "6-fold actor groups", f"{speech['accuracy']*100:.2f}%", f"{speech['macro_f1']:.4f}", f"weighted {speech['weighted_f1']:.4f}", f"{speech['latency_ms_per_file']:.2f} ms"],
        ["Text TF-IDF+SGD", "official test", "N/A", f"{text['test_metrics']['macro_f1']:.4f}", f"micro {text['test_metrics']['micro_f1']:.4f}", f"{text['latency_ms_per_text']:.2f} ms"],
        ["Integrated", "20 warm runs", "N/A", "N/A", f"median {integrated['warm_latency_median_ms']:.2f} ms", f"{integrated['warm_latency_mean_ms']:.2f} ms"],
    ]
    table = Table(table_data, colWidths=[27*mm, 31*mm, 19*mm, 20*mm, 39*mm, 24*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17324D")), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 7), ("LEADING", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#B8C7D9")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("ALIGN", (2,1), (-1,-1), "CENTER"),
        ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"GoEmotions subset accuracy was {text['test_metrics']['subset_accuracy']*100:.2f}%, micro precision {text['test_metrics']['micro_precision']:.4f}, micro recall {text['test_metrics']['micro_recall']:.4f}, and Hamming loss {text['test_metrics']['hamming_loss']:.5f}. The face checkpoint contains {face['num_params']:,} parameters. Speech model size is {speech['model_size_mb']:.2f} MB and text model size is {text['model_size_mb']:.2f} MB.", styles["Bodyx"]))
    story.append(Paragraph(f"Cold start was {integrated['cold_start_ms']/1000:.2f} seconds, dominated by imports and model initialization. Warm latency ranged from {integrated['warm_latency_min_ms']:.2f} to {integrated['warm_latency_max_ms']:.2f} ms; measured p95 was {integrated['warm_latency_p95_ms']:.2f} ms.", styles["Bodyx"]))
    story.append(Paragraph("The demo's three selected positive examples yielded fused happy evidence 0.9274. This is one interaction, not an accuracy measurement.", styles["Callout"]))

    images = [
        (ROOT / "experiments/face_resnet18_fer2013_confusion_matrix.png", "Figure 2. FER2013 held-out test confusion matrix."),
        (ROOT / "experiments/speech_ravdess_mfcc_svm_confusion_matrix.png", "Figure 3. RAVDESS six-fold speaker-independent confusion matrix."),
    ]
    for path, caption in images:
        if path.is_file():
            story.append(PageBreak())
            story.append(Image(str(path), width=150*mm, height=132*mm))
            story.append(Paragraph(caption, styles["Smallx"]))

    story.append(PageBreak())
    story.append(Paragraph("5. Interpretation", styles["H1x"]))
    for paragraph in [
        "Face calibration improved after temperature scaling, but ECE depends on binning and does not establish reliability outside FER2013. Happy and surprise were the strongest face classes; fear and sadness remained difficult.",
        "The speaker-independent speech score is lower than values commonly obtained with random utterance splits. Grouping by actor is stricter and prevents identity cues from leaking across train and evaluation folds. The confusion matrix shows substantial overlap among calm, sad, neutral, and other low-arousal categories.",
        "The text baseline is deliberately compact and linear. Its moderate macro-F1 reflects the difficulty and imbalance of 28-label emotion classification. It nevertheless supplies a real trained model for the integrated prototype within a 12.43 MB artifact.",
        "Independent component results cannot support a 91% fused-accuracy claim. A defensible fused evaluation requires an aligned corpus containing face, speech, text, and shared labels for the same observations.",
    ]:
        story.append(Paragraph(paragraph, styles["Bodyx"]))

    story.append(Paragraph("6. Safety, Privacy, and Ethics", styles["H1x"]))
    story.append(Paragraph("The prototype uses bounded templates rather than unconstrained neural response generation. Explicit crisis phrases trigger a conservative emergency-support message independently of emotion output. No crisis dataset or expert annotation was evaluated, so sensitivity, specificity, F1, and clinical utility are unknown.", styles["Bodyx"]))
    story.append(Paragraph("Evidence Objects omit raw inputs, but the demo is not a complete privacy architecture. It has no authentication, encrypted persistent database, consent workflow, retention controls, human escalation service, or clinical oversight. Real-user deployment would be inappropriate without those controls and prospective safety evaluation.", styles["Bodyx"]))
    story.append(Paragraph("RAVDESS is CC BY-NC-SA 4.0. Google Research states CC BY 4.0 for repository datasets. FER2013 provenance for the local copy still requires formal recording before redistribution.", styles["Bodyx"]))

    story.append(Paragraph("7. Limitations", styles["H1x"]))
    limitations = [
        "Different datasets and label taxonomies; seven-label projection is manual and unvalidated.",
        "FER2013 labels are noisy; RAVDESS is acted North American English; GoEmotions is English Reddit text.",
        "No aligned multimodal accuracy, human conversation study, crisis benchmark, clinical trial, RL experiment, or compression study.",
        "Fusion thresholds and reliability formulas are research defaults; face detection may use a center-crop fallback.",
        "Long cold start in the current Python environment.",
    ]
    for item in limitations:
        story.append(Paragraph(f"- {item}", styles["Bodyx"]))

    story.append(Paragraph("8. Conclusion", styles["H1x"]))
    story.append(Paragraph("DISHA demonstrates a functioning path from three real-data emotion classifiers through reliability-weighted fusion, session tracking, safety routing, and bounded response generation. Its measured results are moderate and its limitations substantial, but every number in this paper is tied to an executed artifact. The prototype is a defensible baseline for aligned multimodal evaluation, trained fusion, stronger calibrated models, culturally diverse validation, and expert-reviewed safety research.", styles["Bodyx"]))

    story.append(Paragraph("References", styles["H1x"]))
    refs = [
        "[1] T. Baltrusaitis, C. Ahuja, and L.-P. Morency, Multimodal Machine Learning: A Survey and Taxonomy, IEEE TPAMI, 2019.",
        "[2] I. J. Goodfellow et al., Challenges in Representation Learning: A Report on Three Machine Learning Contests, ICML Workshop, 2013.",
        "[3] S. R. Livingstone and F. A. Russo, The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), PLOS ONE 13(5), 2018. doi:10.1371/journal.pone.0196391.",
        "[4] D. Demszky et al., GoEmotions: A Dataset of Fine-Grained Emotions, ACL, 2020. Dataset: github.com/google-research/google-research/tree/master/goemotions.",
        "[5] J. Garcia and F. Fernandez, A Comprehensive Survey on Safe Reinforcement Learning, JMLR 16, 2015.",
        "[6] DISHA source code and experiment artifacts, local repository state dated July 15, 2026.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, styles["Smallx"]))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
