from disha.evidence.schema import EvidenceObject, Modality, ReliabilityStatus, AvailabilityStatus

def test_evidence_object_usable():
    e = EvidenceObject(
        modality=Modality.TEXT,
        emotion_probabilities={"neutral": 0.7, "sad": 0.3},
        calibrated_confidence=0.7,
        predictive_entropy=0.5,
        reliability_score=0.9,
        reliability_status=ReliabilityStatus.RELIABLE,
        availability_status=AvailabilityStatus.AVAILABLE,
        timestamp=0.0,
    )
    assert e.is_usable()
    assert e.predicted_emotion == "neutral"
    assert e.to_dict()["modality"] == "text"
