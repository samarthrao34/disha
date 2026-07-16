"""Runnable integrated DISHA research demo."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List

from disha.conversation import render_response
from disha.emotion_state import EmotionTracker
from disha.evidence.schema import (
    AvailabilityStatus,
    EvidenceObject,
    Modality,
    ReliabilityStatus,
)
from disha.safety.policy import SafetyPolicy
from disha.sutra.pipeline import SutraReasoner

EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")


@dataclass(frozen=True)
class DishaResult:
    response: str
    action: str
    safety: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    reasoning_trace: Dict[str, Any]
    session_state: Dict[str, Any] | None
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _unavailable_evidence(
    modality: Modality,
    *,
    reason: str,
    detail: str,
    timestamp: float | None = None,
) -> EvidenceObject:
    uniform = {emotion: 1.0 / len(EMOTIONS) for emotion in EMOTIONS}
    return EvidenceObject(
        modality=modality,
        emotion_probabilities=uniform,
        calibrated_confidence=None,
        predictive_entropy=1.0,
        reliability_score=0.0,
        reliability_status=ReliabilityStatus.UNUSABLE,
        availability_status=AvailabilityStatus.UNAVAILABLE,
        timestamp=time.time() if timestamp is None else timestamp,
        quality_metadata={
            "failure_reason": reason,
            "failure_detail": detail,
            "thresholds": "research_defaults",
        },
        model_name="unavailable_modality_placeholder",
    )


class DishaEngine:
    def __init__(self, *, continue_on_modality_error: bool = True) -> None:
        self.reasoner = SutraReasoner()
        self.safety = SafetyPolicy()
        self.tracker = EmotionTracker()
        self.continue_on_modality_error = continue_on_modality_error

    def _append_or_placeholder(
        self,
        evidence: List[EvidenceObject],
        modality: Modality,
        builder: Callable[[], EvidenceObject],
    ) -> None:
        try:
            evidence.append(builder())
        except Exception as exc:
            if not self.continue_on_modality_error:
                raise
            evidence.append(
                _unavailable_evidence(
                    modality,
                    reason=exc.__class__.__name__,
                    detail=str(exc),
                )
            )

    def process(
        self,
        *,
        text: str | None = None,
        image_path: str | None = None,
        audio_path: str | None = None,
    ) -> DishaResult:
        started = time.perf_counter()
        evidence: List[EvidenceObject] = []
        if text and text.strip():
            from disha.text.evidence import build_text_evidence

            evidence.append(build_text_evidence(text))
        if image_path:
            from disha.face.inference import build_face_evidence_from_file

            self._append_or_placeholder(
                evidence,
                Modality.FACE,
                lambda: build_face_evidence_from_file(image_path),
            )
        if audio_path:
            from disha.speech.evidence import build_speech_evidence

            self._append_or_placeholder(
                evidence,
                Modality.SPEECH,
                lambda: build_speech_evidence(audio_path),
            )

        decision = self.reasoner.decide(evidence)
        crisis_level = str(decision.reasoning_trace.get("crisis_level", "low"))
        safety = self.safety.approve_action(decision.action, crisis_level=crisis_level)
        fused = decision.reasoning_trace.get("fused_probabilities")
        state = self.tracker.update(fused) if fused else None
        return DishaResult(
            response=render_response(safety.action),
            action=safety.action.value,
            safety={"approved": safety.approved, "reason": safety.reason},
            evidence=[item.to_dict() for item in evidence],
            reasoning_trace=decision.reasoning_trace,
            session_state=asdict(state) if state else None,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def process_text(self, text: str) -> DishaResult:
        return self.process(text=text)

    def reset_session(self) -> None:
        self.tracker = EmotionTracker()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DISHA research demo")
    parser.add_argument("--text")
    parser.add_argument("--image")
    parser.add_argument("--audio")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="raise modality errors instead of continuing with unavailable evidence",
    )
    args = parser.parse_args()
    if not any((args.text, args.image, args.audio)):
        parser.error("provide at least one of --text, --image, or --audio")
    result = DishaEngine(continue_on_modality_error=not args.fail_fast).process(
        text=args.text, image_path=args.image, audio_path=args.audio
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
