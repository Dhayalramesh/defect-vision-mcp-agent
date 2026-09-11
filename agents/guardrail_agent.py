import os

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.75))

def check_guardrail(prediction: dict) -> dict:
    """
    Decides whether a prediction is safe to auto-accept or needs human review.
    Mirrors the human-in-the-loop pattern from the Ticket Triage project.
    """
    confidence = prediction.get("confidence", 0)

    if confidence < CONFIDENCE_THRESHOLD:
        decision = "FLAGGED_FOR_REVIEW"
        reason = f"Confidence {confidence} below threshold {CONFIDENCE_THRESHOLD}"
    else:
        decision = "AUTO_ACCEPTED"
        reason = f"Confidence {confidence} meets threshold"

    result = {**prediction, "decision": decision, "reason": reason}
    print(f"[GuardrailAgent] {decision}: {reason}")
    return result