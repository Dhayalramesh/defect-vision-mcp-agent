import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "mcp_server"))
from tools import get_model_metrics

DRIFT_LOW_CONF_RATIO_THRESHOLD = 0.3

def check_drift() -> dict:
    """
    Simple drift signal: if too high a proportion of recent predictions
    are low-confidence, flag the model for review/retrain.
    """
    metrics = get_model_metrics()
    total = metrics.get("total_predictions", 0)
    low_conf = metrics.get("low_confidence_count", 0)

    if total == 0:
        return {"status": "NO_DATA", "metrics": metrics}

    ratio = low_conf / total
    status = "DRIFT_DETECTED" if ratio > DRIFT_LOW_CONF_RATIO_THRESHOLD else "HEALTHY"

    result = {"status": status, "low_confidence_ratio": round(ratio, 4), "metrics": metrics}
    print(f"[DriftMonitorAgent] {status} (low_conf_ratio={ratio:.2f})")
    return result