import os
import json
import time
import torch
from PIL import Image
from torchvision import transforms
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))
from model import load_model

CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT_PATH", "model/checkpoints/defect_model.pt")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "predictions.jsonl")

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_class_names = None

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _ensure_model_loaded():
    global _model, _class_names
    if _model is None:
        # num_classes gets overwritten by the checkpoint's class list length
        state = torch.load(CHECKPOINT_PATH, map_location=_device)
        _class_names = state["class_names"]
        _model, _ = load_model(CHECKPOINT_PATH, num_classes=len(_class_names), device=_device)


def classify_image(image_path: str) -> dict:
    """Runs the defect classifier on a single image and logs the prediction."""
    _ensure_model_loaded()

    image = Image.open(image_path).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        outputs = _model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    result = {
        "image_path": image_path,
        "predicted_class": _class_names[pred_idx.item()],
        "confidence": round(confidence.item(), 4),
        "timestamp": time.time(),
    }

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")

    return result


def get_model_metrics() -> dict:
    """Returns basic stats derived from logged predictions."""
    if not os.path.exists(LOG_PATH):
        return {"total_predictions": 0, "average_confidence": None}

    with open(LOG_PATH, "r") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    if not lines:
        return {"total_predictions": 0, "average_confidence": None}

    avg_conf = sum(l["confidence"] for l in lines) / len(lines)
    low_conf_count = sum(1 for l in lines if l["confidence"] < float(os.getenv("CONFIDENCE_THRESHOLD", 0.75)))

    return {
        "total_predictions": len(lines),
        "average_confidence": round(avg_conf, 4),
        "low_confidence_count": low_conf_count,
    }