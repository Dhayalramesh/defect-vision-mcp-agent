import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "mcp_server"))
from tools import classify_image

def run_inference(image_path: str) -> dict:
    """Calls the classifier and returns the raw prediction result."""
    result = classify_image(image_path)
    print(f"[InferenceAgent] {json.dumps(result)}")
    return result