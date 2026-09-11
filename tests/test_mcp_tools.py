import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "mcp_server"))
from tools import get_model_metrics

def test_get_model_metrics_returns_dict():
    result = get_model_metrics()
    assert isinstance(result, dict)
    assert "total_predictions" in result

def test_get_model_metrics_zero_state(tmp_path, monkeypatch):
    fake_log = tmp_path / "predictions.jsonl"
    import tools
    monkeypatch.setattr(tools, "LOG_PATH", str(fake_log))
    result = get_model_metrics()
    assert result["total_predictions"] == 0