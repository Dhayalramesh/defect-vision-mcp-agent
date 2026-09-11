# Defect Vision MCP Agent

Real-time steel surface defect classification system combining a fine-tuned
PyTorch computer vision model, a genuine MCP (Model Context Protocol) server,
and a LangGraph multi-agent monitoring layer with human-in-the-loop guardrails.

**Live Demo:** https://defect-vision-mcp-agent-j3eq5p8huviz4hbz495imi.streamlit.app/

Built to close three specific gaps: hands-on PyTorch model training, real MCP
protocol integration, and MLOps-style model monitoring — all wired together
as one coherent system rather than three disconnected pieces.

## Results

Fine-tuned a ResNet18-based classifier (transfer learning, frozen backbone,
trained classification head) on the [NEU-DET surface defect dataset](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
— 6 classes, 1,800 images (1,440 train / 360 validation, balanced).

| Metric | Score |
|---|---|
| Validation Accuracy | 98.89% |
| Weighted Precision | 0.9892 |
| Weighted Recall | 0.9889 |
| Weighted F1 | 0.9888 |

**Per-class performance:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| crazing | 0.98 | 1.00 | 0.99 |
| inclusion | 1.00 | 0.95 | 0.97 |
| patches | 1.00 | 0.98 | 0.99 |
| pitted_surface | 0.97 | 1.00 | 0.98 |
| rolled-in_scale | 1.00 | 1.00 | 1.00 |
| scratches | 0.98 | 1.00 | 0.99 |

## Architecture

1. **PyTorch Model** (`model/`) — ResNet18-based transfer-learning classifier.
   Backbone frozen; only the classification head is fine-tuned, converging
   in 6 epochs on CPU.

2. **MCP Server** (`mcp_server/`) — a real, protocol-compliant MCP server
   (built on `MCPServer` from the official `mcp` SDK, verified against the
   MCP Inspector) exposing two tools:
   - `classify_image(image_path)` — runs inference, logs the prediction
   - `get_model_metrics()` — returns aggregate stats from the prediction log

3. **Multi-Agent Layer** (`agents/`) — LangGraph-orchestrated pipeline with
   three nodes in sequence:
   - **Inference agent** — calls the MCP tool to classify an image
   - **Guardrail agent** — auto-accepts high-confidence predictions;
     low-confidence predictions are flagged for human review instead of
     being silently accepted, mirroring a human-in-the-loop pattern rather
     than blind automation
   - **Drift-monitor agent** — tracks the rolling ratio of low-confidence
     predictions and flags `DRIFT_DETECTED` if it crosses a threshold,
     signaling the model may need review or retraining

4. **Dashboard** (`dashboard/`) — Streamlit UI showing live prediction
   counts, average confidence, low-confidence count, a full prediction log,
   and a confidence-over-time chart.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your own GROQ_API_KEY
```

## Train the model

Requires the NEU-DET dataset arranged as:

```
data/processed/
├── train/<class_name>/*.jpg
└── val/<class_name>/*.jpg
```

```bash
python model/train.py
python model/evaluate.py
```

## Run the MCP server

```bash
python mcp_server/server.py
```

Test it interactively with the [MCP Inspector](https://github.com/modelcontextprotocol/inspector):

```bash
npx @modelcontextprotocol/inspector python mcp_server/server.py
```

## Run the agent pipeline

```bash
python agents/graph.py
```

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

## Run tests

```bash
pytest tests/ -v
```

## Tech Stack

Python · PyTorch · torchvision · MCP SDK · LangGraph · LangChain · Streamlit ·
scikit-learn · GitHub Actions (CI)

## Notes

- `data/processed/`, `model/checkpoints/*.pt`, and the live prediction log
  (`logs/predictions.jsonl`) are gitignored — not included in this repo.
  `logs/predictions_sample.jsonl` is committed separately so the deployed
  dashboard has sample data to display.
- Deployed on Streamlit Community Cloud (free tier) — expect a brief cold
  start if the app has been idle.
