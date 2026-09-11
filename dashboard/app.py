import os
import sys
import json
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "mcp_server"))
from tools import get_model_metrics

_real_log = os.path.join(os.path.dirname(__file__), "..", "logs", "predictions.jsonl")
_sample_log = os.path.join(os.path.dirname(__file__), "..", "logs", "predictions_sample.jsonl")
LOG_PATH = _real_log if os.path.exists(_real_log) else _sample_log

st.set_page_config(page_title="Defect Vision Monitor", layout="wide")
st.title("Real-Time Defect Detection — Model Monitoring Dashboard")

if os.path.exists(LOG_PATH):
    with open(LOG_PATH, "r") as f:
        rows = [json.loads(l) for l in f if l.strip()]
else:
    rows = []

total = len(rows)
avg_conf = round(sum(r["confidence"] for r in rows) / total, 4) if total else None
low_conf_count = sum(1 for r in rows if r["confidence"] < float(os.getenv("CONFIDENCE_THRESHOLD", 0.75)))

col1, col2, col3 = st.columns(3)
col1.metric("Total Predictions", total)
col2.metric("Average Confidence", avg_conf if avg_conf is not None else "—")
col3.metric("Low-Confidence Count", low_conf_count)

if rows:
    df = pd.DataFrame(rows)
    st.subheader("Prediction Log")
    st.dataframe(df, use_container_width=True)

    st.subheader("Confidence Over Time")
    st.line_chart(df.set_index("timestamp")["confidence"])
else:
    st.info("No prediction log found yet. Run inference to populate this dashboard.")
