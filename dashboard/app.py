import os
import sys
import json
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "mcp_server"))
from tools import get_model_metrics

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "predictions.jsonl")

st.set_page_config(page_title="Defect Vision Monitor", layout="wide")
st.title("Real-Time Defect Detection — Model Monitoring Dashboard")

metrics = get_model_metrics()

col1, col2, col3 = st.columns(3)
col1.metric("Total Predictions", metrics.get("total_predictions", 0))
col2.metric("Average Confidence", metrics.get("average_confidence", "N/A"))
col3.metric("Low-Confidence Count", metrics.get("low_confidence_count", 0))

if os.path.exists(LOG_PATH):
    with open(LOG_PATH, "r") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if rows:
        df = pd.DataFrame(rows)
        st.subheader("Prediction Log")
        st.dataframe(df, use_container_width=True)

        st.subheader("Confidence Over Time")
        st.line_chart(df.set_index("timestamp")["confidence"])
    else:
        st.info("No predictions logged yet.")
else:
    st.info("No prediction log found yet. Run inference to populate this dashboard.")