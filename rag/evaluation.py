import streamlit as st
import pandas as pd
import json
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
METRICS_FILE = PROJECT_ROOT / "rag_metrics_optimized.jsonl"

def render_evaluation_dashboard():
    st.title("RAG Evaluation")
    st.subheader("Current Baseline")
    st.caption("Metrics are based only on successfully recorded queries. No ground-truth answers are evaluated yet.")

    if not os.path.exists(METRICS_FILE):
        st.info("No metrics data found yet. Please ask some questions in the Chat Assistant first.")
        return

    # Load data
    data = []
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                data.append(record)
            except:
                continue

    if not data:
        st.info("No successful queries recorded yet.")
        return

    df = pd.DataFrame(data)

    # Calculate metrics
    total_queries = len(df)
    
    avg_total_latency = df["total_latency_ms"].mean()
    p50_total_latency = np.percentile(df["total_latency_ms"], 50)
    p95_total_latency = np.percentile(df["total_latency_ms"], 95)

    avg_retrieval_latency = df["retrieval_latency_ms"].mean()
    p50_retrieval_latency = np.percentile(df["retrieval_latency_ms"], 50)
    p95_retrieval_latency = np.percentile(df["retrieval_latency_ms"], 95)

    avg_llm_latency = df["llm_latency_ms"].mean()
    p50_llm_latency = np.percentile(df["llm_latency_ms"], 50)
    p95_llm_latency = np.percentile(df["llm_latency_ms"], 95)

    avg_prompt_tokens = df["estimated_final_prompt_tokens"].mean()
    avg_retrieved_chunks = df["retrieved_chunks"].mean()
    avg_selected_chunks = df["selected_chunks"].mean()

    # Display Metrics
    st.header("Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Successful Queries", total_queries)
    col2.metric("Avg Total Latency (ms)", f"{avg_total_latency:.0f}")
    col3.metric("P50 Total Latency (ms)", f"{p50_total_latency:.0f}")
    col4.metric("P95 Total Latency (ms)", f"{p95_total_latency:.0f}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Prompt Tokens", f"{avg_prompt_tokens:.0f}")
    col2.metric("Avg Retrieval Latency", f"{avg_retrieval_latency:.0f}")
    col3.metric("P50 Retrieval Latency", f"{p50_retrieval_latency:.0f}")
    col4.metric("P95 Retrieval Latency", f"{p95_retrieval_latency:.0f}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Retrieved Chunks", f"{avg_retrieved_chunks:.1f}")
    col2.metric("Avg LLM Latency", f"{avg_llm_latency:.0f}")
    col3.metric("P50 LLM Latency", f"{p50_llm_latency:.0f}")
    col4.metric("P95 LLM Latency", f"{p95_llm_latency:.0f}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Selected Chunks", f"{avg_selected_chunks:.1f}")

    st.divider()

    # Charts
    st.header("Latency Trends")
    
    chart_data = df[["total_latency_ms", "retrieval_latency_ms", "llm_latency_ms", "estimated_final_prompt_tokens"]].copy()
    
    st.subheader("Total Latency per Query")
    st.line_chart(chart_data["total_latency_ms"])

    st.subheader("Retrieval Latency per Query")
    st.line_chart(chart_data["retrieval_latency_ms"])

    st.subheader("LLM Latency per Query")
    st.line_chart(chart_data["llm_latency_ms"])

    st.subheader("Prompt Tokens per Query")
    st.line_chart(chart_data["estimated_final_prompt_tokens"])

    st.divider()

    st.header("Query Logs")
    display_df = df[[
        "timestamp", "question", "retrieval_latency_ms", "context_building_latency_ms", 
        "llm_latency_ms", "total_latency_ms", "retrieved_chunks", "selected_chunks", 
        "estimated_final_prompt_tokens"
    ]].copy()
    st.dataframe(display_df, use_container_width=True)

    st.divider()
    
    with st.expander("Danger Zone"):
        st.warning("Are you sure you want to delete all recorded metrics? This cannot be undone.")
        confirm = st.checkbox("I understand the consequences, delete the data")
        if confirm:
            if st.button("Delete Data", type="primary"):
                try:
                    os.remove(METRICS_FILE)
                    st.success("Data cleared successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing data: {e}")
