"""Streamlit frontend for the AskMamma learning project."""

from __future__ import annotations

import uuid

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="AskMamma Agent", layout="wide")
st.title("AskMamma Agent Studio")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []


def api_get(path: str, **params):
    response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict):
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


try:
    dashboard = api_get("/dashboard")
except Exception as exc:
    st.error(f"Backend is not reachable at {API_BASE_URL}: {exc}")
    st.stop()

graph_info = api_get("/agent/graph")
semantic = api_get("/memory/semantic").get("records", [])
audit = api_get("/memory/audit").get("records", [])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sample Items", dashboard["total_products"])
col2.metric("Low Availability", dashboard["low_stock_products"])
col3.metric("Unavailable", dashboard["out_of_stock_products"])
high_demand = dashboard.get("predicted_high_demand_products", [])
col4.metric("Forecast Signals", len(high_demand))

left, right = st.columns([1.1, 1.2])

with left:
    st.subheader("Inventory Workspace")
    search = st.text_input("Search sample demo items", "")
    button_cols = st.columns(5)
    if button_cols[0].button("Refresh data"):
        st.rerun()
    if button_cols[1].button("Generate report"):
        report = api_get("/reports/askmamma")
        st.success(report["summary"])
    if button_cols[2].button("Run demand forecast"):
        forecast = api_post("/demo/forecast", {"months": 6})
        st.info(forecast.get("explanation", forecast.get("message")))
    if button_cols[3].button("View traces"):
        st.session_state.show_traces = not st.session_state.get("show_traces", False)
    if button_cols[4].button("Ask AI"):
        st.session_state.focus_chat = True

    products = api_get("/demo/items", search=search or None, limit=100)
    if products:
        df = pd.DataFrame(products)
        visible = [
            "sku",
            "name",
            "category",
            "supplier_name",
            "price",
            "stock_quantity",
            "reorder_level",
            "location",
        ]
        st.dataframe(df[[column for column in visible if column in df.columns]], use_container_width=True)
    else:
        st.caption("No sample demo items found.")

    st.subheader("Forecast Snapshot")
    if high_demand:
        chart_df = pd.DataFrame(high_demand)
        if {"name", "sold"}.issubset(chart_df.columns):
            st.bar_chart(chart_df.set_index("name")["sold"])

    st.subheader("LangGraph")
    st.code(graph_info["graph"], language="mermaid")

with right:
    st.subheader("Agent Console")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about demo items, availability, partners, forecasts, documents, or reports")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                result = api_post(
                    "/agent/chat",
                    {"message": prompt, "session_id": st.session_state.session_id},
                )
                answer = result["answer"]
                st.markdown(answer)
                info_cols = st.columns(3)
                info_cols[0].metric("Selected Agent", result.get("selected_agent", "n/a"))
                info_cols[1].metric("Tools", len(result.get("tools_called", [])))
                info_cols[2].metric("Latency ms", result.get("response_time_ms", 0))
                with st.expander("Agent Activity", expanded=True):
                    st.write(result.get("route_path", []))
                    st.json(result.get("intermediate_steps", []))
                with st.expander("Tool Activity"):
                    st.code(", ".join(result.get("tools_called", [])) or "No tools used")
                with st.expander("Report Bundle"):
                    st.json(result.get("report_bundle", {}))
            except Exception as exc:
                answer = f"Sorry, I could not reach the agent: {exc}"
                st.error(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    st.subheader("Memory Viewer")
    tabs = st.tabs(["Semantic", "Audit", "Traces"])
    with tabs[0]:
        st.dataframe(pd.DataFrame(semantic), use_container_width=True)
    with tabs[1]:
        st.dataframe(pd.DataFrame(audit), use_container_width=True)
    with tabs[2]:
        traces = api_get("/agent/traces", limit=10)
        st.dataframe(pd.DataFrame(traces), use_container_width=True)

    st.subheader("MCP Viewer")
    mcp_cols = st.columns(3)
    mcp_cols[0].metric("Tools", len(api_get("/mcp/tools")))
    mcp_cols[1].metric("Resources", len(api_get("/mcp/resources")))
    mcp_cols[2].metric("Prompts", len(api_get("/mcp/prompts")))
