"""Streamlit frontend for the inventory management agent system."""

from __future__ import annotations

import uuid

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="AskMamma Inventory", layout="wide")
st.title("AskMamma Inventory Assistant")

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

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Products", dashboard["total_products"])
col2.metric("Low Stock", dashboard["low_stock_products"])
col3.metric("Out of Stock", dashboard["out_of_stock_products"])
high_demand = dashboard.get("predicted_high_demand_products", [])
col4.metric("High Demand Signals", len(high_demand))

left, right = st.columns([1.25, 1])

with left:
    st.subheader("Products")
    search = st.text_input("Search products", "")
    button_cols = st.columns(5)
    if button_cols[0].button("Refresh inventory"):
        st.rerun()
    if button_cols[1].button("Generate stock report"):
        report = api_get("/reports/inventory")
        st.success(report["summary"])
    if button_cols[2].button("Run demand forecast"):
        forecast = api_post("/forecast/demand", {"months": 6})
        st.info(forecast.get("explanation", forecast.get("message")))
    if button_cols[3].button("View traces"):
        st.session_state.show_traces = not st.session_state.get("show_traces", False)
    if button_cols[4].button("Ask AI"):
        st.session_state.focus_chat = True

    products = api_get("/products", search=search or None, limit=100)
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
        st.caption("No products found.")

    if st.session_state.get("show_traces", False):
        st.subheader("Recent Agent Traces")
        traces = api_get("/agent/traces", limit=10)
        for trace in traces:
            with st.expander(f"{trace['created_at']} - {trace['selected_agent']}"):
                st.write(trace["user_input"])
                st.code(trace["tools_called"])
                st.write(trace["final_answer"])

with right:
    st.subheader("Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about stock, suppliers, forecasts, documents, or reports")
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
                if result.get("tools_called"):
                    with st.expander("Tools used"):
                        st.write(result["selected_agent"])
                        st.code(", ".join(result["tools_called"]))
            except Exception as exc:
                answer = f"Sorry, I could not reach the agent: {exc}"
                st.error(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
