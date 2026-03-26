import streamlit as st
import os
from agent import get_agent

st.set_page_config(page_title="Kumo | Agentic PQL Demo", layout="wide", page_icon="☁️")

st.title("☁️ Kumo-Style Agentic Supply Chain")
st.markdown("""
This application demonstrates **Agentic Predictive Query Language (PQL)** powered by **Relational Foundation Models**.
Instead of manual feature engineering, the underlying database (Suppliers, Warehouses, Products, Shipments) is modeled as a massive **Heterogeneous Graph** using PyTorch Geometric (PyG).
You can ask the AI structural questions (e.g. *"Will product P_40 be delayed shipping to warehouse W_5?"*). The LangChain Agent translates your intent, queries the PyG Graph Transformer, and returns a Zero-Shot/Glass-Box prediction.
""")

st.sidebar.success("✅ Operating in Full Local Mode (No API Key Required)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("E.g. What is the probability of delay for product P_12 to warehouse W_3?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Translating natural language to Graph PQL..."):
            try:
                # API key is now bypassed in agent.py
                agent = get_agent()
                response = agent.invoke({"input": prompt})
                ans = response["output"]
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
            except Exception as e:
                st.error(f"Error executing graph query: {str(e)}")
