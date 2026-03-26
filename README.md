# Agentic Predictive Query Language (PQL) via Relational Foundation Models

This project simulates the core vision of **Kumo.ai**: enabling "Zero-Shot" or "Glass-Box" predictions directly on relational databases without manual feature engineering, fronted by an **Agentic AI** capable of answering declarative business questions. 

## 🧠 The Architecture

Unlike traditional ML pipelines that flatten tabular data into single wide matrices (destroying relational context), this architecture preserves the native schema of the enterprise warehouse using **Graph Neural Networks**.

1. **The Graph Data Base (`graph_builder.py`)**
   - We simulate a multi-table environment (Suppliers, Products, Warehouses, Shipments).
   - We parse these raw relational tables directly into a PyTorch Geometric (PyG) `HeteroData` object.
   - Example relations: `(Supplier) -[supplies]-> (Product) -[shipped_to]-> (Warehouse)`.

2. **The "Relational Foundation Model" (`model.py`)**
   - We implement a **Heterogeneous Graph Transformer (HGT)** natively in PyTorch Geometric. 
   - By acting purely on graph structures, the model dynamically passes messages across tables.
   - The model is trained on historical data to perform Link Prediction on the `shipped_to` edges to forecast supply chain delays.

3. **The Agentic AI Layer (`agent.py` & `app.py`)**
   - Business users shouldn't write code or even SQL. Kumo's vision involves **Predictive Query Language (PQL)**.
   - We introduce a **LangChain Orchestration Agent** equipped with a `predict_shipment_delay` tool. 
   - When a user asks *"Will P_40 be delayed to W_5?"*, the LLM Agent parses the intent, triggers the GNN, and provides an explainable result. 

## 🚀 How to Run

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch networkx pandas langchain langchain-openai streamlit faker torch_geometric scikit-learn
```

### 2. Generate the Synthetic Relational DB
This scripts creates the raw CSV files for Suppliers, Warehouses, Products, and Shipments.
```bash
python data_builder.py
```

### 3. Verify Graph Construction
Ensure PyG converts the relational tables into a valid HeteroData graph.
```bash
python graph_builder.py
```

### 4. Train the Heterogeneous Graph Transformer
Trains the HGT architecture and saves the model to `saved_models/hetero_hgt.pth`.
```bash
python model.py
```

### 5. Launch the Agentic Streamlit App
Fire up the UI. Note: You will need to input an OpenAI API key in the Streamlit UI to use the Agentic layer.
```bash
streamlit run app.py
```

## 🤝 To the Kumo Team
This portfolio piece was heavily inspired by the pioneering work of Jure Leskovec and the PyG team. The goal was specifically to demonstrate an understanding of **how to stop writing feature pipelines and start modeling data declaratively**. 
