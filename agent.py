import os
import torch
import torch_geometric.transforms as T
import re
from model import HeteroLinkPredictor
from graph_builder import load_graph_data

# Load graph data and model globally
data, supp_map, wh_map, prod_map, shipments_df = load_graph_data()
data = T.ToUndirected()(data)

model = HeteroLinkPredictor(hidden_channels=32, out_channels=32, metadata=data.metadata(), edge_attr_dim=4)
try:
    model.load_state_dict(torch.load("saved_models/hetero_hgt.pth", weights_only=True))
except TypeError:
    model.load_state_dict(torch.load("saved_models/hetero_hgt.pth"))
    
model.eval()

def predict_shipment_delay(product_id: str, warehouse_id: str) -> str:
    if product_id not in prod_map: return f"Error: Product '{product_id}' not found in graph database."
    if warehouse_id not in wh_map: return f"Error: Warehouse '{warehouse_id}' not found in graph database."

    p_idx = prod_map[product_id]
    w_idx = wh_map[warehouse_id]

    target_edge_index = torch.tensor([[p_idx], [w_idx]], dtype=torch.long)
    target_edge_attr = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float)

    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict, target_edge_index, target_edge_attr)
        prob = torch.sigmoid(logits).item()

    return (
        f"✅ **Prediction executed via Kumo-style Hetero Graph Transformer.**\n\n"
        f"**Probability of Delay:** `{prob*100:.1f}%` \n\n"
        f"*[Context: Evaluated via Message Passing subgraph from Node 'product:{product_id}' to Node 'warehouse:{warehouse_id}']* \n\n"
        f"The Relational Foundation Model analyzed all historical edges across Suppliers -> Products -> Warehouses to generate this prediction."
    )

class MockAgentExecutor:
    """A simulated local agent so you don't have to pay for OpenAI API credits!"""
    def invoke(self, inputs: dict):
        text = inputs.get("input", "")
        # Find Product ID
        p_match = re.search(r'(P_\d+)', text, re.IGNORECASE)
        # Find Warehouse ID
        w_match = re.search(r'(W_\d+)', text, re.IGNORECASE)
        
        if p_match and w_match:
            prod_id = p_match.group(1).upper()
            wh_id = w_match.group(1).upper()
            result = predict_shipment_delay(prod_id, wh_id)
            return {"output": f"*(Simulated Local Agent)*\nI have translated your inquiry into a Graph Task for Product **{prod_id}** and Warehouse **{wh_id}**.\n\n{result}"}
        else:
            return {"output": "*(Simulated Local Agent)*\nPlease specify an exact Product ID (e.g., 'P_42') and a Warehouse ID (e.g., 'W_7') in your query so I can run the Graph Neural Network!"}

def get_agent(api_key=None):
    # We are bypassing LangChain to use the Mock Agent so you don't hit Quota Errors!
    return MockAgentExecutor()
