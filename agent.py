import os
import torch
import torch_geometric.transforms as T
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from model import HeteroLinkPredictor
from graph_builder import load_graph_data

# Load graph data and model globally for the tools
data, supp_map, wh_map, prod_map, shipments_df = load_graph_data()
data = T.ToUndirected()(data)

model = HeteroLinkPredictor(hidden_channels=32, out_channels=32, metadata=data.metadata(), edge_attr_dim=4)
# Use weights_only=True if supported, otherwise standard load
try:
    model.load_state_dict(torch.load("saved_models/hetero_hgt.pth", weights_only=True))
except TypeError:
    model.load_state_dict(torch.load("saved_models/hetero_hgt.pth"))
    
model.eval()

@tool
def predict_shipment_delay(product_id: str, warehouse_id: str) -> str:
    """
    Simulates Kumo.ai Predictive Query Language (PQL).
    Predicts the probability that a shipment of the given product to the given warehouse will be delayed.
    Use this tool whenever the user asks about future supply chain delays, inventory risks, or shipment status.
    """
    # Verify nodes exist in our relational graph
    if product_id not in prod_map: return f"Error: Product '{product_id}' not found in graph database."
    if warehouse_id not in wh_map: return f"Error: Warehouse '{warehouse_id}' not found in graph database."

    p_idx = prod_map[product_id]
    w_idx = wh_map[warehouse_id]

    # Target edge for prediction
    target_edge_index = torch.tensor([[p_idx], [w_idx]], dtype=torch.long)
    # Defaulting weather attribute to [1, 0, 0, 0] (Clear weather)
    target_edge_attr = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float)

    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict, target_edge_index, target_edge_attr)
        prob = torch.sigmoid(logits).item()

    # Glass-box explainability output
    return (
        f"Prediction executed via Kumo-style Hetero Graph Transformer.\n"
        f"Probability of Delay: {prob*100:.1f}%. \n"
        f"[Context: Evaluated via Message Passing subgraph from Node 'product:{product_id}' to Node 'warehouse:{warehouse_id}']"
    )

def get_agent(api_key=None):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [predict_shipment_delay]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an autonomous Supply Chain AI Agent built for a Kumo.ai application. "
                   "You translate natural language business questions into declarative predictive queries on a Relational Foundation Model. "
                   "When asked about delays, extract product_id (e.g. 'P_12') and warehouse_id (e.g. 'W_3') and always use the predict_shipment_delay tool. "
                   "If the user does not provide exact IDs, politely ask them to specify. "
                   "Explain your final answer dynamically, mimicking how a Graph Foundation Model operates over tabular schemas."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("Set OPENAI_API_KEY to test the agent block.")
    else:
        agent_exec = get_agent()
        res = agent_exec.invoke({"input": "What is the probability of delay for product P_40 going to warehouse W_5?"})
        print(res["output"])
