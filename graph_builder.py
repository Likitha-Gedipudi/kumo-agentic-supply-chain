import pandas as pd
import torch
from torch_geometric.data import HeteroData

def standardize(col):
    return (col - col.mean()) / (col.std() + 1e-6)

def load_graph_data(data_dir="data"):
    # Load CSVs
    suppliers_df = pd.read_csv(f"{data_dir}/suppliers.csv")
    warehouses_df = pd.read_csv(f"{data_dir}/warehouses.csv")
    products_df = pd.read_csv(f"{data_dir}/products.csv")
    shipments_df = pd.read_csv(f"{data_dir}/shipments.csv")

    data = HeteroData()

    # Create mapping dictionaries
    supplier_mapping = {id: i for i, id in enumerate(suppliers_df['supplier_id'].unique())}
    warehouse_mapping = {id: i for i, id in enumerate(warehouses_df['warehouse_id'].unique())}
    product_mapping = {id: i for i, id in enumerate(products_df['product_id'].unique())}

    # 1. Supplier Nodes
    suppliers_df['reliability_score'] = standardize(suppliers_df['reliability_score'])
    suppliers_df['active_since'] = standardize(suppliers_df['active_since'])
    supp_features = suppliers_df[['reliability_score', 'active_since']].values
    data['supplier'].x = torch.tensor(supp_features, dtype=torch.float)

    # 2. Warehouse Nodes
    warehouses_df['capacity'] = standardize(warehouses_df['capacity'])
    warehouses_df['current_load'] = standardize(warehouses_df['current_load'])
    wh_features = warehouses_df[['capacity', 'current_load']].values
    data['warehouse'].x = torch.tensor(wh_features, dtype=torch.float)

    # 3. Product Nodes (One-hot encode categories)
    prod_features = pd.get_dummies(products_df['category']).astype(float).values
    data['product'].x = torch.tensor(prod_features, dtype=torch.float)

    # 4. Edges
    # Supplier -> supplies -> Product
    supp_src = [supplier_mapping[id] for id in products_df['supplier_id']]
    prod_dst = [product_mapping[id] for id in products_df['product_id']]
    data['supplier', 'supplies', 'product'].edge_index = torch.tensor([supp_src, prod_dst], dtype=torch.long)

    # Product -> shipped_to -> Warehouse
    prod_ev_src = [product_mapping[id] for id in shipments_df['product_id']]
    wh_ev_dst = [warehouse_mapping[id] for id in shipments_df['warehouse_id']]
    data['product', 'shipped_to', 'warehouse'].edge_index = torch.tensor([prod_ev_src, wh_ev_dst], dtype=torch.long)

    # Edge attributes for 'shipped_to' (weather_conditions)
    weather_dummies = pd.get_dummies(shipments_df['weather_conditions']).astype(float).values
    data['product', 'shipped_to', 'warehouse'].edge_attr = torch.tensor(weather_dummies, dtype=torch.float)

    # Target labels for 'shipped_to' edges (delayed or not)
    data['product', 'shipped_to', 'warehouse'].y = torch.tensor(shipments_df['delayed'].values, dtype=torch.float)

    return data, supplier_mapping, warehouse_mapping, product_mapping, shipments_df

if __name__ == "__main__":
    graph_data, *_ = load_graph_data()
    print("Graph Data Constructed:")
    print(graph_data)
    print("\nNode Types:", graph_data.node_types)
    print("Edge Types:", graph_data.edge_types)
